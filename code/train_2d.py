# encoding=utf-8
import argparse
import logging
import os
import random
import shutil
import sys
import time

import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.dataset import *

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, val_2d
from utils import contrast_loss

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MCNet_avgpseudo', help='experiment_name')
parser.add_argument('--model', type=str, default='mcnet2d_v1', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=14, help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')

args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, 
                            split="train", 
                            num=None, 
                            transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = MultiEpochsDataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    valloader = MultiEpochsDataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    contrast_loss_fn = contrast_loss.Contrast()
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    sample_num = int(re.findall('sample_(\d+)', args.exp)[0])
    proj_mask_idx = 0
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            single_out_num = 5

            label_batches = [label_batch.unsqueeze(1)]
            for _ in range(1, single_out_num):
                label_batches.append(F.interpolate(label_batches[-1], scale_factor=0.5, recompute_scale_factor=False))

            model.train()
            outputs = model(volume_batch)
            proj_list = outputs[-2:]
            outputs = outputs[:-2]

            num_outputs = len(outputs) // single_out_num

            y_ori = []
            y_ori_softmax = []
            y_pseudo_label = []
            y_mask = []

            for idx in range(single_out_num):
                y_ori.append(torch.zeros((num_outputs,)+outputs[idx].shape, device=volume_batch.device))
                y_ori_softmax.append(torch.zeros((num_outputs,)+outputs[idx].shape, device=volume_batch.device))

            loss_seg_dice = 0 

            for idx in range(num_outputs):
                for sub_idx in range(single_out_num):
                    true_idx = idx * single_out_num + sub_idx
                    y = outputs[true_idx][:labeled_bs,...]
                    y_prob = F.softmax(y, dim=1)
                    loss_seg_dice += dice_loss(y_prob, label_batches[sub_idx][:labeled_bs])

                    y_all = outputs[true_idx]
                    y_ori[sub_idx][idx] = y_all
                    y_prob_all = F.softmax(y_all, dim=1)
                    y_ori_softmax[sub_idx][idx] = y_prob_all

            for idx in range(single_out_num):
                out_0 = y_ori[idx][0].argmax(dim=1)
                out_1 = y_ori[idx][1].argmax(dim=1)
                mask = (out_0==out_1).float()
                y_mask.append(mask)
                y_pseudo_label.append(y_ori_softmax[idx].mean(dim=0).argmax(dim=1))
            
            loss_consist = 0
            for i in range(single_out_num):
                for j in range(num_outputs):
                    loss_consist += ce_loss(y_ori[i][j], y_pseudo_label[i].long())
                    loss_consist += dice_loss(y_ori_softmax[i][j], y_pseudo_label[i].unsqueeze(1))
                
            loss_contrast = 0
            for i in range(num_outputs):
                loss_contrast += contrast_loss_fn(proj_list, i, y_pseudo_label[0], mask=y_mask[proj_mask_idx], sample_num=sample_num)

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            contrast_weight = consistency_weight
            
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist + contrast_weight * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('%s %s iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f, loss_contrast: %03f' % (args.exp, args.model, iter_num, loss, loss_seg_dice, loss_consist, loss_contrast))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
