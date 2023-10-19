# encoding=utf-8
'''
Author: ll
Date: 2022-12-06 19:30:58
LastEditTime: 2023-10-18 23:46:07
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Description: 
FilePath: /CAC4SSL/code/networks/net_factory.py
'''

import re

from networks.unet_mso import CACNet2d

from networks.VNet_mso import CACNet3d

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    emb_num = re.findall('emb_(\d+)', net_type)
    if len(emb_num) > 0:
        emb_num = int(emb_num[0])
    else:
        emb_num = 0
    
    elif net_type.startswith('cacnet2d'):
        net = CACNet2d(in_chns=in_chns, class_num=class_num, emb_num=emb_num).cuda()

    elif net_type.startswith('cacnet3d') and mode == "train":
        net = CACNet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, emb_num=emb_num).cuda()
    elif net_type.startswith('cacnet3d') and mode == "test":
        net = CACNet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, emb_num=emb_num).cuda()

    return net
