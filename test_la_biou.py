'''
Date: 2023-06-24 03:15:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-06-24 11:20:56
FilePath: /SSL/MC/la_iou.py
'''
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from boundary_iou import boundary_iou

root = 'model/LA_cacnet_sample_50_4_labeled/cacnet3d_emb_64_predictions'
# root = 'model/LA_cacnet_sample_50_8_labeled/cacnet3d_emb_64_predictions'
# root = 'model/LA_cacnet_sample_50_16_labeled/cacnet3d_emb_64_predictions'

filelist = []
iou_list1 = []

for name in os.listdir(root):
    if name.endswith('_gt.nii.gz'):
        filelist.append(name)

for name in tqdm(filelist):
    gt = nib.load(os.path.join(root, name)).get_fdata()
    pred = nib.load(os.path.join(root, name.replace('_gt.nii.gz', '_pred.nii.gz'))).get_fdata()
    
    iou1 = []
    for idx in range(gt.shape[-1]):
        sgt = gt[:, :, idx]
        spred = pred[:, :, idx]
        
        m1 = boundary_iou(sgt==1, spred==1)

        if np.isnan(m1):
            m1 = 0

        iou1.append(m1)
    
    iou_list1.append(np.mean(iou1))

mean1 = np.mean(iou_list1)

print('Mean:', mean1*100)
print('Mean: %.2f'%(mean1*100))