'''
Date: 2023-06-24 03:15:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-06-24 04:16:29
FilePath: /SSL/MC/acdc_iou.py
'''
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from boundary_iou import boundary_iou

root = 'model/ACDC_CACNet_sample_50_3_labeled/cacnet2d_emb_64_predictions'
# root = 'model/ACDC_CACNet_sample_50_7_labeled/cacnet2d_emb_64_predictions'
# root = 'model/ACDC_CACNet_sample_50_14_labeled/cacnet2d_emb_64_predictions'

filelist = []
iou_list1 = []
iou_list2 = []
iou_list3 = []

for name in os.listdir(root):
    if name.endswith('_gt.nii.gz'):
        filelist.append(name)

for name in tqdm(filelist):
    gt = nib.load(os.path.join(root, name)).get_fdata()
    pred = nib.load(os.path.join(root, name.replace('_gt.nii.gz', '_pred.nii.gz'))).get_fdata()
    
    iou1 = []
    iou2 = []
    iou3 = []
    for idx in range(gt.shape[-1]):
        sgt = gt[:, :, idx]
        spred = pred[:, :, idx]
        
        m1 = boundary_iou(sgt==1, spred==1)
        m2 = boundary_iou(sgt==2, spred==2)
        m3 = boundary_iou(sgt==3, spred==3)

        if np.isnan(m1):
            m1 = 0
        if np.isnan(m2):
            m2 = 0
        if np.isnan(m3):
            m3 = 0

        iou1.append(m1)
        iou2.append(m2)
        iou3.append(m3)
        
    iou_list1.append(np.mean(iou1))
    iou_list2.append(np.mean(iou2))
    iou_list3.append(np.mean(iou3))        

mean1 = np.mean(iou_list1)
mean2 = np.mean(iou_list2)
mean3 = np.mean(iou_list3)

print("m1:", mean1, "m2:", mean2, "m3:", mean3)
print('Mean:', (mean1+mean2+mean3)/3*100)
print('Mean: %.2f'%((mean1+mean2+mean3)/3*100))
