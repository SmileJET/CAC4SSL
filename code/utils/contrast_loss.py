'''
Date: 2023-10-18 22:10:13
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-18 22:58:47
FilePath: /CAC4SSL/code/utils/contrast_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast, self).__init__()
        self.tau = temperature

    def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        batch_size = mask.shape[0]
        loss = 0
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            try:
                proj = proj_list[i].permute(0, 2, 3, 1)
            except:
                proj = proj_list[i].permute(0, 2, 3, 4, 1)

            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                curr_proj = F.normalize(proj, dim=-1)
            else:
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        mask = mask.contiguous().view(batch_size, -1).long()
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]

            hard_indices = fn_mask_.nonzero()

            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num, num_hard)

            hard_perm = torch.randperm(num_hard)
            hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            indices = hard_indices

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)


            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

        return loss / batch_size
    