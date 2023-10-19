'''
Date: 2023-10-18 22:10:13
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-18 23:07:35
FilePath: /CAC4SSL/code/networks/unet_mso.py
'''
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def contour_attention_func(x):
    min_x = F.max_pool2d(x.softmax(dim=1)*-1, (3, 3), 1, 1) * -1
    max_x = F.max_pool2d(min_x, (3, 3), 1, 1)
    mask = F.relu(max_x-min_x).sum(dim=1).unsqueeze(dim=1)
    mask = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=True)
    return mask


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2, mask=None):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class UpBlock_AG(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock_AG, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2, mask):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        
        x = x + x * mask
        return self.conv(x)
    
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    

class Decoder(nn.Module):
    ''' Decoder with multi scale outputs '''
    def __init__(self, params, emb_num=16):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.proj_dim = emb_num
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)

        self.proj = nn.Sequential(nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=3, padding=1), 
                                    nn.PReLU(), 
                                    nn.Conv2d(self.ft_chns[0], self.proj_dim, kernel_size=3, padding=1))

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        dp4_out_seg = self.out_conv_dp4(x4)
        
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)

        proj_out = self.proj(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg, proj_out
    

class Decoder_Contour(nn.Module):
    ''' 
        Decoder with multi scale outputs  
        Contour
    '''
    def __init__(self, params, att_func=None, emb_num=16):
        super(Decoder_Contour, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.proj_dim = emb_num
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_AG(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock_AG(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock_AG(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock_AG(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)

        self.att_func = att_func
        self.proj = nn.Sequential(nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=3, padding=1), 
                                    nn.PReLU(), 
                                    nn.Conv2d(self.ft_chns[0], self.proj_dim, kernel_size=3, padding=1))

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        dp4_out_seg = self.out_conv_dp4(x4)

        mask = self.att_func(dp4_out_seg)
        x = self.up1(x4, x3, mask.float())
        dp3_out_seg = self.out_conv_dp3(x)
        
        mask = self.att_func(dp3_out_seg)
        x = self.up2(x, x2, mask.float())
        dp2_out_seg = self.out_conv_dp2(x)

        mask = self.att_func(dp2_out_seg)
        x = self.up3(x, x1, mask.float())
        dp1_out_seg = self.out_conv_dp1(x)

        mask = self.att_func(dp1_out_seg)
        x = self.up4(x, x0, mask.float())
        dp0_out_seg = self.out_conv(x)

        proj_out = self.proj(x)

        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg, proj_out
   

class CACNet2d(nn.Module):

    def __init__(self, in_chns, class_num, emb_num=16):
        super(CACNet2d, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
            }

        params_contour = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'
            }
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params, emb_num=emb_num)
        self.decoder_contour = Decoder_Contour(params_contour, att_func=contour_attention_func, emb_num=emb_num)

    def forward(self, x):

        feature = self.encoder(x)
    
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg, proj = self.decoder(feature)

        dp0_out_seg_contour, dp1_out_seg_contour, dp2_out_seg_contour, dp3_out_seg_contour, dp4_out_seg_contour, proj_contour = self.decoder_contour(feature)

        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg, dp0_out_seg_contour, dp1_out_seg_contour, dp2_out_seg_contour, dp3_out_seg_contour, dp4_out_seg_contour, proj, proj_contour

    def inference(self, x):
        out = self.forward(x)
        return out[0]
    