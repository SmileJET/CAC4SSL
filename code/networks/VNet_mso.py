import torch
from torch import nn
import torch.nn.functional as F


def contour_attention_func(x):
    min_x = F.max_pool3d(x.softmax(dim=1)*-1, (3, 3, 3), 1, 1) * -1
    max_x = F.max_pool3d(min_x, (3, 3, 3), 1, 1)
    mask = F.relu(max_x-min_x).sum(dim=1).unsqueeze(dim=1)
    mask = F.interpolate(mask, scale_factor=2, mode="trilinear", align_corners=True)
    return mask


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    

class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0, emb_num=16):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_8 = nn.Conv3d(n_filters*2, n_classes, 1, padding=0)
        self.out_conv_7 = nn.Conv3d(n_filters*4, n_classes, 1, padding=0)
        self.out_conv_6 = nn.Conv3d(n_filters*8, n_classes, 1, padding=0)
        self.out_conv_5 = nn.Conv3d(n_filters*16, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.proj = nn.Sequential(nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv3d(n_filters, emb_num, kernel_size=3, padding=1)
                                  )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
            x8 = self.dropout(x8)
            x7 = self.dropout(x7)
            x6 = self.dropout(x6)
            x5 = self.dropout(x5)
        out_seg = self.out_conv(x9)
        out_8 = self.out_conv_8(x8)
        out_7 = self.out_conv_7(x7)
        out_6 = self.out_conv_6(x6)
        out_5 = self.out_conv_5(x5)

        proj_out = self.proj(x9)
        
        return out_seg, out_8, out_7, out_6, out_5, proj_out
    

class Decoder_Contour(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0, attention_func=None, emb_num=16):
        super(Decoder_Contour, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_8 = nn.Conv3d(n_filters*2, n_classes, 1, padding=0)
        self.out_conv_7 = nn.Conv3d(n_filters*4, n_classes, 1, padding=0)
        self.out_conv_6 = nn.Conv3d(n_filters*8, n_classes, 1, padding=0)
        self.out_conv_5 = nn.Conv3d(n_filters*16, n_classes, 1, padding=0)

        self.attention_func = attention_func
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.proj = nn.Sequential(nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv3d(n_filters, emb_num, kernel_size=3, padding=1)
                                  )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_out = x5
        if self.has_dropout:
            x5_out = self.dropout(x5_out)
        out_5 = self.out_conv_5(x5)
        mask_5 = self.attention_func(out_5)
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4
        x5_up = x5_up + mask_5 * x5_up

        x6 = self.block_six(x5_up)
        x6_out = x6
        if self.has_dropout:
            x6_out = self.dropout(x6_out)
        out_6 = self.out_conv_6(x6)        
        mask_6 = self.attention_func(out_6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        x6_up = x6_up + mask_6 * x6_up

        x7 = self.block_seven(x6_up)
        x7_out = x7
        if self.has_dropout:
            x7_out = self.dropout(x7_out)
        out_7 = self.out_conv_7(x7)        
        mask_7 = self.attention_func(out_7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        x7_up = x7_up + mask_7 * x7_up

        x8 = self.block_eight(x7_up)
        x8_out = x8
        if self.has_dropout:
            x8_out = self.dropout(x8_out)
        out_8 = self.out_conv_8(x8)
        mask_8 = self.attention_func(out_8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x8_up = x8_up + mask_8 * x8_up

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        proj_out = self.proj(x9)

        return out_seg, out_8, out_7, out_6, out_5, proj_out
    

class CACNet3d(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, emb_num=16):
        super(CACNet3d, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0, emb_num=emb_num)
        self.decoder_contour = Decoder_Contour(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0, contour_attention_func, emb_num=emb_num)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2, out_seg3, out_seg4, out_seg5, proj_out1 = self.decoder(features)
        out_contour_seg1, out_contour_seg2, out_contour_seg3, out_contour_seg4, out_contour_seg5, proj_contour_out1 = self.decoder_contour(features)
        
        return out_seg1, out_seg2, out_seg3, out_seg4, out_seg5, out_contour_seg1, out_contour_seg2, out_contour_seg3, out_contour_seg4, out_contour_seg5, proj_out1, proj_contour_out1
        