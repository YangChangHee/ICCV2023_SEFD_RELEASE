from .unet_parts import *

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num,64])

    def forward(self, img_feat, joint_heatmap):
        feat = torch.cat((img_feat, joint_heatmap),1)
        feat = self.conv(feat)
        return feat



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,model_version='1'):
        super(UNet, self).__init__()
        self.model_version=model_version
        if self.model_version=='1':
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.pose2feat=Pose2Feat(30)
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes-1)
        elif self.model_version=="2":
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.pose2feat=Pose2Feat(30)
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1_1 = Up(1024, 512 // factor, bilinear)
            self.up2_1 = Up(512, 256 // factor, bilinear)
            self.up3_1 = Up(256, 128 // factor, bilinear)
            self.up1_2 = Up(1024, 512 // factor, bilinear)
            self.up2_2 = Up(512, 256 // factor, bilinear)
            self.up3_2 = Up(256, 128 // factor, bilinear)
            self.up4_1 = Up(128, 64, bilinear)
            self.up4_2 = Up(128, 64, bilinear)
            self.outc_1 = OutConv(64, n_classes-1)
            self.outc_2 = OutConv(64, n_classes-1)
    def forward(self, x):
        if self.model_version=='1':
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits
        elif self.model_version=='2':
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x_1 = self.up1_1(x5, x4)
            x_1 = self.up2_1(x_1, x3)
            x_1 = self.up3_1(x_1, x2)
            x_1 = self.up4_1(x_1, x1)
            x_2 = self.up1_2(x5, x4)
            x_2 = self.up2_2(x_2, x3)
            x_2 = self.up3_2(x_2, x2)
            x_2 = self.up4_2(x_2, x1)
            logits_1 = self.outc_1(x_1)
            logits_2 = self.outc_2(x_2)
            return logits_1,logits_2