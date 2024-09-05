"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.spade.normalization import SPADE
from model.layers import myBlock


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        fmiddle = min(fin, fout)
        sem_channel = 192
        self.norm_0 = SPADE(fin, sem_channel)
        self.norm_1 = SPADE(fmiddle, sem_channel)

        self.basicblock_0 = myBlock(fin, fmiddle)
        self.basicblock_1 = myBlock(fmiddle, fout)


    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        dx = self.norm_0(x, seg)
        dx = self.basicblock_0(dx)
        dx = self.norm_1(x, seg)
        dx = self.basicblock_1(dx)   
          
        out = x + dx
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
