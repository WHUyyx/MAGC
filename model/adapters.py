import torch
import torch.nn as nn
from model.layers import myResnetBlock


class Adapter_XL(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=2):
        super(Adapter_XL, self).__init__()

        self.cond_encoder =  nn.Sequential( # 3 256 256 -> 192 32 32
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
        )    
        self.channels = channels
        self.nums_rb = nums_rb
        self.first_rb = myResnetBlock(192 + 4, channels[0])    
        self.body = []

        for i in range(len(channels) - 1):
            channel_in = channels[i]
            channel_out = channels[i+1]
            self.body.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))
            self.body.append(myResnetBlock(channel_out, channel_out))
            self.body.append(myResnetBlock(channel_out, channel_out))
        self.body = nn.ModuleList(self.body)

    def forward(self, x, z_noise):
        # x: b 3 256 256
        semantic_feature = self.cond_encoder(x) # b 192 32 32
        ms_features = []
        latent = torch.cat((semantic_feature, z_noise),dim=1)
        latent = self.first_rb(latent) # b 320 32 32
        ms_features.append(latent) # b 320 32 32

        for i in range(len(self.channels)-1):
            for j in range(self.nums_rb+1): 
                idx = i * self.nums_rb + j
                latent = self.body[idx](latent)
            ms_features.append(latent)
        return ms_features

