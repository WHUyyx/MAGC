import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
    

class myBlock(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 5 if large_filter else 3, padding=2 if large_filter else 1), LayerNorm(dim_out), nn.LeakyReLU()
        )
    def forward(self, x):
        return self.block(x)
    

class myResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False):
        super().__init__()
        self.block1 = myBlock(dim, dim_out, large_filter)
        self.block2 = myBlock(dim_out, dim_out, large_filter)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class myDownsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    

class myUpsample(nn.Module):
    def __init__(self, dim_in, upscale_factor=2):
        super().__init__()
        self.pix_shuffle = nn.Sequential(
            conv1x1(dim_in, dim_in * upscale_factor * upscale_factor),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        return self.pix_shuffle(x)
