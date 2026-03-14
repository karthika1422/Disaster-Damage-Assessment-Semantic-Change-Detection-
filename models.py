# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        
        self.enc1_1 = ConvBlock(in_ch, base_ch)
        self.enc1_2 = ConvBlock(base_ch, base_ch*2)
        self.enc1_3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2,2)

        self.bottleneck = ConvBlock(base_ch*8, base_ch*8)

        self.up3 = UpBlock(base_ch*8, base_ch*4)
        self.up2 = UpBlock(base_ch*4, base_ch*2)
        self.up1 = UpBlock(base_ch*2, base_ch)
        self.final_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def encode_branch(self, x):
        x1 = self.enc1_1(x)
        x2 = self.enc1_2(self.pool(x1))
        x3 = self.enc1_3(self.pool(x2))
        return x1, x2, x3

    def forward(self, im1, im2):

        a1,a2,a3 = self.encode_branch(im1)
        b1,b2,b3 = self.encode_branch(im2)
        fuse = torch.cat([a3, b3], dim=1)
        bott = self.bottleneck(self.pool(fuse))
        d3 = self.up3(bott, torch.cat([a3,b3], dim=1))
        d2 = self.up2(d3, torch.cat([a2,b2], dim=1))
        d1 = self.up1(d2, torch.cat([a1,b1], dim=1))
        out = self.final_conv(d1)
        out = torch.sigmoid(out)
        return out
