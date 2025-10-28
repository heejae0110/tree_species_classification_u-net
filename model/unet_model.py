#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A compact U-Net for 66-channel inputs and N classes."""
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels: int = 66, num_classes: int = 10):
        super().__init__()

        def CBR2d(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1_1 = CBR2d(in_channels, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024)
        self.dec5_1 = CBR2d(1024, 512)

        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4_2 = CBR2d(1024, 512)
        self.dec4_1 = CBR2d(512, 256)

        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 128)

        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 64)

        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1_2(self.enc1_1(x)); p1 = self.pool1(e1)
        e2 = self.enc2_2(self.enc2_1(p1)); p2 = self.pool2(e2)
        e3 = self.enc3_2(self.enc3_1(p2)); p3 = self.pool3(e3)
        e4 = self.enc4_2(self.enc4_1(p3)); p4 = self.pool4(e4)

        b = self.dec5_1(self.enc5_1(p4))

        u4 = self.unpool4(b); d4 = self.dec4_1(self.dec4_2(torch.cat([u4, e4], dim=1)))
        u3 = self.unpool3(d4); d3 = self.dec3_1(self.dec3_2(torch.cat([u3, e3], dim=1)))
        u2 = self.unpool2(d3); d2 = self.dec2_1(self.dec2_2(torch.cat([u2, e2], dim=1)))
        u1 = self.unpool1(d2); d1 = self.dec1_1(self.dec1_2(torch.cat([u1, e1], dim=1)))

        return self.head(d1)
