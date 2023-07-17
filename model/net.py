import torch
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import bairunsheng.model.function as func
import bairunsheng.model.block as block

sys.path.append("C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng")


class ReconstructionNet(nn.Module):

    # Size:W*H*C
    def __init__(self, img_size, patch_size, length):
        super(ReconstructionNet, self).__init__()

        self.embedding = func.Embeddings(img_size, patch_size, channel=3, length=length)

        self.encoders = nn.ModuleList()
        num = int(img_size[0] // patch_size[0]) * int(img_size[1] // patch_size[1])  # 分成的区块个数，即向量的个数
        for i in range(0, 6):
            self.encoders.append(block.Encoder(10, num, length))
        for i in range(0, 6):
            self.encoders.append(block.Decoder(10, num, length))


    def forward(self, pic0, ev0_raw):
        pic0 =

    def paraInit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
