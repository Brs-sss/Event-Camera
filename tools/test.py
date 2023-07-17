import math

import torch
from torchsummary import summary

from bairunsheng.model.net import ReconstructionNet
from bairunsheng.model.data import PictureSet
from bairunsheng.tools.UnNormalize import unNormalize
import bairunsheng.model.function as func
import bairunsheng.model.block as block

from torchstat import stat
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToPILImage
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F

raw = torch.rand(5, 3, 346, 260)

ebd = func.Embeddings(img_size=(346, 260), patch_size=(15, 13), channel=3, length=256)
X = ebd(raw)
print(X.shape)

att = func.MultiHeadAttention(dim=512, head_num=8)
Y = att(X)
print(Y.shape)

mlp = func.PositionWiseFFN(length=512)
R = mlp(Y)
print(R.shape)

raw = torch.rand(5, 3, 346, 260)

ebd = func.Embeddings(img_size=(346, 260), patch_size=(15, 13), channel=3, length=256)
X = ebd(raw)
print(X.shape)

encoder = block.Encoder(5, 460, 512)
Y = encoder(X)
print(Y.shape)

decoder = block.Decoder(5, 460, 512)
g = torch.rand(5, 460, 512)
Z = decoder(Y, g)
print(Z.shape)


alpha = torch.rand(2, 5, 5)
for i in range(0, alpha.shape[1]):
    alpha[0, i, i + 1:alpha.shape[2]] = 0
print(alpha)
'''
x = torch.zeros(3, 4, 5)
y = torch.zeros(3, 4, 5)
for i in range(0, 4):
    x[:, i, :] = i
for i in range(0, 5):
    y[:, :, i] = 2 * (i//2) / 5  # pow(10000, 2*(i//2) / 5)
print(x)
print(y)

r = x / pow(10000, y)
for i in range(0, 5//2):
    r[:, :, 2*i + 1] += math.pi / 2
r = torch,sin(r)
'''


