# coding: utf-8

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class PictureSet(Dataset):

    def __init__(self, txt_path, transform=None, target_transform=None):
        dataInfos = []
        base = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/datasets/data_train'
        for dataPath in os.listdir(base):
            for index in range(1, 11):
                if index == 10:
                    name = '00' + str(index)
                else:
                    name = '000' + str(index)
                rawPath = base + dataPath + '/image/' + name + '.png'
                tarPath = base + dataPath + '/gt/' + name + '.png'
                evPath = base + dataPath + '/event/' + name + '.txt'
                tensorPath = base + dataPath + '/voxels/' + name + '.npy'
                dataInfos.append((rawPath, tarPath, evPath, tensorPath))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4948052, 0.48568845, 0.44682974], std=[0.24580306, 0.24236229, 0.2603115])  # 常用标准化
        ])
        self.target_transform = target_transform
        self.dataInfos = dataInfos

    def __getitem__(self, index):
        if index >= len(self.dataInfos):
            return None
        rawPath, tarPath, evPath, tensorPath = self.dataInfos[index]
        rawImg = Img

    def __len__(self):
        return len(self.dataInfos)



