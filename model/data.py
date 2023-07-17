# coding: utf-8

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from PIL import Image


class PictureSet(Dataset):

    def __init__(self, type, base='C:/Users/MSI-NB/Desktop/Python Projects/srt/datasets/', transform=None,
                 target_transform=None):
        if type == 'train':
            base += 'data_train/'
        elif type == 'verify':
            base += 'test_total/'
        else:
            print('Wrong PictureSet Type!')
            raise AssertionError
        dataInfos = []
        for dataPath in os.listdir(base):
            if os.listdir(base + dataPath).__len__() < 3:
                continue
            total = os.listdir(base + dataPath).__len__()
            for index in range(1, total):
                if index == 10:
                    name = '00' + str(index)
                else:
                    name = '000' + str(index)
                rawPath = base + dataPath + '/image/' + name + '.png'
                tarPath = base + dataPath + '/gt/' + name + '.png'
                # evPath = base + dataPath + '/event/' + name + '.txt'
                tensorPath = base + dataPath + '/voxels/' + name + '.npy'
                dataInfos.append((rawPath, tarPath, tensorPath))

        # 默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0024475607, 0.0055219554, 0.010198287],
                                     std=[0.027039433, 0.033439837, 0.03856222])
            ])
        else:
            self.transform = transform

        # 默认目标转换
        if target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0024475607, 0.0055219554, 0.010198287],
                                     std=[0.027039433, 0.033439837, 0.03856222])
            ])
        else:
            self.target_transform = target_transform
        self.dataInfos = dataInfos

    def __getitem__(self, index):  # 返回顺序：原图、预处理事件、目标图像
        if index >= len(self.dataInfos):
            return None
        rawPath, tarPath, tensorPath = self.dataInfos[index]
        # 原始图像
        rawImg = Image.open(rawPath).convert("RGB")
        if rawImg.size[0] == 260:
            rawImg = rawImg.transpose(Image.ROTATE_90)
        if self.transform is not None:
            rawImg = self.transform(rawImg)
        # 目标图像
        tarImg = Image.open(tarPath).convert("RGB")
        if tarImg.size[0] == 260:
            tarImg = tarImg.transpose(Image.ROTATE_90)
        if self.target_transform is not None:
            tarImg = self.target_transform(tarImg)
        # 预处理事件
        evProcessed = np.load(tensorPath)
        # if np.size(evProcessed, 1) == 346:
        #    evProcessed = np.rot90(evProcessed, 1)
        evProcessed = torch.from_numpy(evProcessed)
        if evProcessed.shape[2] == 260:
            evProcessed = F.rotate(evProcessed, 90, expand=True)
        return rawImg, evProcessed, tarImg

    def __len__(self):
        return len(self.dataInfos)
