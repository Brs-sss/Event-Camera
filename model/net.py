import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append("C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng")


class ReconstructionNet(nn.Module):

    # Size:W*H*C
    def __init__(self):
        super(ReconstructionNet, self).__init__()

        # 每一层提取不同规模特征的卷积算子
        in_channel = [6, 18, 36, 18, 18]
        out_channel = [6, 9, 18, 18, 18]

        self._1_1_conv = nn.ModuleList()
        self._3_3_conv = nn.ModuleList()
        self._5_5_conv = nn.ModuleList()
        self._ReLu = nn.ModuleList()

        for i in range(0, 5):
            self._1_1_conv.append(
                nn.Conv2d(in_channels=in_channel[i], out_channels=out_channel[i], kernel_size=(1, 1), padding=0)
            )
            self._3_3_conv.append(
                nn.Conv2d(in_channels=in_channel[i], out_channels=out_channel[i], kernel_size=(3, 3), padding=1)
            )
            mediate_channel = int((in_channel[i] + out_channel[i]) / 2)
            self._5_5_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channel[i], out_channels=mediate_channel, kernel_size=(3, 3), padding=1),
                nn.Conv2d(in_channels=mediate_channel, out_channels=out_channel[i], kernel_size=(3, 3), padding=1)
            ))
            self._ReLu.append(nn.Sequential(
                nn.BatchNorm2d(num_features=out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))

        # 两次循环中的33卷积
        self.cir1 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=(3, 3), padding=1)
        self.cir2 = nn.Sequential(
                nn.Conv2d(in_channels=18, out_channels=18, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=out_channel[i], affine=True),
                nn.ReLU(inplace=True))
        self.cir3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=(3, 3), padding=1)
        self.cir4 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=18, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_channel[i], affine=True),
            nn.ReLU(inplace=True))

        # 对事件流预处理tensor提取特征过程的卷积算子
        ev_in_channel = [5, 10, 15]
        ev_out_channel = [1, 2, 3]

        self.ev_1_1_conv = nn.ModuleList()
        self.ev_3_3_conv = nn.ModuleList()
        self.ev_5_5_conv = nn.ModuleList()
        self.ev_merge_ReLu = nn.ModuleList()
        self._5_to_10 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(1, 1), padding=0)
        self._10_to_15 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(1, 1), padding=0)

        for i in range(0, 3):
            self.ev_1_1_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=ev_out_channel[i], kernel_size=(1, 1), padding=0),
            ))
            self.ev_3_3_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=ev_out_channel[i], kernel_size=(3, 3), padding=1),
            ))
            mediate_channel = int((ev_in_channel[i] + ev_out_channel[i])/2)
            self.ev_5_5_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=mediate_channel, kernel_size=(3, 3), padding=1),
                nn.Conv2d(in_channels=mediate_channel, out_channels=ev_out_channel[i], kernel_size=(3, 3), padding=1),
            ))
            self.ev_merge_ReLu.append(nn.Sequential(
                nn.BatchNorm2d(num_features=3 * ev_out_channel[i], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=3*ev_out_channel[i], out_channels=3*ev_out_channel[i], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=3*ev_out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))

        # 最后处理
        self.picFinalPro1 = nn.Conv2d(in_channels=18, out_channels=9, kernel_size=(3, 3), padding=1)
        self.picFinalPro2 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=(3, 3), padding=1)

    def forward(self, pic0, ev0_raw):
        # 图像与事件融合过程
        ev0_1_1_conv = self.ev_1_1_conv[0](ev0_raw)
        ev0_3_3_conv = self.ev_3_3_conv[0](ev0_raw)
        ev0_5_5_conv = self.ev_5_5_conv[0](ev0_raw)
        ev0 = torch.cat((ev0_1_1_conv, ev0_3_3_conv, ev0_5_5_conv), dim=1)
        ev0 = self.ev_merge_ReLu[0](ev0)
        pic0 = torch.cat((ev0, pic0), dim=1)
        pic0_1_1_conv = self._1_1_conv[0](pic0)
        pic0_3_3_conv = self._3_3_conv[0](pic0)
        pic0_5_5_conv = self._5_5_conv[0](pic0)
        pic0_ = (pic0_1_1_conv + pic0_3_3_conv + pic0_5_5_conv) / 3
        pic1 = self._ReLu[0](pic0_)

        ev1_raw = self._5_to_10(ev0_raw)
        ev1_1_1_conv = self.ev_1_1_conv[1](ev1_raw)
        ev1_3_3_conv = self.ev_3_3_conv[1](ev1_raw)
        ev1_5_5_conv = self.ev_5_5_conv[1](ev1_raw)
        ev1 = torch.cat((ev1_1_1_conv, ev1_3_3_conv, ev1_5_5_conv), dim=1)
        ev1 = self.ev_merge_ReLu[1](ev1)
        pic1 = torch.cat((ev1, pic1, pic0), dim=1)
        pic1_1_1_conv = self._1_1_conv[1](pic1)
        pic1_3_3_conv = self._3_3_conv[1](pic1)
        pic1_5_5_conv = self._5_5_conv[1](pic1)
        pic1_ = (pic1_1_1_conv + pic1_3_3_conv + pic1_5_5_conv) / 3
        pic2 = self._ReLu[1](pic1_)

        ev2_raw = self._10_to_15(ev1_raw)
        ev2_1_1_conv = self.ev_1_1_conv[2](ev2_raw)
        ev2_3_3_conv = self.ev_3_3_conv[2](ev2_raw)
        ev2_5_5_conv = self.ev_5_5_conv[2](ev2_raw)
        ev2 = torch.cat((ev2_1_1_conv, ev2_3_3_conv, ev2_5_5_conv), dim=1)
        ev2 = self.ev_merge_ReLu[2](ev2)
        pic2 = torch.cat((ev2, pic2, pic1), dim=1)
        pic2_1_1_conv = self._1_1_conv[2](pic2)
        pic2_3_3_conv = self._3_3_conv[2](pic2)
        pic2_5_5_conv = self._5_5_conv[2](pic2)
        pic2_ = (pic2_1_1_conv + pic2_3_3_conv + pic2_5_5_conv) / 3
        picpro0 = self._ReLu[2](pic2_)

        picpro1_1_1_conv = self._1_1_conv[3](picpro0)
        picpro1_3_3_conv = self._3_3_conv[3](picpro0)
        picpro1_5_5_conv = self._5_5_conv[3](picpro0)
        picpro1 = (picpro1_1_1_conv + picpro1_3_3_conv + picpro1_5_5_conv) / 3
        picpro1 = (picpro1 + picpro0) / 2
        picpro2 = self.cir1(picpro1)
        picpro2 = (picpro2 + 2*picpro1) / 3
        picpro3 = self.cir2(picpro2)

        picpro4_1_1_conv = self._1_1_conv[4](picpro3)
        picpro4_3_3_conv = self._3_3_conv[4](picpro3)
        picpro4_5_5_conv = self._5_5_conv[4](picpro3)
        picpro4 = (picpro4_1_1_conv + picpro4_3_3_conv + picpro4_5_5_conv) / 3
        picpro4 = (picpro4 + picpro3) / 2
        picpro5 = self.cir3(picpro4)
        picpro5 = (picpro5 + 2*picpro4) / 3
        pic = self.cir4(picpro5)

        result = self.picFinalPro1(pic)
        result = self.picFinalPro2(result)

        return result

    def paraInit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
