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

        # 对png图片进行downward操作
        self.picDown1 = (nn.Conv2d(in_channels=3 + 5, out_channels=3, kernel_size=(3, 3), padding=1),
                         nn.Sequential(
                             nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(4, 4), padding=1,
                                       stride=(2, 2)),
                             nn.BatchNorm2d(24, affine=True),
                             nn.ReLU(inplace=True)))

        self.picDown2 = (nn.Conv2d(in_channels=24 + 10, out_channels=24, kernel_size=(3, 3), padding=1),
                         nn.Sequential(
                             nn.Conv2d(in_channels=24, out_channels=72, kernel_size=(4, 4), padding=1,
                                       stride=(2, 2)),
                             nn.BatchNorm2d(72, affine=True),
                             nn.ReLU(inplace=True)))

        self.picDown3 = (nn.Conv2d(in_channels=72 + 24, out_channels=72, kernel_size=(3, 3), padding=1),
                         nn.Sequential(
                             nn.Conv2d(in_channels=72, out_channels=144, kernel_size=(4, 4), padding=1,
                                       stride=(2, 2)),
                             nn.BatchNorm2d(144, affine=True),
                             nn.ReLU(inplace=True)))

        self.picDown4 = (nn.Conv2d(in_channels=144 + 48, out_channels=144, kernel_size=(3, 3), padding=1),
                         nn.Sequential(
                             nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(4, 4), padding=1,
                                       stride=(2, 2)),
                             nn.BatchNorm2d(288, affine=True),
                             nn.ReLU(inplace=True)))

        # 对预处理图片进行操作
        self.firstPros = []
        for i in range(0, 5):
            self.firstPros.append(nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3), padding=1))

        self.evDown1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(10, affine=True),
            nn.ReLU(inplace=True)
        )

        self.secondPros = []
        for i in range(0, 4):
            self.secondPros.append(nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1))

        self.evDown2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=24, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(24, affine=True),
            nn.ReLU(inplace=True)
        )

        self.thirdPros = []
        for i in range(0, 3):
            self.thirdPros.append(nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1))

        self.evDown3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(48, affine=True),
            nn.ReLU(inplace=True)
        )

        self.forthPros = []
        for i in range(0, 2):
            self.forthPros.append(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1))

        self.evDown4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(96, affine=True),
            nn.ReLU(inplace=True))

        self.fifthPro = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding=1)
        # 对最下层图片进行处理
        self.bottomProcess = nn.Sequential(
            nn.Conv2d(in_channels=288 + 96, out_channels=288, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(288, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=288, out_channels=288, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(288, affine=True),
            nn.ReLU(inplace=True)
        )

        # upward
        self.picUp4 = (nn.Sequential(nn.ConvTranspose2d(in_channels=288, out_channels=144, kernel_size=4, stride=2,
                                                        padding=1, output_padding=(0, 1)),
                                     nn.BatchNorm2d(144, affine=True),
                                     nn.ReLU(inplace=True)),
                       nn.Sequential(nn.Conv2d(in_channels=288, out_channels=144, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(144, affine=True),
                                     nn.ReLU(inplace=True)))

        self.picUp3 = (nn.Sequential(nn.ConvTranspose2d(in_channels=144, out_channels=72, kernel_size=4, stride=2,
                                                        padding=1, output_padding=(1, 0)),
                                     nn.BatchNorm2d(72, affine=True),
                                     nn.ReLU(inplace=True)),
                       nn.Sequential(nn.Conv2d(in_channels=144, out_channels=72, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(72, affine=True),
                                     nn.ReLU(inplace=True)))

        self.picUp2 = (nn.Sequential(nn.ConvTranspose2d(in_channels=72, out_channels=24, kernel_size=4, stride=2,
                                                        padding=1, output_padding=(0, 1)),
                                     nn.BatchNorm2d(24, affine=True),
                                     nn.ReLU(inplace=True)),
                       nn.Sequential(nn.Conv2d(in_channels=48, out_channels=24, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(24, affine=True),
                                     nn.ReLU(inplace=True)))

        self.picUp1 = (nn.Sequential(nn.ConvTranspose2d(in_channels=24, out_channels=3, kernel_size=4, stride=2,
                                                        padding=1, output_padding=(0, 0)),
                                     nn.BatchNorm2d(3, affine=True),
                                     nn.ReLU(inplace=True)),
                       nn.Sequential(nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(3, affine=True),
                                     nn.ReLU(inplace=True)))

        # 最后处理
        self.picFinalPro = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)

    def forward(self, pic1_raw, ev1):
        # downward过程
        ev2 = self.evDown1(ev1)
        for pro in self.firstPros:
            ev1 = pro(ev1)
        pic1_raw = torch.cat((pic1_raw, ev1), dim=1)  # dim需检查
        pic1 = self.picDown1[0](pic1_raw)

        pic2_raw = self.picDown1[1](pic1)
        ev3 = self.evDown2(ev2)
        for pro in self.secondPros:
            ev2 = pro(ev2)
        pic2_raw = torch.cat((pic2_raw, ev2), dim=1)
        pic2 = self.picDown2[0](pic2_raw)

        pic3_raw = self.picDown2[1](pic2)
        ev4 = self.evDown3(ev3)
        for pro in self.thirdPros:
            ev3 = pro(ev3)
        pic3_raw = torch.cat((pic3_raw, ev3), dim=1)
        pic3 = self.picDown3[0](pic3_raw)

        pic4_raw = self.picDown3[1](pic3)
        ev5 = self.evDown4(ev4)
        for pro in self.forthPros:
            ev4 = pro(ev4)
        pic4_raw = torch.cat((pic4_raw, ev4), dim=1)
        pic4 = self.picDown4[0](pic4_raw)

        pic5 = self.picDown4[1](pic4)
        ev5 = self.fifthPro(ev5)
        pic5 = torch.cat((pic5, ev5), dim=1)

        # 底层处理
        pic5_pro = self.bottomProcess(pic5)

        # upward 过程
        pic4_pro = self.picUp4[0](pic5_pro)
        pic4_pro = torch.cat((pic4, pic4_pro), dim=1)
        pic4_pro = self.picUp4[1](pic4_pro)

        pic3_pro = self.picUp3[0](pic4_pro)
        pic3_pro = torch.cat((pic3, pic3_pro), dim=1)
        pic3_pro = self.picUp3[1](pic3_pro)

        pic2_pro = self.picUp2[0](pic3_pro)
        pic2_pro = torch.cat((pic2, pic2_pro), dim=1)
        pic2_pro = self.picUp2[1](pic2_pro)

        pic1_pro = self.picUp1[0](pic2_pro)
        pic1_pro = torch.cat((pic1, pic1_pro), dim=1)
        pic1_pro = self.picUp1[1](pic1_pro)

        result = self.picFinalPro(pic1_pro)
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
