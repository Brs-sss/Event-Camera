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

        # 对阴暗图片提取特征过程中每一层提取不同规模特征的卷积算子
        in_channel = [6, 24, 96, 192, 384]
        out_channel = [4, 16, 32, 64, 128]

        self._1_1_conv = []
        self._3_3_conv = []
        self._5_5_conv = []

        for i in range(0, 5):
            self._1_1_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channel[i], out_channels=out_channel[i], kernel_size=(1, 1), padding=0),
                nn.BatchNorm2d(num_features=out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))
            self._3_3_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channel[i], out_channels=out_channel[i], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))
            mediate_channel = int((in_channel[i] + out_channel[i]) / 2)
            self._5_5_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channel[i], out_channels=mediate_channel, kernel_size=(3, 3), padding=1),
                nn.Conv2d(in_channels=mediate_channel, out_channels=out_channel[i], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))

        # 对阴暗图片进行下采样的卷积算子
        # 可能在down开始添加一步融会增进学习能力
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=12, affine=True),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=48, affine=True),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=96, affine=True),
            nn.ReLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=192, affine=True),
            nn.ReLU(inplace=True)
        )

        # 对事件流预处理tensor提取特征过程中每一层提取不同规模特征的卷积算子
        ev_in_channel = [5, 16, 32, 64, 128]
        ev_out_channel = [1, 4, 16, 32, 64]

        self.ev_1_1_conv = []
        self.ev_3_3_conv = []
        self.ev_5_5_conv = []

        for i in range(0, 5):
            self.ev_1_1_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=ev_out_channel[i], kernel_size=(1, 1), padding=0),
                nn.BatchNorm2d(num_features=ev_out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))
            self.ev_3_3_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=ev_out_channel[i], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=ev_out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))
            mediate_channel = int((ev_in_channel[i] + ev_out_channel[i])/2)
            self.ev_5_5_conv.append(nn.Sequential(
                nn.Conv2d(in_channels=ev_in_channel[i], out_channels=mediate_channel, kernel_size=(3, 3), padding=1),
                nn.Conv2d(in_channels=mediate_channel, out_channels=ev_out_channel[i], kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=ev_out_channel[i], affine=True),
                nn.ReLU(inplace=True)
            ))

        # 对事件流预处理tensor进行下采样的卷积算子
        # 可能删除down的一步融和会增进学习能力
        self.evDown1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=16, affine=True),
            nn.ReLU(inplace=True)
        )
        self.evDown2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=32, affine=True),
            nn.ReLU(inplace=True)
        )
        self.evDown3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.evDown4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=128, affine=True),
            nn.ReLU(inplace=True)
        )

        # 对最下层图片进行处理
        self.bottomProcess = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=384, affine=True),
            nn.ReLU(inplace=True)
        )

        # 上采样卷积算子
        # 可能交换conv与tranposed顺序会产生影响
        self.up4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=4, stride=2,
                               padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=192, affine=True),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2,
                               padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(num_features=96, affine=True),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=48, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=4, stride=2,
                               padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=48, affine=True),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=12, kernel_size=(3, 3), padding=1),
            nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=4, stride=2,
                               padding=1, output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=12, affine=True),
            nn.ReLU(inplace=True)
        )

        # 最后处理
        self.picFinalPro1 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(3, 3), padding=1)
        self.picFinalPro2 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=(3, 3), padding=1)

    def forward(self, pic1_raw, ev):
        # 下采样过程
        _1_1_conv_ev1 = self.ev_1_1_conv[0](ev)
        _3_3_conv_ev1 = self.ev_3_3_conv[0](ev)
        _5_5_conv_ev1 = self.ev_5_5_conv[0](ev)
        pic1_raw = torch.cat((pic1_raw, _1_1_conv_ev1, _3_3_conv_ev1, _5_5_conv_ev1), dim=1)  # dim需检查
        _1_1_conv_pic1 = self._1_1_conv[0](pic1_raw)
        _3_3_conv_pic1 = self._3_3_conv[0](pic1_raw)
        _5_5_conv_pic1 = self._5_5_conv[0](pic1_raw)
        pic1 = torch.cat((_1_1_conv_pic1, _3_3_conv_pic1, _5_5_conv_pic1), dim=1)
        ev = self.evDown1(ev)
        pic2_raw = self.down1(pic1)

        _1_1_conv_ev2 = self.ev_1_1_conv[1](ev)
        _3_3_conv_ev2 = self.ev_3_3_conv[1](ev)
        _5_5_conv_ev2 = self.ev_5_5_conv[1](ev)
        pic2_raw = torch.cat((pic2_raw, _1_1_conv_ev2, _3_3_conv_ev2, _5_5_conv_ev2), dim=1)  # dim需检查
        _1_1_conv_pic2 = self._1_1_conv[1](pic2_raw)
        _3_3_conv_pic2 = self._3_3_conv[1](pic2_raw)
        _5_5_conv_pic2 = self._5_5_conv[1](pic2_raw)
        pic2 = torch.cat((_1_1_conv_pic2, _3_3_conv_pic2, _5_5_conv_pic2), dim=1)
        ev = self.evDown2(ev)
        pic3_raw = self.down2(pic2)

        _1_1_conv_ev3 = self.ev_1_1_conv[2](ev)
        _3_3_conv_ev3 = self.ev_3_3_conv[2](ev)
        _5_5_conv_ev3 = self.ev_5_5_conv[2](ev)
        pic3_raw = torch.cat((pic3_raw, _1_1_conv_ev3, _3_3_conv_ev3, _5_5_conv_ev3), dim=1)  # dim需检查
        _1_1_conv_pic3 = self._1_1_conv[2](pic3_raw)
        _3_3_conv_pic3 = self._3_3_conv[2](pic3_raw)
        _5_5_conv_pic3 = self._5_5_conv[2](pic3_raw)
        pic3 = torch.cat((_1_1_conv_pic3, _3_3_conv_pic3, _5_5_conv_pic3), dim=1)
        ev = self.evDown3(ev)
        pic4_raw = self.down3(pic3)

        _1_1_conv_ev4 = self.ev_1_1_conv[3](ev)
        _3_3_conv_ev4 = self.ev_3_3_conv[3](ev)
        _5_5_conv_ev4 = self.ev_5_5_conv[3](ev)
        pic4_raw = torch.cat((pic4_raw, _1_1_conv_ev4, _3_3_conv_ev4, _5_5_conv_ev4), dim=1)  # dim需检查
        _1_1_conv_pic4 = self._1_1_conv[3](pic4_raw)
        _3_3_conv_pic4 = self._3_3_conv[3](pic4_raw)
        _5_5_conv_pic4 = self._5_5_conv[3](pic4_raw)
        pic4 = torch.cat((_1_1_conv_pic4, _3_3_conv_pic4, _5_5_conv_pic4), dim=1)
        ev = self.evDown4(ev)
        pic5_raw = self.down4(pic4)

        _1_1_conv_ev5 = self.ev_1_1_conv[4](ev)
        _3_3_conv_ev5 = self.ev_3_3_conv[4](ev)
        _5_5_conv_ev5 = self.ev_5_5_conv[4](ev)
        pic5_raw = torch.cat((pic5_raw, _1_1_conv_ev5, _3_3_conv_ev5, _5_5_conv_ev5), dim=1)  # dim需检查
        _1_1_conv_pic5 = self._1_1_conv[4](pic5_raw)
        _3_3_conv_pic5 = self._3_3_conv[4](pic5_raw)
        _5_5_conv_pic5 = self._5_5_conv[4](pic5_raw)
        pic5 = torch.cat((_1_1_conv_pic5, _3_3_conv_pic5, _5_5_conv_pic5), dim=1)

        # 底层处理
        pic5_pro = self.bottomProcess(pic5)

        # upward 过程
        pic4_pro = self.up4(pic5_pro)
        pic4_pro = torch.cat((pic4_pro, pic4), dim=1)

        pic3_pro = self.up3(pic4_pro)
        pic3_pro = torch.cat((pic3_pro, pic3), dim=1)

        pic2_pro = self.up2(pic3_pro)
        pic2_pro = torch.cat((pic2_pro, pic2), dim=1)

        pic1_pro = self.up1(pic2_pro)
        pic1_pro = torch.cat((pic1_pro, pic1), dim=1)

        mediate = self.picFinalPro1(pic1_pro)
        result = self.picFinalPro2(mediate)
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
