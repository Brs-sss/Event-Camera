import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from bairunsheng.model.data import PictureSet
from bairunsheng.model.net import ReconstructionNet
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
sys.path.append("C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng")

# ----------------------------------训练过程---------------------------------------
# 基本准备
net = ReconstructionNet()
maxEpoch = 75

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.8, dampening=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

log = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/logs'
writer = SummaryWriter(log_dir=log)

# 数据集获取 data_train/
train_data = PictureSet(type='train')
valid_data = PictureSet(type='verify')
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=1)

# 训练过程
for epoch in range(0, maxEpoch):

    loss_sigma = 0.0  # 记录一个epoch的loss之和
    i = 0

    for data in train_loader:
        i += 1
        # 获取数据
        raw, ev, tar = data
        # raw = torch.squeeze(raw, 0)
        # ev = torch.squeeze(ev, 0)

        optimizer.zero_grad()
        pro = net.forward(raw, ev)
        loss = criterion(pro, tar)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 0:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Learning Rate: {:.4f}".format(
                epoch + 1, maxEpoch, i, len(train_loader), loss_avg, scheduler.get_last_lr()[0]))

            # 记录训练loss
            writer.add_scalar('loss', loss_avg, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

    scheduler.step()  # 更新学习率
    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

net_save_path = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/results/net_params.pkl'
torch.save(net.state_dict(), net_save_path)

# ----------------------------------训练过程---------------------------------------

# 选取20个例子进行验证
for i, data in enumerate(valid_loader):
    if i >= 20:
        break
    raw, ev, tar = data
    result = net.forward(raw, ev)
    pic_save_path = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/results/' + str(i+1)
    transforms.ToPILImage()(tar).save(pic_save_path + 'tar.png')
    transforms.ToPILImage()(result).save(pic_save_path + 'result.png')
