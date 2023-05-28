import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from bairunsheng.model.data import PictureSet
from bairunsheng.model.net import ReconstructionNet
from  bairunsheng.tools.UnNormalize import unNormalize
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
sys.path.append("C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng")

# ----------------------------------基本准备---------------------------------------
net = ReconstructionNet()
maxEpoch = 2
meanCal = [0.0024475607, 0.0055219554, 0.010198287]
stdCal = [0.027039433, 0.033439837, 0.03856222]

criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.005)
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

log = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/logs'
writer = SummaryWriter(log_dir=log)

# 数据集获取 data_train/
train_data = PictureSet(type='train')
valid_data = PictureSet(type='verify')
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=1)

net_save_path = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/results/net_params.pkl'
net_load_path = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/results/net_params_third_2.pkl'
net.load_state_dict(torch.load(net_load_path))  # 该行可选，目前结果已训练1遍

device = torch.device("cuda")  # 使用gpu训练
net = net.to(device)
criterion = criterion.to(device)
# ----------------------------------训练部分---------------------------------------
print(next(net.parameters()).is_cuda)
for epoch in range(0, maxEpoch):

    loss_sigma = 0.0  # 记录一个epoch的loss之和
    i = 0

    for data in train_loader:
        i += 1
        # 获取数据
        raw, ev, tar = data
        # raw = torch.squeeze(raw, 0)
        # ev = torch.squeeze(ev, 0)
        raw = raw.to(device)
        ev = ev.to(device)
        tar = tar.to(device)

        optimizer.zero_grad()
        pro = net.forward(raw, ev)
        loss = criterion(pro, tar)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        loss_sigma += loss.item()

        # 每3个iteration 打印一次训练信息，loss为4个iteration的平均
        if i % 2 == 0:
            loss_avg = loss_sigma / 2
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Learning Rate: {:.4f}".format(
                epoch + 1, maxEpoch, i, len(train_loader), loss_avg, scheduler.get_last_lr()[0]))

            # 记录训练loss
            writer.add_scalar('loss', loss_avg, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

    scheduler.step()  # 更新学习率
    # 每个epoch，记录梯度，权值
    '''
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
    '''

torch.save(net.state_dict(), net_save_path)

writer.close()

# ----------------------------------验证---------------------------------------
# 选取20个例子进行验证
for i, data in enumerate(valid_loader):
    if i >= 20:
        break
    raw, ev, tar = data
    raw = raw.to(device)
    ev = ev.to(device)
    tar = tar.to(device)
    result = net.forward(raw, ev)
    loss = criterion(result, tar)
    print(loss)
    raw = torch.squeeze(unNormalize(raw, mean=meanCal, std=stdCal), 0)
    tar = torch.squeeze(unNormalize(tar, mean=meanCal, std=stdCal), 0)
    result = torch.squeeze(unNormalize(result, mean=meanCal, std=stdCal), 0)
    pic_save_path = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/bairunsheng/results/' + str(i+1)
    transforms.ToPILImage()(raw).save(pic_save_path + '/raw.png')
    transforms.ToPILImage()(tar).save(pic_save_path + '/tar.png')
    transforms.ToPILImage()(result).save(pic_save_path + '/result.png')
