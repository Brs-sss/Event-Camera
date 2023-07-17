import math

import torch
import torch.nn as nn
import torch.nn.functional as func


class Embeddings(nn.Module):

    def __init__(self, img_size: (int, int), patch_size: (int, int), channel: int, length: int):
        """
        Args:
            img_size: size of input image
            patch_size: size of patch used to split picture
            channel: channel of input image, usually 3 which represents RGB
            length: expected size of vector converted from a patch
        """
        super(Embeddings, self).__init__()
        # 变量的定义
        self.img_size = img_size
        self.patch_size = patch_size
        '''
        assert (img_size[0] % patch_size[0] == 0) and (img_size[1] % patch_size[10] == 0), \
            f"Image size ({img_size[0]}*{img_size[1]}) is not dividable by " \
            f"Image size ({patch_size[0]}*{patch_size[1]})"
        '''
        self.length = length
        self.patch_num = int(img_size[0] / patch_size[0]) * int(img_size[1] / patch_size[1])  # 分成的区块个数，即向量的个数

        # 对内容信息的处理卷积
        self.patch_embeddings = nn.Conv2d(in_channels=channel,
                                          out_channels=length,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        torch.nn.init.kaiming_normal_(self.patch_embeddings.weight.data)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.patch_embeddings(x)
        x = torch.transpose(torch.flatten(x, start_dim=2), dim0=1, dim1=2)

        # todo: 位置信息加入，这里实际采用固定方式，而非init中定义的可学习信息
        # pos信息，即第几个向量
        pos1 = torch.zeros(B, self.patch_num, self.length)
        for i in range(0, self.patch_num):
            pos1[:, i, :] = i
        # i信息，即在向量的哪个位置
        pos2 = torch.zeros(B, self.patch_num, self.length)
        for i in range(0, self.length):
            pos2[:, :, i] = 2 * (i // 2) / self.length
        # 完整的pos信息
        pos = pos1 / pow(10000, pos2)
        for i in range(0, self.length // 2):
            pos[:, :, 2 * i + 1] += math.pi / 2
        pos = torch.sin(pos)
        # x = torch.cat((x, pos), dim=-1)
        x = x + pos
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, head_num: int, seperated_qkv=False, masked=False):
        """
        Args:
            dim: length of input vector, also the factor for scale
            head_num: number of different head
        """
        super(MultiHeadAttention, self).__init__()

        assert dim % head_num == 0, \
            f"Vector size ({dim}) is not dividable by the expected number of heads ({head_num})."
        self.seperated_qkv = seperated_qkv
        self.dim = dim
        self.head_dim = int(dim / head_num)
        self.head_num = head_num
        self.masked = masked

        self.Wq = nn.Parameter(torch.eye(dim))
        self.Wk = nn.Parameter(torch.eye(dim))
        self.Wv = nn.Parameter(torch.eye(dim))

        '''
        self.qCreators = []
        self.kCreators = []
        self.vCreators = []
        
        for i in range(0, head_num):
            self.qCreators.append(nn.Parameter(torch.eye(dim)))
            self.kCreators.append(nn.Parameter(torch.eye(dim)))
            self.vCreators.append(nn.Parameter(torch.eye(dim)))
        '''

    def forward(self, content):
        """
        Args:
            content: if seperated_qkv then content 3 sequences which mul W to form (Q, K, V),
                     else content is vector sequences
        Returns:
            (Q, K, V): Query, Key, Value
        """
        # batch * num * length
        if self.seperated_qkv:
            Q, K, V = content
            Q = Q @ self.Wq
            K = K @ self.Wk
            V = V @ self.Wv
            # assert (Bq == Bk == Bv) and (Nq == Nk == Nv) and (Lq == Lk == Lv), \
            #     f"Input Q({Bq}*{Nq}*{Lq}), K({Bk}*{Nk}*{Lk}), V({Bv}*{Nv}*{Lv}) doesn't match)."
        else:
            B, N, L = content.shape
            assert L == self.dim, \
                f"Input size of vectors ({L}) doesn't match model dimension {self.dim})."
            Q = content @ self.Wq
            K = content @ self.Wk
            V = content @ self.Wv

        Qis = torch.split(Q, self.head_dim, dim=-1)
        Kis = torch.split(K, self.head_dim, dim=-1)
        Vis = torch.split(V, self.head_dim, dim=-1)

        #        softmax:  Q * K的转置，(0, 2, 1)即对每个batch转置           除以根号dk进行scale         dim=2即对每行softmax
        alpha = func.softmax(Qis[0] @ torch.permute(Kis[0], (0, 2, 1)) / math.sqrt(self.head_dim), dim=2)
        if self.masked:
            for i in range(0, alpha.shape[1]):
                alpha[0, i, i+1:alpha.shape[2]] = 0
        Y = alpha @ Vis[0]

        for i in range(1, self.head_num):
            alpha = func.softmax(Qis[i] @ torch.permute(Kis[i], (0, 2, 1)) / math.sqrt(self.head_dim), dim=2)
            if self.masked:
                for j in range(0, alpha.shape[1]):
                    alpha[0, j, j + 1:alpha.shape[2]] = 0
            # 连接已有的Y和新计算出来的结果
            Y = torch.cat((Y, alpha @ Vis[i]), dim=-1)

        return Y


class PositionWiseFFN(nn.Module):
    def __init__(self, length: int):
        """
        Args:
            length: length per vector
        """
        super(PositionWiseFFN, self).__init__()
        self.length = length
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=length,
                              kernel_size=(1, length))
        torch.nn.init.kaiming_normal_(self.conv.weight.data)

    def forward(self, y):
        B, N, L = y.shape
        assert L == self.length, \
            f"Input size of vectors ({L}) doesn't match model expecting length{self.dim})."
        y = torch.unsqueeze(y, dim=1)
        y = self.conv(y)
        y = torch.squeeze(y)
        y = torch.permute(y, (0, 2, 1))
        return y
