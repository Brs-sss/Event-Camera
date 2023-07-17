import torch
import torch.nn as nn
import torch.nn.functional as F
import bairunsheng.model.function as func


class Encoder(nn.Module):
    def __init__(self, batch: int, num: int, length: int):
        """
        Args:
            batch: batch number
            num: number of vector in sequence
            length: length per vector
            qkv: if qkv then encoder returns the calculated qkv
        """
        super(Encoder, self).__init__()
        self.num = num
        self.length = length
        self.MHA = func.MultiHeadAttention(dim=length, head_num=4)
        self.FFN = func.PositionWiseFFN(length=length)
        self.norms = nn.ModuleList()
        for i in range(0, 2):
            self.norms.append(nn.LayerNorm(normalized_shape=[batch, num, length]))

    def forward(self, sequence):
        B, N, L = sequence.shape
        assert N == self.num and L == self.length, \
            f"Input sequence size ({N}*{L}) doesn't match model ({self.num}*{self.length})"

        sequence_pro1 = self.MHA(sequence)
        sequence_pro1 = self.norms[0](sequence + sequence_pro1)

        sequence_pro2 = self.FFN(sequence)
        sequence_pro2 = self.norms[1](sequence_pro1 + sequence_pro2)

        return sequence_pro2


class Decoder(nn.Module):
    def __init__(self, batch: int, num: int, length: int):
        """
        Args:
            batch: batch number
            num: number of vector in sequence
            length: length per vector
        """
        super(Decoder, self).__init__()
        self.num = num
        self.length = length
        self.masked = func.MultiHeadAttention(dim=length, head_num=4, masked=True)
        self.seperated = func.MultiHeadAttention(dim=length, head_num=4, seperated_qkv=True)
        self.FFN = func.PositionWiseFFN(length=length)
        self.norms = nn.ModuleList()
        for i in range(0, 3):
            self.norms.append(nn.LayerNorm(normalized_shape=[batch, num, length]))

    def forward(self, encoder_sequence, gt_sequence):
        """
        Args:
            encoder_sequence: the sequence calculated by encoder
            gt_sequence: the sequence converted from ground-truth img
        Returns:

        """
        sequence_pro1 = self.masked(gt_sequence)
        sequence_pro1 = self.norms[0](gt_sequence + sequence_pro1)

        sequence_pro2 = self.seperated((sequence_pro1, encoder_sequence, encoder_sequence))
        sequence_pro2 = self.norms[1](sequence_pro1 + sequence_pro2)

        sequence_pro3 = self.FFN(sequence_pro2)
        sequence_pro3 = self.norms[2](sequence_pro2 + sequence_pro3)

        return sequence_pro3
