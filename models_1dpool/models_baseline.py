import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.BN(self.conv(x)))
        out = self.pool(out)
        return out


class Crnn(nn.Module):
    def __init__(self, num_freq, class_num):
        super(Crnn, self).__init__()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        self.block1 = ConvBlock(in_channels=1, out_channels=16)
        self.block2 = ConvBlock(in_channels=16, out_channels=32)
        self.block3 = ConvBlock(in_channels=32, out_channels=64)
        self.global_pool = nn.AdaptiveAvgPool2d((62, 1))
        # [b_s, c, T/8, F/8]
        self.fc = nn.Linear(64*2, class_num, bias=False)
        self.rnn = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        """
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. 
        See the Inputs/Outputs sections below for details. Default: False
        """

    def detection(self, x, debug=False):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = x.unsqueeze(dim=1)
        # x [32, 501, 64] -> [32, 1, 501, 64]
        out = self.block1(x)
        # out [32, 16, 501, 64] -> [32, 16, 250, 32]
        out = self.block2(out)
        # out [32, 32, 250, 32] -> [32, 32, 125, 16]
        out = self.block3(out)
        # out [32, 64, 125, 16] -> [32, 64, 62, 8] [b_s, c, T/8, F/8]
        out = self.global_pool(out)  # [32, 64, 62, 1]  [b_s, c, T/8, 1]
        # .squeeze()
        out = out.transpose(1, 2).squeeze(dim=3)  # [32, 62, 64, 1] [b_s, T/8, c, 1]
        # [b_s, T/8, c]
        out, hidden = self.rnn(out)  # [32, 62, 128] [b_s, T/8, 2c]
        # hidden [2, 32, 64] [2, b_s, c]
        # BiGRU
        out = self.fc(out)  # [32, 62, 10] [b_s, T/8, class_num]
        out = torch.sigmoid(out)  # forward中有linear_softmax_pooling
        return out

    def forward(self, x, debug=False):
        # x [32, 501, 64]
        frame_wise_prob = self.detection(x, debug)
        # frame_wise_prob [32, 62, 10] [b_s, T/8, class_num]
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
