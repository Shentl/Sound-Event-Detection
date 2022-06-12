import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    # [batch_size, channel, time_steps, num_freq]
    def __init__(self, in_channels=1, out_channels=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        out = F.relu(self.BN(self.conv(x)))
        out = self.pool(out)
        return out


class Crnn_4conv(nn.Module):
    def __init__(self, num_freq, class_num):
        super(Crnn_4conv, self).__init__()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        self.block1 = ConvBlock(in_channels=1, out_channels=16)
        self.block2 = ConvBlock(in_channels=16, out_channels=32)
        self.block3 = ConvBlock(in_channels=32, out_channels=64)
        self.block4 = ConvBlock(in_channels=64, out_channels=128)
        self.global_pool = nn.AdaptiveAvgPool2d((501, 1))
        # [b_s, c, T/8, F/8]
        self.fc = nn.Linear(128*2, class_num)
        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0, bidirectional=True)
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
        # x [32, 501, 64] -> [32, 1, 501, 64] [batch_size, 1, time_steps, f
        out = self.block1(x)
        # out [32, 16, 501, 64] -> [32, 16, 501, 32] [batch_size, 16, time_steps, f/2]
        out = self.block2(out)
        # out [32, 32, 501, 32] -> [32, 32, 501, 16] [batch_size, 32, time_steps, f/4]
        out = self.block3(out)
        # out [32, 64, 501, 16] -> [32, 64, 501, 8] [b_s, c, t, f/8]
        out = self.block4(out)
        out = self.global_pool(out)  # [32, 64, 501, 1]  [b_s, c, t, 1]
        # .squeeze()
        out = out.transpose(1, 2).squeeze(dim=3)  # [32, 501, 64, 1] [b_s, T, c, 1]
        # [b_s, T, c]
        out, hidden = self.rnn(out)  # [32, 501, 128] [b_s, t, 2c]
        # hidden [2, 32, 64] [2, b_s, c]
        # BiGRU
        out = self.fc(out)  # [32, 501, 10] [b_s, T, class_num]
        out = torch.sigmoid(out)  # forward中有linear_softmax_pooling
        # print('out', out.shape)
        # [32, 501, 10]
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
