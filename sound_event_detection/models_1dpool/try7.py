import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class ConvBlock(nn.Module):
    # [batch_size, channel, time_steps, num_freq]
    def __init__(self, in_channels=1, out_channels=1, t_ksize=3, f_ksize=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(t_ksize, f_ksize),
                              stride=1, padding=((t_ksize-1)//2, (f_ksize-1)//2))
        self.BN = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # try GELU

    def forward(self, x):
        out = F.relu(self.BN(self.conv(x)))
        out = self.pool(out)
        return out


class Try7_4conv(nn.Module):
    def __init__(self, num_freq, class_num):
        super(Try7_4conv, self).__init__()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        self.block = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=16, t_ksize=9, f_ksize=9),
            ConvBlock(in_channels=16, out_channels=32, t_ksize=9, f_ksize=7),
            ConvBlock(in_channels=32, out_channels=64, t_ksize=7, f_ksize=7),
            ConvBlock(in_channels=64, out_channels=128, t_ksize=7, f_ksize=7),
            ConvBlock(in_channels=128, out_channels=256, t_ksize=7, f_ksize=7)
        )
        # [b_s, c, T/8, F/8]
        self.fc = nn.Linear(2*512, class_num)
        # self.fc2 = nn.Linear(128, class_num)
        self.rnn = nn.GRU(input_size=8*256, hidden_size=512, num_layers=3, batch_first=True, dropout=0, bidirectional=True)

    def detection(self, x, debug=False):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = x.unsqueeze(dim=1)
        # x [32, 501, 256] -> [32, 1, 501, 256] [batch_size, 1, time_steps, f
        out = self.block(x)
        # using stack
        out = out.permute(0, 2, 3, 1).flatten(2)  # stack at time_dim [b_s, t, f/32, c]
        # out = out.permute(0, 2, 1, 3).flatten(2)  # stack at time_dim [b_s, t, c, f/32]
        # out = out.reshape(out.shape[0], out.shape[1], -1)
        """
        out = self.global_pool(out)  # [32, 128, 501, 1]  [b_s, c, t, 1]
        out = out.transpose(1, 2).squeeze(dim=3)  # [32, 501, 128] [b_s, T, c]
        """
        out, hidden = self.rnn(out)  # [32, 501, 128*2] [b_s, t, 2c]
        out = self.fc(out)  # [32, 501, 10] [b_s, T, class_num]
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