import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        hidden_dim = int(embed_dim / 2)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # mlp softmax
        # self.linear1 = MLP(input_dim=embedding_dim, embed_dim=hidden_dim)
        # self.linear2 = MLP(input_dim=hidden_dim, embed_dim=2)

        # conv softmax
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # print(c4.shape)
        # print(c3.shape)
        # print(c2.shape)
        # print(c1.shape)
        # torch.Size([16, 512, 15, 20])
        # torch.Size([16, 320, 30, 40])
        # torch.Size([16, 128, 60, 80])
        # torch.Size([16, 64, 120, 160])

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        # print(x.shape) torch.Size([16, 512, 120, 160])
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.linear1(x).permute(0,2,1)
        # x = self.linear2(x).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        x = self.linear_pred(x)
        # x = self.relu(x)


        # x = self.sigmoid(x) #bs, 2, 120, 160
        # x = self.softmax(x)

        return x

class fc_decoder(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(fc_decoder, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        hidden_dim = int(embed_dim / 2)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim,
                      out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
            norm_layer(embedding_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim,
                      out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
            norm_layer(embedding_dim),
            nn.ReLU())

        # self.avg = nn.AvgPool2d((15, 20))
        self.avg = nn.AvgPool2d((7, 10))
        self.linear_cls = MLP(input_dim=c4_in_channels, embed_dim=self.num_classes)



    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c4.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c4.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c3, size=c4.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # 16 512 15 20
        x = self.dropout(_c)

        x = self.conv1(x) # 16 512 7 10
        x = self.conv2(x) + x

        x = self.avg(x)
        x = self.linear_cls(x).squeeze()
        # 16 2



        return x