# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 16:48
# @Author  : Ran.Gu
# @Email   : guran924@std.uestc.edu.cn
'''
cs-cada uses different normalization.
'''
import random
import torch
import torch.nn as nn
from models.layers.dsbn import DomainSpecificBatchNorm2d

class Unet_dsbn_cont(nn.Module):
    def __init__(self, net_params:dict):
        super(Unet_dsbn_cont, self).__init__()
        self.num_filters = net_params['num_filters']
        self.num_channels = net_params['num_channels']
        self.num_classes = net_params['num_classes']
        self.normalization = net_params['normalization']
        self.num_domain = net_params['num_domains']
        filters = [self.num_filters,
                   self.num_filters * 2,
                   self.num_filters * 4,
                   self.num_filters * 8,
                   self.num_filters * 16]

        self.conv1 = conv_block(self.num_channels, filters[0], normalization=self.normalization, num_domain=self.num_domain)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv_block(filters[0], filters[1], normalization=self.normalization, num_domain=self.num_domain)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = conv_block(filters[1], filters[2], normalization=self.normalization, num_domain=self.num_domain)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(filters[2], filters[3], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(filters[3], filters[4], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)

        # f1 and g1 encoder
        self.f1 = nn.Sequential(nn.Conv2d(filters[4], 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv2d(64, 16, kernel_size=1))
        self.g1 = nn.Sequential(nn.Linear(in_features=4096, out_features=1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=256))

        # upsample
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True, normalization=self.normalization, num_domain=self.num_domain)
        self.up3 = UpCatconv(filters[3], filters[2], normalization=self.normalization, num_domain=self.num_domain)
        self.up2 = UpCatconv(filters[2], filters[1], normalization=self.normalization, num_domain=self.num_domain)
        self.up1 = UpCatconv(filters[1], filters[0], normalization=self.normalization, num_domain=self.num_domain)
        
        self.final = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=1),
                                   nn.Conv2d(filters[0], self.num_classes, kernel_size=1))

    def forward(self, x, domain_label):
        conv1 = self.conv1(x, domain_label)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1, domain_label)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2, domain_label)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3, domain_label)
        pool4 = self.pool4(conv4)

        center = self.center(pool4, domain_label)
        high_d = self.f1(center)
        high_d_represent = self.g1(high_d.reshape(high_d.size(0), -1))

        up_4 = self.up4(conv4, center, domain_label)
        up_3 = self.up3(conv3, up_4, domain_label)
        up_2 = self.up2(conv2, up_3, domain_label)
        up_1 = self.up1(conv1, up_2, domain_label)

        out = self.final(up_1)
        return out, high_d_represent


# conv_block(nn.Module) for U-net convolution block
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False, normalization='none', num_domain = 6):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)
        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm2d(ch_out)
        elif normalization == 'instancenorm':
            self.bn = nn.InstanceNorm2d(ch_out)
        elif normalization == 'dsbn':
            self.bn = DomainSpecificBatchNorm2d(ch_out, num_domain)
        elif normalization != 'none':
            assert False
        self.relu = nn.ReLU(inplace=True)
        self.dropout = drop_out

    def forward(self, x, domain_label):
        x = self.conv1(x)
        if self.normalization != 'none':
            if self.normalization == 'dsbn':
                x = self.bn(x, domain_label)
            else:
                x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.normalization != 'none':
            if self.normalization == 'dsbn':
                x = self.bn(x, domain_label)
            else:
                x = self.bn(x)
        x = self.relu(x)

        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x


# # UpCatconv(nn.Module) for U-net UP convolution
class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False, normalization='none', num_domain = 6):
        super(UpCatconv, self).__init__()
        self.normalization = normalization

        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, drop_out=drop_out, normalization=self.normalization,
                                   num_domain=num_domain)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out, normalization=self.normalization,
                                   num_domain=num_domain)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, down_outputs, domain_label):
        outputs = self.up(down_outputs)
        out = self.conv(outputs, domain_label)

        return out

if __name__ == '__main__':
    import numpy as np
    net_params = {'num_classes':2, 'num_channels':3, 'num_filters':32,
                  'num_filters_cond':16, 'num_domains':6, 'normalization':'dsbn'}
    model = Unet_dsbn_cont(net_params).cuda()
    x = torch.tensor(np.random.random([5, 3, 384, 384]), dtype=torch.float32)
    x = x.cuda()
    pred = model(x,5)
    print(pred.shape)