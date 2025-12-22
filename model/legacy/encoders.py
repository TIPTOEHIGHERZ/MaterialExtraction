import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
# import clip
from einops import rearrange, repeat
import kornia
import math
import numpy as np


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class SpatialDownsampling(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False,
                 size=32,
                 context_dim=1024,
                 key='c_crossattn'):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        # self.size = int(math.sqrt(context_dim))
        self.size = size
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            # self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)
            self.channel_mapper = nn.Conv2d(in_channels,context_dim,1,bias=bias)

    def forward(self, x):        
        # x is a list of tensors
        x = x[0]
        for stage in range(self.n_stages):
            x = self.interpolator(x, size=self.size)

        if self.remap_output:
            x = self.channel_mapper(x)
        return [x.view(x.shape[0], x.shape[1], -1)]

    def encode(self, x):
        return self(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
            inputs :
                input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batchsize, C, width, height = input.size()
        proj_query = self.query_conv(input).view(batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(input).view(batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(input).view(batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)

        out = self.gamma * out + input
        return out

class ConvBlock(nn.Module):
   def __init__(self, in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
      super(ConvBlock, self).__init__()
      self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
      self.bn = nn.BatchNorm2d(out_planes)
   def forward(self, x):
      return F.relu(self.bn(self.conv2d(x)), inplace=False)

class PartialConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), multi_channel=False):
        super(PartialConv, self).__init__()
        # self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2d = PartialConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, multi_channel=multi_channel, return_mask=True)
        # self.bn = nn.BatchNorm2d(out_planes)
        # self.ln = nn.LayerNorm()
        self.gn = nn.GroupNorm(32, out_planes)
    def forward(self, x, mask=None):
        x, mask = self.conv2d(x, mask)
        # x = F.relu(self.bn(x), inplace=False)
        x = F.relu(self.gn(x), inplace=False)
        return x, mask
        # return F.relu(self.bn(self.conv2d(x)), inplace=False)

# downsampling 3 times
class SAEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 size=32,
                 context_dim=1024,
                 key='c_crossattn'):
        super(SAEncoder, self).__init__()
        self.inconv = ConvBlock(in_channels, 64)
        self.down1 = ConvBlock(64, 128, stride=(2, 2))
        self.down2 = ConvBlock(128, 256, stride=(2, 2))
        self.down3 = ConvBlock(256, 512, stride=(2, 2))
        self.SA = Self_Attn(512)
        self.outconv = ConvBlock(512, context_dim)
    def forward(self, x):
        x = x[0]
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.SA(x)
        x = self.outconv(x)
        return [x.view(x.shape[0], x.shape[1], -1)]


class PartialSAEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 context_dim=1024,
                 key='c_crossattn',
                 multi_channel=False):
        super(PartialSAEncoder, self).__init__()
        self.inconv = PartialConv(in_channels, 64, multi_channel=multi_channel)
        self.down1 = PartialConv(64, 128, stride=(2, 2), multi_channel=multi_channel)
        self.conv1 = PartialConv(128, 128, multi_channel=multi_channel)
        self.down2 = PartialConv(128, 256, stride=(2, 2), multi_channel=multi_channel)
        self.conv2 = PartialConv(256, 256, multi_channel=multi_channel)
        self.down3 = PartialConv(256, 512, stride=(2, 2), multi_channel=multi_channel)
        self.conv3 = PartialConv(512, 512, multi_channel=multi_channel)
        self.SA = Self_Attn(512)
        self.outconv = PartialConv(512, context_dim, multi_channel=multi_channel)
    def forward(self, x, mask):
        x, mask = self.inconv(x, mask)
        x, mask = self.down1(x, mask)
        x, mask = self.conv1(x, mask)
        x, mask = self.down2(x, mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.down3(x, mask)
        x, mask = self.conv3(x, mask)
        x = self.SA(x)
        x, mask = self.outconv(x, mask)

        x = x.view(*x.shape[:2], -1)
        return x
        # return [x.view(x.shape[0], x.shape[1], -1)]
        # return x.view(x.shape[0], x.shape[1], -1)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        # print(input.shape)
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output