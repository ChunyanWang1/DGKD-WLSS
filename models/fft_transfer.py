import math
import copy
import torch
import torch.nn.functional as F

from random import random
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
#from beartype import beartype
from functools import partial
from torch.fft import fft2, ifft2
from collections import namedtuple
from einops.layers.torch import Rearrange

# 判断变量是否存在
def exists(x):
    return x is not None

# 变量的默认选择
def default(val, d):
    # 判断变量是否存在, 如果存在, 直接返回结果. 否则进行变量2的判断
    if exists(val):
        return val

    # 判断输入变量2是否为函数, 如果为函数, 返回函数结果, 否则直接返回变量2
    if callable(d):
        return d()
    else:
        return d

# 层归一化模块
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

# 通道注意力机制 CAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, (1, 1), bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


# 空间注意力机制 SAM
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        #self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)#kernel_size=(7, 7)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out



# Contrast Enhancement Module
class CEMLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, (1, 1)), LayerNorm(dim))

        # 空间和通道注意力机制
        self.channel_attention = ChannelAttentionModule(hidden_dim)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # SAM 和 CAM 获取过程
        qkv = list(qkv)
        qkv[0] = self.channel_attention(qkv[0]) * qkv[0] + qkv[0]
        qkv[1] = self.spatial_attention(qkv[1]) * qkv[1] + qkv[1]

        # 格式转换
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 线性注意力机制的计算过程
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 获取线性注意力机制的输出结果
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)

# 扩散模型的FFT编码过程
class Conditioning(nn.Module):
    def __init__(self, fmap_h,fmap_w, dim):
        super().__init__()

        # 初始化调制高频的注意力图
        self.ff_theta = nn.Parameter(torch.ones(dim, 1, 1))
        self.ff_parser_attn_map_r = nn.Parameter(torch.ones(dim, fmap_h, fmap_w))
        self.ff_parser_attn_map_i = nn.Parameter(torch.ones(dim, fmap_h, fmap_w))

        # 输入变量归一化
        self.norm_input = LayerNorm(dim, bias=True)

        # # 构建残差模块
        # self.block = ResnetBlock(dim, dim)

        # 自注意力机制
        self.attention_f = CEMLinearAttention(dim, heads=4, dim_head=32)

    def forward(self, x):
        # 调制高频的注意力图
        x_type = x.dtype

        # 二维傅里叶变换
        z = fft2(x)

        # 获取傅里叶变换后的 实部 和 虚部
        z_real = z.real
        z_imag = z.imag

        # 频域滤波器 保持低频，增强高频
        # 可学习高频滤波 或者 高频滤波器 (实部 和 虚部的加权处理)
        z_real = z_real * self.ff_parser_attn_map_r
        z_imag = z_imag * self.ff_parser_attn_map_i

        # 合并为复数形式
        z = torch.complex(z_real * self.ff_theta, z_imag * self.ff_theta)

        # 反变换后只需要实部，虚部 为误差
        z = ifft2(z).real

        # 格式转换
        z = z.type(x_type)

        # 条件变量和输入变量的融合
        norm_z = self.norm_input(z)

        # 利用自注意力机制增强学习到的特征
        norm_z = self.attention_f(norm_z + x)

        # 添加一个额外的块以允许更多信息集成，在条件块之后有一个下采样（但也许有一个比下采样之前更好的条件）
        return norm_z #self.block(norm_z)