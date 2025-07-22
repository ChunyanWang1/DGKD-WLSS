import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

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


class MIDAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm_x = LayerNorm(dim)
        self.pre_norm_c = LayerNorm(dim)

        self.to_qkv_x = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_qkv_c = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1))

    def forward(self, x, c_x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm_x(x)
        c_x = self.pre_norm_c(c_x)

        qkv_x = self.to_qkv_x(x).chunk(3, dim=1)
        qkv_c = self.to_qkv_c(c_x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)
        q_c, k_c, v_c = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_c)

        # 获取 QKV 的计算结果
        q_c = q_c * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q_c, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # 维度转换后获取网络输出
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


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
        #self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, (1, 1)), LayerNorm(dim))
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1))

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

class SFTLayer(nn.Module):
    def __init__(self,inplane,outplane):
        super(SFTLayer, self).__init__()
        # self.CondNet = nn.Sequential(
        #     nn.Conv2d(3, 8, 3, 1), nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
        #     nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, outplane, 1))#32
        # self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        # self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        # self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        # self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_scale_conv0 = nn.Conv2d(inplane,inplane, 1)
        self.SFT_scale_conv1 = nn.Conv2d(inplane,outplane, 1)
        self.SFT_shift_conv0 = nn.Conv2d(inplane,inplane, 1)
        self.SFT_shift_conv1 = nn.Conv2d(inplane,outplane, 1)
        # # 自注意力机制
        # self.attention_f=MIDAttention(outplane, heads=4, dim_head=32) #CEMLinearAttention(outplane, heads=4, dim_head=32)
        # self.att_self=CEMLinearAttention(outplane, heads=4, dim_head=32)

    def forward(self, x,y):
        # x: fea; y: cond
        #y=self.CondNet(y)
        n,c,h,w=x.size()
        y=F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(y), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(y), 0.1, inplace=True))
        new_x=x * (scale + 1) + shift

        # # # 利用自注意力机制增强学习到的特征
        # # if c==256:
        # norm_z = self.att_self(x + new_x)
        # # else:
        # #     norm_z = self.attention_f(x,new_x)
        #return norm_z
        return new_x #x * (scale + 1) + shift