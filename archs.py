import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import torch.nn.functional as F
from Seg_UKAN.network.backbone.regnet import DenseNetWrapper
# import MobileNetV2
from torch import nn, einsum
from Seg_UKAN.network.VisionLSTM import *
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class EfficientSelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads,
            reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        return self.to_out(out)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCALayer, self).__init__()

        self.esa = EfficientSelfAttention(dim=64, heads=2, reduction_ratio=reduction)

        hidden_dim = channel * 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channel, 1),
        )
    def forward(self, x):#v3
        y = self.esa(x)
        y = self.conv3(y)
        return y


class SIMDB(nn.Module):  # OLD
    def __init__(self, in_channels, distillation_rate=0.25):
        super(SIMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        # self.remaining_channels = int(in_channels - self.distilled_channels)
        # self.c1 = Conv2dSWL(in_channels, in_channels, 2)
        # self.c2 = Conv2dSWR(self.remaining_channels, in_channels, 2)
        # self.c3 = Conv2dSWU(self.remaining_channels, in_channels, 2)
        # self.c4 = Conv2dSWD(self.remaining_channels, self.distilled_channels, 2)

        # self.c1=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c2=nn.Conv2d(self.remaining_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c3=nn.Conv2d(self.remaining_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c4=nn.Conv2d(self.remaining_channels, self.distilled_channels, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels, self.distilled_channels, kernel_size=3, stride=1, padding=1)
        self.act = activation('lrelu', neg_slope=0.05)
        # self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c6 = conv_layer(in_channels, self.distilled_channels, 1)
        # self.cca = CCALayer(self.distilled_channels * 4)
        # self.esa = EfficientSelfAttention(dim=64,heads=2,reduction_ratio=4)
        # self.se=SEModule(64)
        # self.cbam = CBAM(64)
        self.relu = nn.ReLU(inplace=True)
        # self.conv_aggregation= FeatureFusionModule(64,64, 64)

    def forward(self, input):  # m:cca改成了semodule split改成普通的
        out_c1 = self.act(self.c1(input))
        # distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(out_c1))
        # distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(out_c2))
        # distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(out_c3)
        out = torch.cat((self.c6(out_c1), self.c6(out_c2), self.c6(out_c3), out_c4), dim=1)

        # out_fused = self.c5(self.cbam(out)) + input
        out_fused = self.relu(out + input)
        # out_fused = self.conv_aggregation(out , input)#p2
        return out_fused

#修改部分
class ECALayer(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:  [B, C, H, W]
        y = self.avg_pool(x)                # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [B, 1, C]
        y = self.conv(y)                    # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y
class LiteSIMDB(nn.Module):
    def __init__(self, in_channels):
        super(LiteSIMDB, self).__init__()
        # 深度可分离卷积
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                                groups=in_channels, bias=False)
        self.pwconv1 = nn.Conv2d(in_channels, in_channels*2, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.pwconv2 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.bn(out)
        return out + identity











class FPN(Module):
    def __init__(
            self,
            ch_in,  # 输入张量x和y的通道数
            ch_out: int,  # 输出张量的通道数
    ):
        super().__init__()
        if ch_in is None:
            ch_in = [16, 24, 32, 96, 320]
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch3 = 64
        self.ch2 = 32
        self.ch1 = 16
        self.chfusein = 144
        # self.ch3=32
        # self.ch2=32
        # self.ch1=32
        # self.chfusein=128
        # self.relu = nn.ReLU(inplace=True)
        # self.conv_c2=nn.Conv2d(self.ch_in[0], self.ch_out, kernel_size=1)
        # self.conv_c3=nn.Conv2d(self.ch_in[1], self.ch_out, kernel_size=1)
        # self.conv_c4=nn.Conv2d(self.ch_in[2], self.ch_out, kernel_size=1)
        # self.conv_c5=nn.Conv2d(self.ch_in[3], self.ch_out, kernel_size=1)

        self.convs1_1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[0], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True)
        )
        self.convs1_2 = nn.Sequential(
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs1_3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs1_4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 添加上采样层
        )

        self.convs2_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[0], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True)
        )
        self.convs2_2 = nn.Sequential(
            nn.Conv2d(self.ch_in[1], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs2_3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs2_4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 添加上采样层
        )

        self.convs3_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.ch_in[0], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True)
        )
        self.convs3_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs3_3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs3_4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )

        self.convs4_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.ch_in[0], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True)
        )
        self.convs4_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs4_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs4_4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        # self.conv_fuse = nn.Sequential(
        #     nn.Conv2d(self.chfusein, self.ch_out, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.ch_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.ch_out, self.ch_out, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.ch_out),
        #     nn.ReLU(inplace=True)
        # )

        self.conv_aggregation_s1 = FeatureFusionModule(self.chfusein, self.ch_in[0], self.ch_out)
        self.conv_aggregation_s2 = FeatureFusionModule(self.chfusein, self.ch_in[1], self.ch_out)
        self.conv_aggregation_s3 = FeatureFusionModule(self.chfusein, self.ch_in[2], self.ch_out)
        self.conv_aggregation_s4 = FeatureFusionModule(self.chfusein, self.ch_in[3], self.ch_out)

    def forward(self, c1, c2, c3, c4) -> Tensor:  # v2.0
        s1_c1 = self.convs1_1(c1)
        s1_c2 = self.convs1_2(c2)
        s1_c3 = self.convs1_3(c3)
        s1_c4 = self.convs1_4(c4)
        s1 = self.conv_aggregation_s1(torch.cat([s1_c1, s1_c2, s1_c3, s1_c4], dim=1), c1)
        # s1=self.conv_fuse(torch.cat([s1_c1, s1_c2,s1_c3,s1_c4], dim=1))

        s2_c1 = self.convs2_1(c1)
        s2_c2 = self.convs2_2(c2)
        s2_c3 = self.convs2_3(c3)
        s2_c4 = self.convs2_4(c4)
        # s2=self.conv_fuse(torch.cat([s2_c1, s2_c2,s2_c3,s2_c4], dim=1))
        s2 = self.conv_aggregation_s2(torch.cat([s2_c1, s2_c2, s2_c3, s2_c4], dim=1), c2)

        s3_c1 = self.convs3_1(c1)
        s3_c2 = self.convs3_2(c2)
        s3_c3 = self.convs3_3(c3)
        s3_c4 = self.convs3_4(c4)
        # s3=self.conv_fuse(torch.cat([s3_c1, s3_c2,s3_c3,s3_c4], dim=1))
        s3 = self.conv_aggregation_s3(torch.cat([s3_c1, s3_c2, s3_c3, s3_c4], dim=1), c3)

        s4_c1 = self.convs4_1(c1)
        s4_c2 = self.convs4_2(c2)
        s4_c3 = self.convs4_3(c3)
        s4_c4 = self.convs4_4(c4)

        # s4=self.conv_fuse(torch.cat([s4_c1, s4_c2,s4_c3,s4_c4], dim=1))
        s4 = self.conv_aggregation_s4(torch.cat([s4_c1, s4_c2, s4_c3, s4_c4], dim=1), c4)

        return s1 , s2 , s3 , s4



class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # self.esa = EfficientSelfAttention(dim=64,heads=2,reduction_ratio=4)
        # self.se_module = SEModule(self.out_d, reduction_ratio=16)
        # self.cbam = CBAM(64)
    def forward(self, c_fuse, c):#1.0
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse * self.conv_identity(c))
        # c_out = self.relu(c_fuse * self.se_module(self.conv_identity(c)))
        return c_out


class FFm(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(FFm, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(16, self.in_d, kernel_size=1)
        self.block0 = EMA(64).cuda()
        #self.block0 = ECALayer(64).cuda()

        self.msff = SIMDB(64)
        #self.msff = LiteSIMDB(64)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.sigmoid=nn.Sigmoid()
        # self.bn=nn.BatchNorm2d(self.in_d)

    # def featureFuse0(self,x1,x2):#mul
    #     df=torch.abs(x1 - x2)
    #     df=self.conv1(df)
    #     c1=self.msff(df)
    #     return c1
    def featureFuse(self, x1, x2):  # mul
       # df = torch.abs(x1 - x2)
        df = torch.abs(x1 )
        c1 = self.msff(df)
        return c1


    def forward(self, x1_2, x1_3, x1_4, x1_5, ):  # V6p2
        # temporal fusion
        #c2 = torch.abs(x1_2 - x2_2)
        c2 = torch.abs(x1_2 )
        c2 = self.block0(c2)#原版 EMA

        # c2 = self.featureFuse(x1_2, x2_2)
        #c3 = torch.abs(x1_3 - x2_3) * self.sigmoid(self.avg_pool(self.msff(c2)))
        c3 = torch.abs(x1_3) * self.sigmoid(self.avg_pool(self.msff(c2)))
        c3 = self.block0(c3)

        #c4 = torch.abs(x1_4 - x2_4) * self.sigmoid(self.avg_pool(self.msff(c3)))
        c4 = torch.abs(x1_4 ) * self.sigmoid(self.avg_pool(self.msff(c3)))
        c4 = self.block0(c4)

        #c5 = torch.abs(x1_5 - x2_5) * self.sigmoid(self.avg_pool(self.msff(c4)))
        c5 = torch.abs(x1_5 ) * self.sigmoid(self.avg_pool(self.msff(c4)))
        c5 = self.block0(c5)

        return c2, c3, c4, c5

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class Decoder(nn.Module):
    def __init__(self, mid_d=320,img_dim=96, in_channels=1, out_channels=64,
                 depth=12,
                 dim=256,
                 drop_path_rate=0.0,
                 stride=None,
                 alternation="bidirectional",
                 drop_path_decay=False,
                 legacy_norm=False):
        super(Decoder, self).__init__()
        self.patch_embed = PatchEmbeddingBlock(in_channels=64,
                                               img_size=8,
                                               patch_size=2,
                                               hidden_size=256,
                                               num_heads=1,
                                               proj_type='perceptron',
                                               spatial_dims=2)
        self.mid_d = mid_d
        self.alternation = alternation
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # directions
        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")
        # fusion
        self.blocks = nn.ModuleList(
            [
                ViLBlock(
                    dim=dim,
                    drop_path=dpr[i],
                    direction=directions[i],
                )
                for i in range(depth)
            ]
        )
        if legacy_norm:
            self.legacy_norm = LayerNorm(dim, bias=False)
        else:
            self.legacy_norm = nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

        self.cca = CCALayer(64)

    def forward(self, d2, d3, d4, d5):  # w2
        # high-level
        # mask_p5=self.cls(d5)
        # d5=self.conv_p2(d5) 4,64,8,8
        d5 = self.patch_embed(d5)
        for block in self.blocks:
            d5 = block(d5)#原版
            #d5 = d5
        d5 = self.legacy_norm(d5)
        d5 = self.norm(d5)
        d5 = einops.rearrange(d5, "b p (c h w) -> b p c h w", c=64, h=2, w=2)  # (4, 16, 64, 2, 2)
        p5 = einops.rearrange(d5, "b (h2 w2) c h w -> b c (h2 h) (w2 w)", h2=4, w2=4)  # (4, 64, 8, 8)
        # p5 = self.cca(d5)
        mask_p5 = self.cls(p5)

        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))
        # mask_p4=self.cls(p4)
        # p4 = self.cca(p4)

        mask_p4 = self.cls(p4)

        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))
        # mask_p3=self.cls(p3)
        # p3 = self.cca(p3)
        mask_p3 = self.cls(p3)

        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        # p2 = self.cca(p2)
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, attention,x1,x2,x3):
        x1 = x1+torch.mul(x1, self.upsample4(attention))
        x2 = x2+torch.mul(x2,self.upsample2(attention))
        x3 = x3+torch.mul(x3,attention)

        return x1,x2,x3






#添加部分
class SemiModel(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True, embed_dims=[128, 160, 256],
                 no_kan=True):
        super(SemiModel, self).__init__()

        # 基础设置
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        self.no_kan = no_kan
        self.en_d = 32
        self.mid_d = self.en_d * 2

        # 主干网络
        self.backbone = DenseNetWrapper()
        # 特征金字塔网络
        self.ccf = FPN([48, 120, 336, 888], 64)
        # 特征融合模块
        self.ffm = FFm(self.mid_d, self.en_d * 2)
        # 解码器
        self.decoder = Decoder(self.en_d * 2)
        self.soft = nn.Softmax(dim=1)
        # 深度监督相关层

    def forward(self,A ):
        # 通过backbone提取特征
        x1_1, x1_2, x1_3, x1_4 = self.backbone(A)
        # 通过FPN聚合特征CSA
        x1_2, x1_3, x1_4, x1_5 = self.ccf(x1_1, x1_2, x1_3, x1_4)
        # 特征融合
        c2, c3, c4, c5 = self.ffm(x1_2, x1_3, x1_4, x1_5)
        # 解码器生成多尺度输出
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)
        # 如果启用了深度监督，返回多个掩码
        # 上采样至256x256
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')

        return mask_p2

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers
#
if __name__ == '__main__':
    # 创建输入图像 A（假设它是一个形状为 (batch_size, channels, height, width) 的张量）
    input_A = torch.rand(4, 3, 256, 256).cuda()  # 假设批大小为 4，单通道医学图像大小为 256x256

    # 创建模型并移至 GPU
    model = SemiModel().cuda()

    # 传递输入 A 给模型，获取掩码输出
    mask_p2 = model(input_A)  # 仅传递一个输入

    # 打印输出掩码的尺寸
    print(f"Output mask size: {mask_p2.size()}")
