import torch
import torch.nn as nn
from ..backbone.common import Conv, RepNCSPELAN4, Concat, AConv
from utils.general import make_divisible

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)

class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))

class YoloV9DualTinyNeck(nn.Module):
    # TODO : activations
    def __init__(self, cfg):
        super(YoloV9DualTinyNeck, self).__init__()

        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        input_p3, input_p4, input_p5 = cfg.Model.Neck.in_channels
        output_p3, output_p4, output_p5, output_p6, output_p7, output_p8 = cfg.Model.Neck.out_channels

        self.channels = {
            'input_p3': input_p3,
            'input_p4': input_p4,
            'input_p5': input_p5,

            'output_p3': output_p3,
            'output_p4': output_p4,
            'output_p5': output_p5,
            'output_p6': output_p6,
            'output_p7': output_p7,
            'output_p8': output_p8
        }

        self.re_channels_out()

        self.input_p3 = self.channels['input_p3'] # 4, refer to yolov7.yaml from https://github.com/WongKinYiu/yolov7
        self.input_p4 = self.channels['input_p4'] # 6
        self.input_p5 = self.channels['input_p5'] # 8

        self.output_p3 = self.channels['output_p3']
        self.output_p4 = self.channels['output_p4']
        self.output_p5 = self.channels['output_p5']
        self.output_p6 = self.channels['output_p6']
        self.output_p7 = self.channels['output_p7']
        self.output_p8 = self.channels['output_p8']

        if cfg.Model.Neck.activation == 'SiLU':
            CONV_ACT = 'silu'
        elif cfg.Model.Neck.activation == 'ReLU':
            CONV_ACT = 'relu'
        elif cfg.Model.Neck.activation == 'LeakyReLU': 
            CONV_ACT = 'lrelu'
        else:
            CONV_ACT = 'hard_swish'

        self.concat = Concat()
        self.sppelan0 = SPPELAN(self.input_p5, 128, 64) # 9

        self.upSamp0 = nn.Upsample(None, 2, 'nearest')
        self.repncspelan0 = RepNCSPELAN4(128 + self.input_p4, 96, 96, 48, 3, act=CONV_ACT) # 12, input from concat upsampled 9 and 6 (upsampling doesn't change channel dimension)

        self.upSamp1 = nn.Upsample(None, 2, 'nearest')
        self.repncspelan1 = RepNCSPELAN4(96 + self.input_p3, 64, 64, 32, 3, act=CONV_ACT) # input from concat upsampled 12 and 4
        self.a0 = AConv(64, 48)

        self.repncspelan2 = RepNCSPELAN4(48 + 96, 96, 96, 48, 3, act=CONV_ACT) # prev and 12
        self.a1 = AConv(96, 64)

        self.repncspelan3 = RepNCSPELAN4(64 + 128, 128, 128, 64, 3, act=CONV_ACT)
        self.sppelan1 = SPPELAN(self.input_p5, 128, 64) # from 8, in backbone

        self.upSamp2 = nn.Upsample(None, 2, 'nearest')
        self.repncspelan4 = RepNCSPELAN4(128 + self.input_p4, 96, 96, 48, 3, act=CONV_ACT) # from concat upsampled 22 and 6

        self.upSamp3 = nn.Upsample(None, 2, 'nearest')
        self.repncspelan5 = RepNCSPELAN4(96 + self.input_p3, 64, 64, 32, 3, act=CONV_ACT) # from concat upsampled 25 and 4

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        small, med, large = inputs

        x0 = self.sppelan0(large) # 9
        x1 = self.upSamp0(x0)
        x2 = self.concat([x1, med])

        x3 = self.repncspelan0(x2) # 12
        x4 = self.upSamp1(x3)
        x5 = self.concat([x4, small])

        x6 = self.repncspelan1(x5) # 15
        x7 = self.a0(x6)
        x8 = self.concat([x7, x3])

        x9 = self.repncspelan2(x8) # 18
        x10 = self.a1(x9)
        x11 = self.concat([x10, x0])

        x12 = self.repncspelan3(x11) # 21

        x13 = self.sppelan1(large)
        x14 = self.upSamp2(x13)
        x15 = self.concat([x14, med])

        x16 = self.repncspelan4(x15) # 25
        x17 = self.upSamp3(x16)
        x18 = self.concat([x17, small])

        x19 = self.repncspelan5(x18) # 28

        return x19, x16, x13, x6, x9, x12 # 28, 25, 22, 15, 18, 21