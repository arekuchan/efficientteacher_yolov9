from models.backbone.common import *

class ELAN1(nn.Module):
    def __init__(self, c1, c2, c3, c4, act="silu"):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1, act=act)
        self.cv2 = Conv(c3//2, c4, 3, 1, act=act)
        self.cv3 = Conv(c4, c4, 3, 1, act=act)
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1, act=act)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    
class GELAN(nn.Module):
    def __init__(self, CONV_ACT="silu"):
        super(GELAN, self).__init__()

        self.cv0 = Conv(3, 16, 3, 2, act=CONV_ACT)
        self.cv1 = Conv(16, 32, 3, 2, act=CONV_ACT)
        self.e0 = ELAN1(32, 32, 32, 16, act=CONV_ACT)
        self.a0 = AConv(32, 64, act=CONV_ACT)
        self.repNCS0 = RepNCSPELAN4(64, 64, 32, 3, act=CONV_ACT)
        self.a1 = AConv(64, 96, act=CONV_ACT)
        self.repNCS1 = RepNCSPELAN4(96, 96, 48, 3, act=CONV_ACT)
        self.a2 = AConv(96, 128, act=CONV_ACT)
        self.repNCS2 = RepNCSPELAN4(128, 128, 64, 3, act=CONV_ACT)

    def forward(self, x):
        x0 = self.cv0(x)
        x1 = self.cv1(x0)
        x2 = self.e0(x1)

        x3 = self.a0(x2)
        x4 = self.repNCS0(x3)

        x5 = self.a1(x4)
        x6 = self.repNCS1(x5)

        x7 = self.a2(x6)
        x8 = self.repNCS2(x7)

        return x4, x6, x8 # feature pyramid, large, med, small

class YoloV9TinyBackBone(nn.Module):
    def __init__(self, cfg):
        super(YoloV9TinyBackBone, self).__init__()

        if cfg.Model.Backbone.activation == 'SiLU': 
            self.CONV_ACT = 'silu'
            self.C_ACT = 'silu'
        elif cfg.Model.Backbone.activation == 'ReLU': 
            self.CONV_ACT = 'relu'
            self.C_ACT = 'relu'
        else:
            self.CONV_ACT = 'hard_swish'
            self.C_ACT = 'hard_swish'

        self.gelan = GELAN(CONV_ACT=self.CONV_ACT)

    def forward(self, x):
        return self.gelan(x)