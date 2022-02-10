import torch
from torch import nn
from torch.nn import functional as F


class convBlock(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(convBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(outChannel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=outChannel, out_channels=outChannel, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(outChannel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class downSample(nn.Module):
    def __init__(self, channel):
        super(downSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class upSample(nn.Module):
    def __init__(self, channel):
        super(upSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, featureMap):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, featureMap), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = convBlock(1, 64)
        self.d1 = downSample(64)
        self.c2 = convBlock(64, 128)
        self.d2 = downSample(128)
        self.c3 = convBlock(128, 256)
        self.d3 = downSample(256)
        self.c4 = convBlock(256, 512)
        self.d4 = downSample(512)
        self.c5 = convBlock(512, 1024)
        self.u1 = upSample(1024)
        self.c6 = convBlock(1024, 512)
        self.u2 = upSample(512)
        self.c7 = convBlock(512, 256)
        self.u3 = upSample(256)
        self.c8 = convBlock(256, 128)
        self.u4 = upSample(128)
        self.c9 = convBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 3, 1, 1)
        self.th = nn.Sigmoid()

    def forward(self, x):
        L1 = self.c1(x)
        L2 = self.c2(self.d1(L1))
        L3 = self.c3(self.d2(L2))
        L4 = self.c4(self.d3(L3))
        L5 = self.c5(self.d4(L4))
        R4 = self.c6(self.u1(L5, L4))
        R3 = self.c7(self.u2(R4, L3))
        R2 = self.c8(self.u3(R3, L2))
        R1 = self.c9(self.u4(R2, L1))

        return self.th(self.out(R1))

class UNet_three_in(nn.Module):
    def __init__(self):
        super(UNet_three_in, self).__init__()
        self.c1 = convBlock(3, 64)
        self.d1 = downSample(64)
        self.c2 = convBlock(64, 128)
        self.d2 = downSample(128)
        self.c3 = convBlock(128, 256)
        self.d3 = downSample(256)
        self.c4 = convBlock(256, 512)
        self.d4 = downSample(512)
        self.c5 = convBlock(512, 1024)
        self.u1 = upSample(1024)
        self.c6 = convBlock(1024, 512)
        self.u2 = upSample(512)
        self.c7 = convBlock(512, 256)
        self.u3 = upSample(256)
        self.c8 = convBlock(256, 128)
        self.u4 = upSample(128)
        self.c9 = convBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 3, 1, 1)
        self.th = nn.Sigmoid()

    def forward(self, x):
        L1 = self.c1(x)
        L2 = self.c2(self.d1(L1))
        L3 = self.c3(self.d2(L2))
        L4 = self.c4(self.d3(L3))
        L5 = self.c5(self.d4(L4))
        R4 = self.c6(self.u1(L5, L4))
        R3 = self.c7(self.u2(R4, L3))
        R2 = self.c8(self.u3(R3, L2))
        R1 = self.c9(self.u4(R2, L1))

        return self.th(self.out(R1))




if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = convBlock(3,64).to('cpu')
    x = x.to('cpu')
    print(net(x).shape)