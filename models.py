import torch.nn as nn
import torch
from config import Config
from torchsummary import summary
import torch.nn.functional as F

CONFIG = Config()
device = CONFIG.DEVICE

class Convdw(nn.Module):
    def __init__(self, inp, oup, kernel=3):
        super(Convdw, self).__init__()
        self.convDepthwise = nn.Sequential(
                nn.Conv2d(inp, inp, kernel, stride= 1, padding=kernel//2, groups=inp),
                nn.PReLU(num_parameters=inp),
                nn.Conv2d(inp, oup, 1, stride=1, padding=0),
                # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
    def forward(self,x):
        out = self.convDepthwise(x)
        # print(out.mean())
        return out

class ResidualDenseBlock_3C_dp(nn.Module):
    def __init__(self, nf, gc, ker=3):
        super(ResidualDenseBlock_3C_dp, self).__init__()
        # gc: growth channel
        self.conv1 = Convdw(nf, gc, ker)
        self.conv2 = Convdw(nf + gc, gc, ker)
        self.conv3 = Convdw(nf + 2 * gc, nf, ker)
        self.prelu = nn.PReLU(num_parameters=gc)


    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.prelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x



class RRDBblock_SENet(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc, factor=8, ker = 3):
        super(RRDBblock_SENet, self).__init__()
        self.RDB1 = ResidualDenseBlock_3C_dp(nf, gc, ker)
        self.RDB2 = ResidualDenseBlock_3C_dp(nf, gc)
        self.RDB3 = ResidualDenseBlock_3C_dp(nf, gc)
        self.SElayer1 = nn.Conv2d(nf, nf//factor, kernel_size=1)
        self.SElayer2 = nn.Conv2d(nf//factor, nf, kernel_size=1)

    def forward(self, x):
        out1 = self.RDB1(x)
        out2 = self.RDB2(out1)
        out3 = self.RDB3(out2)
        w = F.avg_pool2d(out3, (out3.shape[2],out3.shape[3]))
        w = F.relu(self.SElayer1(w))
        w = F.sigmoid(self.SElayer2(w))
        out = out3*w
        return (out + out2 + out1 )* 0.2 + x

class RRDBNet_SENet(nn.Module):
    def __init__(self, in_nc=4, nf=16, gc=8, ker_conv=3, ker_RRDB=3):
        super(RRDBNet_SENet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, ker_conv, padding=ker_conv//2, stride=1, bias=True)
        self.RRDB_block = RRDBblock_SENet(nf, gc, ker=ker_RRDB)
        self.prelu = nn.PReLU(num_parameters=nf)

    def forward(self, x):
        fea = self.conv_first(x)
        fea = self.RRDB_block(fea)
        out = self.prelu(fea)
        return out

if __name__ == "__main__":

    NET = RRDBNet_SENet().to(device)
    summary(NET, (4, 128, 128), device = 'cuda')