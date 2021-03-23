import torch
import torch.nn as nn
from torchsummary import summary
from models import RRDBNet_SENet
from utils import get_GaussKernel_2
from config import Config

CONFIG = Config()
device = CONFIG.DEVICE

class Reconstruction(nn.Module):
    def __init__(self, inp, oup, kernel=3, strd=1):
        super(Reconstruction, self).__init__()
        self.Rec = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel, stride=strd, padding=1),
                nn.PReLU(num_parameters=inp),
                nn.Conv2d(inp, oup, kernel_size=1, stride=strd, padding=0),
                nn.PReLU(num_parameters=oup),
            )
    def forward(self,x):
        out = self.Rec(x)
        return out

class Conv_1x1(nn.Module):
    def __init__(self, inp, oup, strd=1):
        super(Conv_1x1, self).__init__()
        self.convDepthwise = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=strd),
                nn.PReLU(num_parameters=oup),
            )
    def forward(self,x):
        out = self.convDepthwise(x)
        return out


class Upsamplingx2(nn.Module):
    def __init__(self, inp, oup):
        super(Upsamplingx2, self).__init__()
        self.FinalUp = nn.ConvTranspose2d(in_channels=inp, out_channels=oup, padding = 1, output_padding=1, kernel_size=3, stride=2)
    def forward(self,x):
        out = self.FinalUp(x)
        return out

class DMNestUnet(nn.Module):
    def __init__(self, in_channels, n_classes, deep_supervision=CONFIG.DeepSupervision, gaussN=CONFIG.GaussSmoothingKernel, sigma=CONFIG.GaussFilterSigma):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.gauss_padding = gaussN//2
        filters = [8, 16, 32, 64, 128]
        conv_kernel_size = [3, 5, 7, 9]
        # j == 0
        self.x_00 = RRDBNet_SENet(in_nc=in_channels, nf=filters[0], gc=in_channels, ker_conv=conv_kernel_size[0])
        self.kernel2d = get_GaussKernel_2(gaussN, sigma)
        self.kernel = torch.cat([self.kernel2d, self.kernel2d, self.kernel2d, self.kernel2d], 0).to(device)
        self.x_01 = Reconstruction(filters[0] * 2, filters[0])
        self.x_02 = Reconstruction(filters[0] * 3, filters[0])
        self.x_03 = Reconstruction(filters[0] * 4, filters[0])
        self.x_04 = Reconstruction(filters[0] * 5, filters[0])

        self.up_10_to_01 = Conv_1x1(filters[1], filters[0])
        self.up_11_to_02 = Conv_1x1(filters[1], filters[0])
        self.up_12_to_03 = Conv_1x1(filters[1], filters[0])
        self.up_13_to_04 = Conv_1x1(filters[1], filters[0])
        # j == 1
        self.x_10 = RRDBNet_SENet(in_nc=in_channels, nf=filters[1], gc=filters[0], ker_conv=conv_kernel_size[1])

        self.x_11 = Reconstruction(filters[1] * 2, filters[1])
        self.x_12 = Reconstruction(filters[1] * 3, filters[1])
        self.x_13 = Reconstruction(filters[1] * 4, filters[1])

        self.up_20_to_11 = Conv_1x1(filters[2], filters[1])
        self.up_21_to_12 = Conv_1x1(filters[2], filters[1])
        self.up_22_to_13 = Conv_1x1(filters[2], filters[1])

        # j == 2
        self.x_20 = RRDBNet_SENet(in_nc=in_channels, nf=filters[2], gc=filters[1], ker_conv=conv_kernel_size[1])

        self.x_21 = Reconstruction(filters[2] * 2, filters[2])
        self.x_22 = Reconstruction(filters[2] * 3, filters[2])

        self.up_30_to_21 = Conv_1x1(filters[3], filters[2])
        self.up_31_to_22 = Conv_1x1(filters[3], filters[2])

        # j == 3
        self.x_30 = RRDBNet_SENet(in_nc=in_channels, nf=filters[3], gc=filters[2], ker_conv=conv_kernel_size[2])

        self.x_31 = Reconstruction(filters[3] * 2, filters[3])

        self.up_40_to_31 = Conv_1x1(filters[4], filters[3])
        # j == 4
        self.x_40 = RRDBNet_SENet(in_nc=in_channels, nf=filters[4], gc=filters[3], ker_conv=conv_kernel_size[2])



        self.final_upSample = Upsamplingx2(filters[0], n_classes)

    def forward(self, inputs, L= CONFIG.DMUnetL):

        if not (1 <= L <= 4):
            raise ValueError("the model pruning factor `L` should be 1 <= L <= 4")

        x_00_output = self.x_00(inputs)
        # x_10_down = self.pool(inputs)
        x_10_down = nn.functional.conv2d(inputs, self.kernel, stride=1, padding=self.gauss_padding, groups=4)
        x_10_output = self.x_10(x_10_down)
        x_10_up_sample = self.up_10_to_01(x_10_output)
        x_01_output = self.x_01(torch.cat([x_00_output, x_10_up_sample], 1))
        nestnet_output_1 = self.final_upSample(x_01_output)

        if L == 1:
            return nestnet_output_1

        # x_20_down = self.pool(x_10_down)
        x_20_down = nn.functional.conv2d(x_10_down, self.kernel, stride=1, padding=self.gauss_padding, groups=4)
        x_20_output = self.x_20(x_20_down)
        x_20_up_sample = self.up_20_to_11(x_20_output)
        x_11_output = self.x_11(torch.cat([x_10_output, x_20_up_sample], 1))
        x_11_up_sample = self.up_11_to_02(x_11_output)
        x_02_output = self.x_02(torch.cat([x_00_output, x_01_output, x_11_up_sample], 1))
        nestnet_output_2 = self.final_upSample(x_02_output)

        if L == 2:
            if self.deep_supervision:
                # return the average of output layers
                return [nestnet_output_1, nestnet_output_2]
            else:
                return nestnet_output_2

        # x_30_down = self.pool(x_20_down)
        x_30_down = nn.functional.conv2d(x_20_down, self.kernel, stride=1, padding=self.gauss_padding, groups=4)
        x_30_output = self.x_30(x_30_down)
        x_30_up_sample = self.up_30_to_21(x_30_output)
        x_21_output = self.x_21(torch.cat([x_20_output, x_30_up_sample], 1))
        x_21_up_sample = self.up_21_to_12(x_21_output)
        x_12_output = self.x_12(torch.cat([x_10_output, x_11_output, x_21_up_sample], 1))
        x_12_up_sample = self.up_12_to_03(x_12_output)
        x_03_output = self.x_03(torch.cat([x_00_output, x_01_output, x_02_output, x_12_up_sample], 1))
        nestnet_output_3 = self.final_upSample(x_03_output)

        if L == 3:
            # return the average of output layers
            if self.deep_supervision:
                return [nestnet_output_1, nestnet_output_2, nestnet_output_3]
            else:
                return nestnet_output_3

        # x_40_down = self.pool(x_30_down)
        x_40_down = nn.functional.conv2d(x_30_down, self.kernel, stride=1, padding=self.gauss_padding, groups=4)
        x_40_output = self.x_40(x_40_down)
        x_40_up_sample = self.up_40_to_31(x_40_output)
        x_31_output = self.x_31(torch.cat([x_30_output, x_40_up_sample], 1))
        x_31_up_sample = self.up_31_to_22(x_31_output)
        x_22_output = self.x_22(torch.cat([x_20_output, x_21_output, x_31_up_sample], 1))
        x_22_up_sample = self.up_22_to_13(x_22_output)
        x_13_output = self.x_13(torch.cat([x_10_output, x_11_output, x_12_output, x_22_up_sample], 1))
        x_13_up_sample = self.up_13_to_04(x_13_output)
        x_04_output = self.x_04(torch.cat([x_00_output, x_01_output, x_02_output, x_03_output, x_13_up_sample], 1))
        nestnet_output_4 = self.final_upSample(x_04_output)

        if L == 4:
            if self.deep_supervision:
                # return the average of output layers
                return [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4]
            else:
                return nestnet_output_4



if __name__ == '__main__':

    from torchsummary import summary
    device = torch.device('cuda')
    net = DMNestUnet(in_channels=4, n_classes=3).to(device)
    summary(net, (4, 64, 64), device='cuda')