# Basic UNet-3D

import torch as t
from torch import nn
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device1 = t.device('cuda:0')
device2 = t.device('cuda:1')

class CONVx2(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_x_2 = nn.Sequential(
            nn.Conv3d( in_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm3d( out_channels),
            nn.LeakyReLU( inplace=True),
            nn.Conv3d( out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d( out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv_x_2(x)

class Downsampling(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.downstep = nn.Sequential(
            nn.MaxPool3d(2),
            CONVx2(in_channels,out_channels)
        )
    
    def forward(self,x):
        return self.downstep(x)
    
class GridAttentionGateLocal3D(nn.Module):

    def __init__(self, Fg, Fl, Fint, learn_upsampling=False, batchnorm=False):
        super(GridAttentionGateLocal3D, self).__init__()

        if batchnorm:
            self.Wg = nn.Sequential(
                nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(Fint)
            )
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(Fint),
                nn.MaxPool3d(2)
            )

            self.y = nn.Sequential(
                nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(1)
            )

        else:
            self.Wg = nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True)
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.MaxPool3d(2)
            )

            self.y = nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=Fl, out_channels=Fl, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(Fl),
        )

    def forward(self, xl, g):

        xl_size_orig = xl.size()
        xl_ = self.Wx(xl)

        g = self.Wg(g)

        relu = F.relu(xl_ + g, inplace=True)
        y = self.y(relu)
        sigmoid = t.sigmoid(y)

        upsampled_sigmoid = F.interpolate(sigmoid, size=xl_size_orig[2:], mode='trilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_sigmoid.expand_as(xl)

        return self.out(attention * xl)
    

class Upsampling(nn.Module):

    def __init__(self,in_channels,out_channels): #can try out bilinear upsampling too instead
        super().__init__()
        self.att = GridAttentionGateLocal3D(Fg=in_channels, Fl=out_channels, Fint=out_channels)
        self.upsampled = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=2,stride=2)
        self.conv3d = CONVx2(out_channels*2,out_channels)
        
    def forward(self,x1,x2):
        g3 = self.att(x2, x1)
        x1 = self.upsampled(x1)
        # input is CHW
        
        diffZ = g3.size()[2] - x1.size()[2]
        diffY = g3.size()[3] - x1.size()[3]
        diffX = g3.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX//2, diffX -( diffX//2), diffY//2, diffY-(diffY//2) , diffZ//2, diffZ-(diffZ//2)])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = t.cat([g3, x1], dim=1)

        return self.conv3d(x)

class OutputConv(nn.Module):
    def __init__(self,in_channels,out_classes):
        super(OutputConv,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_classes, kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)

class UnetModel(nn.Module):
    def __init__(self,in_channels,out_classes):
        super(UnetModel,self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.cv1 = CONVx2(in_channels,32).to(device1)
        self.down1 = Downsampling(32,64).to(device1)
        self.down2 = Downsampling(64,128).to(device1)
        self.down3 = Downsampling(128,256).to(device1)

        self.up1 = Upsampling(256,128).to(device1)
        self.up2 = Upsampling(128,64).to(device1)
        self.up3 = Upsampling(64,32).to(device2)

        self.cv2 = OutputConv(32,out_classes).to(device2)

    def forward(self,x):
        x1 = self.cv1(x.to(device1))
        x2 = self.down1(x1.to(device1))
        x3 = self.down2(x2.to(device1))
        x4 = self.down3(x3.to(device1))

        x = self.up1(x4.to(device1), x3.to(device1))
        x = self.up2(x.to(device1), x2.to(device1))
        x = self.up3(x.to(device2), x1.to(device2))
        logits = self.cv2(x.to(device2))
        return logits
