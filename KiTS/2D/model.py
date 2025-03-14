import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()
        
        self.opt = opt
        self.in_ch = self.opt.in_ch
        self.out_ch = self.opt.out_ch
        self.n_features = self.opt.num_features
        self.bilinear = False
        
        # factor = 2 if self.bilinear else 1
        
        self.inc0 = nn.Sequential(nn.Conv2d(self.in_ch, self.n_features * 1, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 1), nn.ReLU(inplace=True))
        self.inc1 = nn.Conv2d(self.n_features * 1, self.n_features * 1, kernel_size=3, padding=1)
        
        self.pool1 = nn.MaxPool2d(2)
        self.convD11 = nn.Sequential(nn.Conv2d(self.n_features * 1, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convD12 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convD13 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convD14 = nn.Conv2d(self.n_features * 2, self.n_features * 2, kernel_size=3, padding=1)
        
        self.pool2 = nn.MaxPool2d(2)
        self.convD21 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convD22 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convD23 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convD24 = nn.Conv2d(self.n_features * 4, self.n_features * 4, kernel_size=3, padding=1)
        
        self.pool3 = nn.MaxPool2d(2)
        self.convD31 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 8, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 8), nn.ReLU(inplace=True))
        self.convD32 = nn.Sequential(nn.Conv2d(self.n_features * 8, self.n_features * 8, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 8), nn.ReLU(inplace=True))
        self.convD33 = nn.Sequential(nn.Conv2d(self.n_features * 8, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convD34 = nn.Conv2d(self.n_features * 4, self.n_features * 4, kernel_size=3, padding=1)
        
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.ConvTranspose2d(self.n_features * 4, self.n_features * 4, 2, stride=2, padding=0)
        self.convU11 = nn.Sequential(nn.Conv2d(self.n_features * 8, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convU12 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 4), nn.ReLU(inplace=True))
        self.convU13 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convU14 = nn.Conv2d(self.n_features * 2, self.n_features * 2, kernel_size=3, padding=1)
        
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.ConvTranspose2d(self.n_features * 2, self.n_features * 2, 2, stride=2, padding=0)
        self.convU21 = nn.Sequential(nn.Conv2d(self.n_features * 4, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convU22 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 2), nn.ReLU(inplace=True))
        self.convU23 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 1, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 1), nn.ReLU(inplace=True))
        self.convU24 = nn.Conv2d(self.n_features * 1, self.n_features * 1, kernel_size=3, padding=1)
        
        # self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.ConvTranspose2d(self.n_features * 1, self.n_features * 1, 2, stride=2, padding=0)
        self.convU31 = nn.Sequential(nn.Conv2d(self.n_features * 2, self.n_features * 1, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 1), nn.ReLU(inplace=True))
        self.convU32 = nn.Sequential(nn.Conv2d(self.n_features * 1, self.n_features * 1, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 1), nn.ReLU(inplace=True))
        self.convU33 = nn.Sequential(nn.Conv2d(self.n_features * 1, self.n_features * 1, kernel_size=3, padding=1), nn.BatchNorm2d(self.n_features * 1), nn.ReLU(inplace=True))
        self.convU34 = nn.Conv2d(self.n_features * 1, self.n_features * 1, kernel_size=3, padding=1)
        
        self.outc = nn.Sequential(nn.Conv2d(self.n_features * 1, self.out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        # self.res = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.inc0(x)
        x0 = self.inc1(x0)
        
        xD1 = self.pool1(x0)
        xD1 = self.convD11(xD1)
        xD1 = self.convD12(xD1)
        xD1 = self.convD13(xD1)
        xD1 = self.convD14(xD1)
        
        xD2 = self.pool2(xD1)
        xD2 = self.convD21(xD2)
        xD2 = self.convD22(xD2)
        xD2 = self.convD23(xD2)
        xD2 = self.convD24(xD2)
        
        xD3 = self.pool3(xD2)
        xD3 = self.convD31(xD3)
        xD3 = self.convD32(xD3)
        xD3 = self.convD33(xD3)
        xD3 = self.convD34(xD3)
        
        xU1 = self.up1(xD3)
        # diffY = xD2.size()[2] - xU1.size()[2]
        # diffX = xD2.size()[3] - xU1.size()[3]
        # xU1 = F.pad(xU1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        xU1 = self.convU11(torch.cat((xD2,xU1), dim=1))
        xU1 = self.convU12(xU1)
        xU1 = self.convU13(xU1)
        xU1 = self.convU14(xU1)
        
        xU2 = self.up2(xU1)
        # diffY = xD1.size()[2] - xU2.size()[2]
        # diffX = xD1.size()[3] - xU2.size()[3]
        # xU2 = F.pad(xU2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        xU2 = self.convU21(torch.cat((xD1,xU2), dim=1))
        xU2 = self.convU22(xU2)
        xU2 = self.convU23(xU2)
        xU2 = self.convU24(xU2)
        
        xU3 = self.up3(xU2)
        # diffY = x0.size()[2] - xU3.size()[2]
        # diffX = x0.size()[3] - xU3.size()[3]
        # xU3 = F.pad(xU3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        xU3 = self.convU31(torch.cat((x0,xU3), dim=1))
        xU3 = self.convU32(xU3)
        xU3 = self.convU33(xU3)
        xU3 = self.convU34(xU3)
        
        xout = self.outc(xU3)
        
        # return torch.sigmoid(xout)
        
        return torch.softmax(xout, dim=1)