import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Extra_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.extra=False

    def SRResNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(1,1,
                                           kernel_size=9,
                                           stride=1,
                                           padding=4))
        return self

    def SRDenseNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(1,1,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))
        return self

    def forward(self,x):
        if self.extra==True:
            x=self.layer(x)
        return x      

class FSRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_channel=56
        self.feature_extract=nn.Sequential(nn.Conv2d(1,56,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2),
                                           nn.PReLU())
        self.shrink=nn.Sequential(nn.Conv2d(56,12,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                            nn.PReLU())
        self.nonlinear_mapping=nn.Sequential(nn.Conv2d(12,12,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.PReLU(),
                                            nn.Conv2d(12,12,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.PReLU(),
                                            nn.Conv2d(12,12,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.PReLU(),
                                            nn.Conv2d(12,12,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.PReLU())
        self.expand=nn.Sequential(nn.Conv2d(12,
                                            self.output_channel,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                  nn.PReLU())

    def forward(self,x):
        x=self.feature_extract(x)
        x=self.shrink(x)
        x=self.nonlinear_mapping(x)
        x=self.expand(x)
        return x


class ESPCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_channel=32
        self.nonlinear_mapping=nn.Sequential(nn.Conv2d(1,64,
                                                       kernel_size=5,
                                                       stride=1,
                                                       padding=2),
                                             nn.Tanh(),
                                             nn.Conv2d(64,self.output_channel,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1),
                                            nn.Tanh())

    def forward(self,x):
        x=self.nonlinear_mapping(x)
        return x


class VDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(utils.VDSR_Block(1,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                utils.VDSR_Block(64,64),
                                nn.Conv2d(64,1,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1))

    def forward(self,x):
        #input is ILR
        x_copy=x
        x=self.conv(x)
        x=x+x_copy
        return x


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_channel=64
        self.feature_extract=nn.Sequential(nn.Conv2d(1,64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(64,64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(64,64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(64,64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU())
        self.residual_block=nn.Sequential(utils.Residual_Block(64),
                                          utils.Residual_Block(64),
                                          utils.Residual_Block(64),
                                          utils.Residual_Block(64),
                                          utils.Residual_Block(64))
        self.skip_connection=nn.Sequential(nn.Conv2d(64,self.output_channel,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1))

    def forward(self,x):
        x=self.feature_extract(x)
        x_copy=x
        x=self.residual_block(x)
        x=self.skip_connection(x)
        x=x+x_copy
        return x


class SRDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_channel=256
        self.lowlevel=nn.Sequential(nn.Conv2d(1,128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.PReLU())
        self.dense1=utils.Dense_Block(128)
        self.dense2=utils.Dense_Block(256)
        self.dense3=utils.Dense_Block(384)
        self.dense4=utils.Dense_Block(512)
        self.dense5=utils.Dense_Block(640)
        self.dense6=utils.Dense_Block(768)
        self.dense7=utils.Dense_Block(896)
        self.dense8=utils.Dense_Block(1024)
        self.bottleneck=nn.Sequential(nn.Conv2d(1152,self.output_channel,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.PReLU())

    def forward(self,x):
        residual=self.lowlevel(x)
        x=self.dense1(residual)
        concat=torch.cat([residual,x],1)
        x=self.dense2(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense3(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense4(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense5(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense6(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense7(concat)
        concat=torch.cat([concat,x],1)
        x=self.dense8(concat)
        concat=torch.cat([concat,x],1)
        x=self.bottleneck(concat)
        return x


class Subpixel_Layer(nn.Module):
    def __init__(self,input_channel,upscale_factor):
        super().__init__()
        self.subpixel=nn.Sequential(nn.Conv2d(input_channel,
                                              upscale_factor**2,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.PReLU(),
                                    nn.PixelShuffle(upscale_factor),
                                    nn.PReLU())

    def forward(self,x):
        x=self.subpixel(x)
        return x
    

class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequetial(nn.Conv2d(1,64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(64,64,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),
                              nn.BatchNorm2d(64),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(64,128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(128),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(128,128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),
                              nn.BatchNorm2d(128),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(128,256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(256),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(256,256,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),
                              nn.BatchNorm2d(256),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(256,512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(512),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(512,512,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(512),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Conv2d(512,1024,kernel_size=1),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Conv2d(1024,1,kernel_size=1),
                              nn.Sigmoid())
    
        def forward(self,x):
            x=self.net(x)
            return x
