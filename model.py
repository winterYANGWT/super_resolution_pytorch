import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Extra_Layer(nn.Module):
    def __init__(self):
        super(Extra_Layer,self).__init__()
        self.extra=False

    def SRResNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(1,1,
                                           kernel_size=9,
                                           stride=1,
                                           padding=4))
        return self

    def forward(self,x):
        if self.extra==True:
            x=self.layer(x)
        return x      

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN,self).__init__()
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
        super(ESPCN,self).__init__()
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
        super(VDSR,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Conv2d(1,64,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1))

    def forward(self,x):
        #input is ILR
        copy=x
        x=self.conv(x)
        x=x+copy
        return x


class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet,self).__init__()
        self.output_channel=64
        self.feature_extract=nn.Sequential(nn.Conv2d(1,8,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(8,16,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(16,32,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU(),
                                           nn.Conv2d(32,64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                           nn.PReLU())
        self.residual_block=nn.Sequential(utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64),
                                          utils.Residual_Block(64,64))
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
        
        

class Model2(nn.Module):
    def __init__(self):
        super(Model2,self).__init__()

    def forward(self,x):
        pass


class Subpixel_Layer(nn.Module):
    def __init__(self,input_channel,upscale_factor):
        super(Subpixel_Layer,self).__init__()
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
    
