import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

channel={'RGB':3,
         'Y':1}

class ExtraLayer(nn.Module):
    def __init__(self,input_channel,format_type):
        super().__init__()
        self.extra=False
        self.input_channel=input_channel
        self.output_channel=channel[format_type]

    def SRResNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(self.input_channel,self.output_channel,
                                           kernel_size=9,
                                           stride=1,
                                           padding=4))
        return self

    def SRDenseNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(self.input_channel,self.output_channel,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))
        return self
    
    def RRDBNet(self):
        self.extra=True
        self.layer=nn.Sequential(nn.Conv2d(self.input_channel,64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU(),
                                nn.Conv2d(64,self.output_channel,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.PReLU())
        return self

    def forward(self,x):
        if self.extra==True:
            x=self.layer(x)
        return x      

class FSRCNN(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.output_channel=56
        self.feature_extract=nn.Sequential(nn.Conv2d(channel[format_type],56,
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
        self.expand=nn.Sequential(nn.Conv2d(12,self.output_channel,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                  nn.PReLU())

    def forward(self,x):
        feature_extract=self.feature_extract(x)
        shrink=self.shrink(feature_extract)
        nolinear=self.nonlinear_mapping(shrink)
        expand=self.expand(nolinear)
        return expand


class ESPCN(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.output_channel=32
        self.nonlinear_mapping=nn.Sequential(nn.Conv2d(channel[format_type],64,
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
        nonlinear_mapping=self.nonlinear_mapping(x)
        return nonlinear_mapping


class VDSR(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.conv=nn.Sequential(utils.VDSRBlock(channel[format_type],64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                utils.VDSRBlock(64,64),
                                nn.Conv2d(64,channel[format_type],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1))

    def forward(self,x):
        #input is ILR
        conv=self.conv(x)
        return x+conv


class SRResNet(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.output_channel=64
        self.feature_extract=nn.Sequential(nn.Conv2d(channel[format_type],64,
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
        self.RB=nn.Sequential(utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64),
                              utils.ResidualBlock(64,64))
        self.skip_connection=nn.Sequential(nn.Conv2d(64,self.output_channel,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1))

    def forward(self,x):
        feature_extract=self.feature_extract(x)
        RB=self.RB(feature_extract)
        skip_connection=self.skip_connection(RB)
        return x+skip_connection


class SRDenseNet(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.output_channel=256
        self.lowlevel=nn.Sequential(nn.Conv2d(channel[format_type],128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.PReLU())
        self.DB1=utils.DenseBlock(128)
        self.DB2=utils.DenseBlock(256)
        self.DB3=utils.DenseBlock(384)
        self.DB4=utils.DenseBlock(512)
        self.DB5=utils.DenseBlock(640)
        self.DB6=utils.DenseBlock(768)
        self.DB7=utils.DenseBlock(896)
        self.DB8=utils.DenseBlock(1024)
        self.bottleneck=nn.Sequential(nn.Conv2d(1152,self.output_channel,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.PReLU())

    def forward(self,x):
        lowlevel=self.lowlevel(x)
        DB1=self.DB1(lowlevel)
        concat=torch.cat([lowlevel,DB1],1)
        DB2=self.DB2(concat)
        concat=torch.cat([concat,DB2],1)
        DB3=self.DB3(concat)
        concat=torch.cat([concat,DB3],1)
        DB4=self.DB4(concat)
        concat=torch.cat([concat,DB4],1)
        DB5=self.DB5(concat)
        concat=torch.cat([concat,DB5],1)
        DB6=self.DB6(concat)
        concat=torch.cat([concat,DB6],1)
        DB7=self.DB7(concat)
        concat=torch.cat([concat,DB7],1)
        DB8=self.DB8(concat)
        concat=torch.cat([concat,DB8],1)
        bottleneck=self.bottleneck(concat)
        return bottleneck



class RRDBNet(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.output_channel=64
        self.conv1=nn.Sequential(nn.Conv2d(channel[format_type],64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.RRDB=nn.Sequential(utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32),
                                utils.ResidualResidualDenseBlock(64,32))
        self.conv2=nn.Sequential(nn.Conv2d(64,self.output_channel,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())

    def forward(self,x):
        conv1=self.conv1(x)
        RRDB=self.RRDB(conv1)
        conv2=self.conv2(RRDB)
        return conv1+conv2

class SubPixelLayer(nn.Module):
    def __init__(self,input_channel,output_channel,upscale_factor,format_type):
        super().__init__()
        self.output_channel=output_channel
        self.subpixel=nn.Sequential(nn.Conv2d(input_channel,self.output_channel*channel[format_type]*upscale_factor**2,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.PReLU(),
                                    nn.PixelShuffle(upscale_factor),
                                    nn.PReLU())

    def forward(self,x):
        subpixel=self.subpixel(x)
        return subpixel
    

class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x


class VGG(nn.Module):
    def __init__(self,format_type):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channel[format_type],64,
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
                               nn.Conv2d(1024,1,kernel_size=1))
    
    def forward(self,x):
        net=self.net(x)
        net=torch.squeeze(net,3)
        net=torch.squeeze(net,2)
        return net
