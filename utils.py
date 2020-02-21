import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import torchvision.transforms as transforms
from PIL import Image

class Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Residual_Block,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.PReLU(),
                                nn.Conv2d(out_channels,out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1))
        self.activation=nn.PReLU()

    def forward(self,x):
        x_copy=x
        x=self.conv(x)
        x=self.activation(x+x_copy)
        return x


class Dense_Block(nn.Module):
    pass


#train and test
def calc_PSNR(mse):
    return 10*math.log10(1/mse)

def gaussian(window_size,sigma):
    gauss=torch.Tensor([math.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size,channel=1):
    _1D_window=gaussian(window_size,1.5).unsqueeze(1)
    _2D_window=_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window=_2D_window.expand(channel,1,window_size,window_size).contiguous()
    return window

window=create_window(11,1)

def calc_SSIM(img1,img2):
    L=1
    pad=0
    _,channel,height,width=img1.size()
    mu1=F.conv2d(img1,window,padding=pad,groups=channel)
    mu2=F.conv2d(img2,window,padding=pad,groups=channel)
    mu1_sq=mu1.pow(2)
    mu2_sq=mu2.pow(2)
    mu12=mu1*mu2
    sigma1_sq=F.conv2d(img1**2,window,padding=pad,groups=channel)-mu1_sq
    sigma2_sq=F.conv2d(img2**2,window,padding=pad,groups=channel)-mu2_sq
    sigma12=F.conv2d(img1*img2,window,padding=pad,groups=channel)-mu12
    C1=(0.01*L)**2
    C2=(0.03*L)**2
    v1=2.0*sigma12+C2
    v2=2.0*mu12+C1
    v3=sigma1_sq+sigma2_sq+C2
    v4=mu1_sq+mu2_sq+C1
    cs=torch.mean(v1/v2)
    ssim_map=(v1*v2)/(v3*v4)
    if size_average:
        ret=ssim_map.mean()
    else:
        ret=ssim_map.mean(1).mean(1).mean(1)
    return ret

#dataset
class RandomSelectedRotation(object):
    def __init__(self,select_list):
        super(RandomSelectedRotation,self).__init__()
        assert isinstance(select_list,list)
        self.select_list=select_list
 
    def __call__(self,sample):
        angle=random.choice(self.select_list)
        return transforms.functional.rotate(sample,angle)


class Normalize(object):
    def __init__(self):
        super(Normalize,self).__init__()

    def __call__(self,sample):
        return sample/255.0


def RGB2Y(image):
    image=np.array(image).astype(np.float32)
    image_Y=16+0.2568*image[...,0]+0.5041*image[...,1]+0.0978*image[...,2]
    return image_Y

def RGB2YCbCr(image):
    image=np.array(image).astype(np.float32)
    image_Y=16+0.2568*image[...,0]+0.5041*image[...,1]+0.0978*image[...,2]
    image_Cb=128-0.1482*image[...,0]-0.2910*image[...,1]+0.4392*image[...,2]
    image_Cr=128+0.4392*image[...,0]-0.3678*image[...,1]-0.0714*image[...,2]
    return (image_Y,image_Cb,image_Cr)

def YCbCr2RGB(image):
    image=np.array(image).astype(np.float32)
    image_R=1.1644*(image[...,0]-16)+1.5960*(image[...,2]-128)
    image_G=1.1644*(image[...,0]-16)-0.3918*(image[...,1]-128)-0.8130*(image[...,2]-128)
    image_R=1.1644*(image[...,0]-16)+2.0172*(image[...,1]-128)
    return (image_R,image_G,image_B)

