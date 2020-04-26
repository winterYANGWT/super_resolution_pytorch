import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#PSNR
def calc_PSNR(img1,img2):
    mse=F.mse_loss(img1,img2)
    return 10*math.log10(1/mse)

#SSIM
def gaussian(window_size,sigma):
    gauss=torch.Tensor([math.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size,channel=1):
    _1D_window=gaussian(window_size,1.5).unsqueeze(1)
    _2D_window=_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window=_2D_window.expand(channel,1,window_size,window_size).contiguous()
    return window

window=create_window(11,1).to(device)

def calc_SSIM(img1,img2,size_average=True):
    L=1
    pad=0
    _,channel,height,width=img1.size()
    mu1=F.conv2d(img1,window,padding=pad,groups=channel).to(device)
    mu2=F.conv2d(img2,window,padding=pad,groups=channel).to(device)
    mu1_sq=mu1.pow(2)
    mu2_sq=mu2.pow(2)
    mu12=mu1*mu2
    sigma1_sq=F.conv2d(img1**2,window,padding=pad,groups=channel).to(device)-mu1_sq
    sigma2_sq=F.conv2d(img2**2,window,padding=pad,groups=channel).to(device)-mu2_sq
    sigma12=F.conv2d(img1*img2,window,padding=pad,groups=channel).to(device)-mu12
    C1=(0.01*L)**2
    C2=(0.03*L)**2
    v1=2.0*sigma12+C2
    v2=2.0*mu12+C1
    v3=sigma1_sq+sigma2_sq+C2
    v4=mu1_sq+mu2_sq+C1
    ssim_map=(v1*v2)/(v3*v4)
    if size_average:
        ret=ssim_map.mean()
    else:
        ret=ssim_map.mean(1).mean(1).mean(1)
    return ret.item()

