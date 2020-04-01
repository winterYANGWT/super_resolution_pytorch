import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import torchvision.transforms as transforms
from PIL import Image

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VDSRBlock(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv=nn.Conv2d(input_size,output_size,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.activation=nn.PReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(input_size,output_size,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.PReLU(),
                                nn.Conv2d(output_size,output_size,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1))
        self.activation=nn.PReLU()

    def forward(self,x):
        x_copy=x
        x=self.conv(x)
        x=self.activation(x+x_copy)
        return x


class DenseBlock(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(input_size,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv2=nn.Sequential(nn.Conv2d(16,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv3=nn.Sequential(nn.Conv2d(32,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv4=nn.Sequential(nn.Conv2d(48,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv5=nn.Sequential(nn.Conv2d(64,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv6=nn.Sequential(nn.Conv2d(80,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv7=nn.Sequential(nn.Conv2d(96,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())
        self.conv8=nn.Sequential(nn.Conv2d(112,16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.PReLU())

    def forward(self,x):
        conv1=self.conv1(x)
        conv2=self.conv2(conv1)
        conv2_dense=torch.cat([conv1,conv2],1)
        conv3=self.conv3(conv2_dense)
        conv3_dense=torch.cat([conv2_dense,conv3],1)
        conv4=self.conv4(conv3_dense)
        conv4_dense=torch.cat([conv3_dense,conv4],1)
        conv5=self.conv5(conv4_dense)
        conv5_dense=torch.cat([conv4_dense,conv5],1)
        conv6=self.conv6(conv5_dense)
        conv6_dense=torch.cat([conv5_dense,conv6],1)
        conv7=self.conv7(conv6_dense)
        conv7_dense=torch.cat([conv6_dense,conv7],1)
        conv8=self.conv8(conv7_dense)
        conv8_dense=torch.cat([conv7_dense,conv8],1)
        return conv8_dense


class ResidualDenseBlock(nn.Module):
    def __init__(self,nf=64,gc=32):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(nf,gc,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())
        self.conv2=nn.Sequential(nn.Conv2d(nf+1*gc,gc,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())
        self.conv3=nn.Sequential(nn.Conv2d(nf+2*gc,gc,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())
        self.conv4=nn.Sequential(nn.Conv2d(nf+3*gc,gc,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())
        self.conv5=nn.Sequential(nn.Conv2d(nf+4*gc,nf,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                nn.PReLU())
        
    def forward(self,x):
        conv1=self.conv1(x)
        conv1_dense=torch.cat([x,conv1],1)
        conv2=self.conv2(conv1_dense)
        conv2_dense=torch.cat([conv1_dense,conv2],1)
        conv3=self.conv3(conv2_dense)
        conv3_dense=torch.cat([conv2_dense,conv3],1)
        conv4=self.conv4(conv3_dense)
        conv4_dense=torch.cat([conv3_dense,conv4],1)
        conv5=self.conv5(conv4_dense)
        return conv5*0.2+x


class ResidualResidualDenseBlock(nn.Module):
    def __init__(self,nf,gc=32):
        self.RDB1=ResidualDenseBlock(nf,gc)
        self.RDB2=ResidualDenseBlock(nf,gc)
        self.RDB3=ResidualDenseBlock(nf,gc)

    def forward(self,x):
        RDB1=self.RDB1(x)
        RDB2=self.RDB2(RDB1)
        RDB3=self.RDB3(RDB2)
        return RDB3



class SpatialPyramidPooling(nn.Module):
    def __init__(self,num_levels,pool_type):
        super().__init__()
        self.num_levels=num_levels
        self.pool_type=pool_type

    def forward(self,x):
        num,c,h,w=x.size()
        for level in range(1,self.num_levels+1):
            kernel_size=(math.ceil(h/level),math.ceil(w/level))
            stride=kernel_size
            pooling=(math.floor((kernel_size[0]*level-h+1)/2),math.floor((kernel_size[1]*level-w+1)/2))

            if self.pool_type=='max_pool':
                tensor=F.max_pool2d(x,kernel_size=kernel_size,stride=stride,padding=pooling).view(num,-1)
            else:
                tensor=F.avg_pool2d(x,kernel_size=kernel_size,stride=stride,padding=pooling).view(num,-1)

            if level==1:
                x_flatten=tensor.view(num,-1)
            else:
                x_flatten=tensor.cat((x_flatten,tensor.view(num,-1)),1)
        return x_flatten


#train and test
def load_model(models,scale,output_dir,epoch):
    if(epoch==0):
        load_path=os.path.join(output_dir,str(scale),'best')
    else:
        load_path=os.path.join(output_dir,str(scale),str(epoch))

    #load
    for key in models.keys():
        if type(models[key]) is not dict:
            models[key].load_state_dict(torch.load(os.path.join(load_path,key),map_location=device))
        else:
            models[key][scale].load_state_dict(torch.load(os.path.join(load_path,key),map_location=device))

def save_model(models,scale,output_dir,epoch):
    #check if output dir exists
    if(epoch==0):
        save_path=os.path.join(output_dir,str(scale),'best')
    else:
        save_path=os.path.join(output_dir,str(scale),str(epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #save
    for key in models.keys():
        if type(models[key]) is not dict:
            torch.save(models[key].state_dict(),os.path.join(save_path,key))
        else:
            torch.save(models[key][scale].state_dict(),os.path.join(save_path,key))

def save_result(PSNR,SSIM,best,output_dir,name):
    #check if output dir exists
    save_path=os.path.join(output_dir,name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    PSNR=np.array(PSNR)
    SSIM=np.array(SSIM)
    best=np.array(best)
    
    #save
    np.save(os.path.join(save_path,'PSNR.npy'),PSNR)
    np.save(os.path.join(save_path,'SSIM.npy'),SSIM)
    np.save(os.path.join(save_path,'best.npy'),best)

def load_result(output_dir,name):
    #check if output dir exists
    load_path=os.path.join(output_dir,name)
    
    PSNR=np.load(os.path.join(load_path,'PSNR.npy'))
    SSIM=np.load(os.path.join(load_path,'SSIM.npy'))
    best=np.load(os.path.join(load_path,'best.npy'))
    return PSNR,SSIM,best


#dataset
class Normalize(object):
    def __init__(self):
        super().__init__()

    def __call__(self,sample):
        return sample/255.0


class RandomSelectedRotation(object):
    def __init__(self,select_list):
        super().__init__()
        assert isinstance(select_list,list)
        self.select_list=select_list
 
    def __call__(self,sample):
        angle=random.choice(self.select_list)
        return transforms.functional.rotate(sample,angle)
        

transform_PIL=transforms.Compose([transforms.RandomCrop(96),
                                  RandomSelectedRotation([0,90,180,270]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip()])
                                   
transform_Tensor=transforms.Compose([Normalize(),
                                     transforms.ToTensor()])

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
    image=[np.array(channel).astype(np.float32) for channel in image]
    image_R=1.1644*(image[0]-16)+1.5960*(image[2]-128)
    image_G=1.1644*(image[0]-16)-0.3918*(image[1]-128)-0.8130*(image[2]-128)
    image_B=1.1644*(image[0]-16)+2.0172*(image[1]-128)
    return (image_R,image_G,image_B)

