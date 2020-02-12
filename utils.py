import numpy as np
import math
from PIL import Image

def calc_PSNR(mse):
    return 10*math.log10(1/mse)

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

