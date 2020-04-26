import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

class Identity(object):
    def __init__(self):
        super().__init__()

    def __call__(self,image):
        return image


class Normalize(object):
    def __init__(self):
        super().__init__()

    def __call__(self,image):
        return image/255.0


class RandomSelectedRotation(object):
    def __init__(self,select_list):
        super().__init__()
        assert isinstance(select_list,list)
        self.select_list=select_list

    def __call__(self,image):
        angle=random.choice(self.select_list)
        return transforms.functional.rotate(image,angle)


class RGB2Y(object):
    def __init__(self):
        super().__init__()

    def __call__(self,image):
        image_np=np.array(image).astype(np.float32)
        image_Y=16+0.2568*image_np[...,0]+0.5041*image_np[...,1]+0.0978*image_np[...,2]
        return Image.fromarray(image_Y).convert('L')


class RGB2YCbCr(object):
    def __init__(self):
        super().__init__()

    def __call__(self,image):
        image_np=np.array(image).astype(np.float32)
        image_np_Y=16+0.2568*image_np[...,0]+0.5041*image_np[...,1]+0.0978*image_np[...,2]
        image_np_Cb=128-0.1482*image_np[...,0]-0.2910*image_np[...,1]+0.4392*image_np[...,2]
        image_np_Cr=128+0.4392*image_np[...,0]-0.3678*image_np[...,1]-0.0714*image_np[...,2]
        image_np=[image_np_Y,image_np_Cb,image_np_Cr]
        image=[Image.fromarray(channel).transform('L') for channel in image_np]
        return Image.merge('YCbCr',image)


class YCbCr2RGB(object):
    def __init__(self):
        super().__init__()

    def __call__(self,image):
        image_np=np.array(image).astype(np.float32)
        image_np_R=1.1644*(image_np[0]-16)+1.5960*(image_np[2]-128)
        image_np_G=1.1644*(image_np[0]-16)-0.3918*(image_np[1]-128)-0.8130*(image_np[2]-128)
        image_np_B=1.1644*(image_np[0]-16)+2.0172*(image_np[1]-128)
        image_np=[image_np_R,image_np_G,image_np_B]
        image=[Image.fromarray(channel).transform('L') for channel in image_np]
        return Image.merge('RGB',image)


transform_RGB=transforms.Compose([transforms.RandomCrop(96),
                                  RandomSelectedRotation([0,90,180,270]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  Identity()])

transform_Y=transforms.Compose([transforms.RandomCrop(96),
                                RandomSelectedRotation([0,90,180,270]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                RGB2Y()])

transform_Identity=Identity()

transform_RGB2Y=RGB2Y()

transform_Tensor=transforms.Compose([transforms.ToTensor()])

