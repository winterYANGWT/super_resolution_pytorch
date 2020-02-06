import torch
import torch.nn as nn
import torch.nn.functional as F

class model1(nn.Module):
    def __init__(self,upscale_factor):
        super(model1,self).__init__()

        #build model
        self.conv1=nn.Sequential(nn.Conv2d(1,56,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2),
                                 nn.PReLU(),
                                 nn.Conv2d(56,12,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0),
                                 nn.PReLU())
        self.conv2=nn.Sequential(nn.Conv2d(12,12,
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
        self.conv3=nn.Sequential(nn.Conv2d(12,56,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0),
                                 nn.PReLU(),
                                 nn.Conv2d(56,upscale_factor**2,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0),
                                 nn.PReLU(),
                                 nn.PixelShuffle(upscale_factor))

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        return x
