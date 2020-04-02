import torch
import torchvision
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCELoss()

    def forward(self,output_image,target_images):
        return self.bce(output_image,target_images)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()

    def forward(self,output_images,target_images):
        return self.mse(output_images,target_images)


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        batch_size=x.size()[0]
        height=x.size()[2]
        width=x.size()[3]
        count_height=self.tensor_size(x[:,:,1:,:])
        count_width=self.tensor_size(x[:,:,:,1:])
        height_tv=torch.pow(x[:,:,1:,:]-x[:,:,:hegith-1,:],2)
        width_tv=torch.pow(x[:,:,:,1:]-x[:,:,:,:width-1],2)
        return 2*(height_tv/count_height+width_tv/count_width)/batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg=torchvision.models.vgg.vgg16(pretrained=True)
        self.loss=nn.Sequential(*(list(vgg.features)[:31])).eval()
        for param in self.loss.parameters():
            param.requires_grad=False
        self.mse=MESLoss()

    def forward(self,real,fake):
            return self.mse(self.loss(real),self.loss(fake))


class SRGANLoss(nn.Module):
    def __init__(self):
        self.feature_loss=FeatureLoss()
        self.mse=MSELoss()
        self.tv=TVLoss()

    def forward(self,fake_labels,fake_images,real_images):
        adversarial_loss=torch.sum(-torch.log2(1-fake_label))
        vgg_loss=self.feature_loss(real_images,fake_images)
        mse_loss=self.mse(fake_images,real_images)
        tv_loss=self.tv_loss(fake_images)
        return mse_loss+0.001*adversarial_loss+0.006*vgg_loss+tv_loss

