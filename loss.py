import torch
import torchvision
import torch.nn as nn

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


class SRGANLoss(nn.Module):
    def __init__(self):
        vgg=torchvision.models.vgg.vgg16(pretrained=True)
        loss_network=nn.Sequential(*(list(vgg.features)[:31])).eval()
        for param in loss_network.parameters():
            param.requires_grad=False
        self.loss_network=loss_network
        self.mse=nn.MSELoss()
        self.tv=TVLoss()

    def forward(self,out_labels,out_images,target_images):
        adversarial_loss=torch.mean(1-out_labels)
        perception_loss=self.mse(self.loss_network(out_images),self.loss_network(target_images))
        image_loss=self.mse(out_images,target_images)
        tv_loss=self.tv_loss(out_images)
        return image_loss+0.001*adversarial_loss+0.006*perception_loss+tv_loss


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

