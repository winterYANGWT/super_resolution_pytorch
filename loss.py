import torch
import torchvision
import torch.nn as nn

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BCELoss(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.bce=nn.BCELoss(reduction=reduction).to(device)

    def forward(self,output_labels,target_labels):
        return self.bce(output_labels,target_labels)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.SmoothL1Loss().to(device)

    def forward(self,output_images,target_images):
        return self.l1(output_images,target_images)

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss().to(device)

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
        height_tv=torch.pow(x[:,:,1:,:]-x[:,:,:height-1,:],2).sum()
        width_tv=torch.pow(x[:,:,:,1:]-x[:,:,:,:width-1],2).sum()
        return 2*(height_tv/count_height+width_tv/count_width)/batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg=torchvision.models.vgg.vgg19(pretrained=True)
        self.loss=nn.Sequential(*(list(vgg.features)[:35])).to(device)
        self.loss.eval()
        for param in self.loss.parameters():
            param.requires_grad=False
        self.mse=MSELoss()

    def forward(self,real,fake):
        real=real.expand(-1,3,-1,-1)
        fake=fake.expand(-1,3,-1,-1)
        return self.mse(self.loss(real),self.loss(fake))


class SRGANGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature=FeatureLoss().to(device)
        self.mse=MSELoss()
        self.tv=TVLoss()
        self.bce=BCELoss(reduction='sum')
        self.sigmoid=nn.Sigmoid()

    def forward(self,real_labels,fake_labels,real_images,fake_images):
        fake_labels_sigmoid=self.sigmoid(fake_labels)
        batch_size=real_labels.size()[0]
        ones=torch.ones(batch_size,1).to(device)
        adversarial_loss=self.bce(fake_labels_sigmoid,ones)
        perceptual_loss=self.feature(real_images,fake_images)
        content_loss=self.mse(fake_images,real_images)
        tv_loss=self.tv(fake_images)
        return content_loss+0.001*adversarial_loss+0.006*perceptual_loss+tv_loss


class SRGANDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=BCELoss()
        self.sigmoid=nn.Sigmoid()

    def forward(self,real_labels,fake_labels):
        real_labels_sigmoid=self.sigmoid(real_labels)
        fake_labels_sigmoid=self.sigmoid(fake_labels)
        batch_size=real_labels.size()[0]
        ones=torch.ones(batch_size,1).to(device)
        zeros=torch.zeros(batch_size,1).to(device)
        real_loss=self.bce(real_labels_sigmoid,ones)
        fake_loss=self.bce(fake_labels_sigmoid,zeros)
        return real_loss+fake_loss


class ESRGANGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature=FeatureLoss()
        self.l1=L1Loss()
        self.bce=BCELoss()
        self.sigmoid=nn.Sigmoid()

    def forward(self,real_labels,fake_labels,real_images,fake_images):
        batch_size=real_labels.size()[0]
        rf=self.sigmoid(real_labels-fake_labels.mean())
        fr=self.sigmoid(fake_labels-real_labels.mean())
        ones=torch.ones(batch_size,1).to(device)
        zeros=torch.zeros(batch_size,1).to(device)
        adversarial_rf=self.bce(rf,zeros)
        adversarial_fr=self.bce(fr,ones)
        adversarial_loss=adversarial_rf+adversarial_fr
        perceptual_loss=self.feature(real_images,fake_images)
        content_loss=self.l1(real_images,fake_images)
        return perceptual_loss+0.005*adversarial_loss+0.01*content_loss


class ESRGANDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=BCELoss()
        self.sigmoid=nn.Sigmoid()

    def forward(self,real_labels,fake_labels):
        batch_size=real_labels.size()[0]
        rf=self.sigmoid(real_labels-fake_labels.mean())
        fr=self.sigmoid(fake_labels-real_labels.mean())
        ones=torch.ones(batch_size,1).to(device)
        zeros=torch.zeros(batch_size,1).to(device)
        adversarial_rf=self.bce(rf,ones)
        adversarial_fr=self.bce(fr,zeros)
        return adversarial_rf+adversarial_fr

