import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import copy
import model
import datasets
import meter
import utils

BATCH_SIZE=1
UPSCALE_FACTOR_LIST=[4]
EPOCH_START=0*3
EPOCH=5*3
ITER_PER_EPOCH=10
LEARNING_RATE=0.1**4
SAVE_PATH='./Model/SRGAN_DIV2K_234'
CONTINUE=(EPOCH_START!=0)
torch.set_num_threads(4)

def train(models,upscale_factor,data_loader,criterion,optimizer,meter,interpolate):
    #extract model
    generative=models['generative']
    upscale=models['upscale'][upscale_factor]
    extra=models['extra']
    disciminator=models['discriminator']

    #train
    generative.train()
    upscale.train()
    extra.train()
    discriminator.train()
    for inputs,labels in data_loader:
        if interpolate==True:
            inputs=F.interpolate(inputs,scale_factor=upscale_factor).to(device)    
        else:
            inputs=inputs.to(device)
        labels=labels.to(device)

        outputs=generative(inputs)
        outputs=upscale(outputs)
        outputs=extra(outputs)

        real_predicts=discriminator(labels)
        fake_predicts=discriminator(outputs)
        ones=torch.ones(BATCH_SIZE,1)
        zeros=torch.zeros(BATCH_SIZE,1)
        real_loss=criterion['discriminator'](real_predicts,ones)
        fake_loss=criterion['discriminator'](fake_predicts,zeros)
        d_loss=real_loss+fake_loss
        optimizer['dicriminator'].zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer['discriminator'].step()
        
        train_loss=criterion['mse'](outputs,labels);
        g_loss=criterion['generator'](fake_predicts,outputs,labels)
        meter.update(train_loss.item(),len(inputs))
        optimizer['generator'].zero_grad()
        g_loss.backward()
        optimizer['generator'].step()
    
if __name__=='__main__':
    #set device 
    cudnn.benchmark=True
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load data
    train_data=datasets.TrainData_DIV2K
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)

    #load model
    models={}
    models['generative']=model.SRResNet().to(device)
    models['upscale']={}
    for scale in UPSCALE_FACTOR_LIST:
        models['upscale'][scale]=model.SubPixelLayer(models['generative'].output_channel,scale).to(device)
    models['extra']=model.Extra_Layer().SRResNet().to(device)
    models['discriminator']=model.VGG().to(device)    
    
    if CONTINUE==True:
        for scale in UPSCALE_FACTOR_LIST:
            utils.load_model(models,scale,SAVE_PATH,EPOCH_START//len(UPSCALE_FACTOR_LIST))
    else:
        for key in models.keys():
            print(models[key])

    #set optimizer and criterion
    criterion={}
    criterion['mse']=loss.MSELoss()
    criterion['gernerator']=loss.SRGANLoss()
    criterion['discriminator']=loss.BCELoss()
    opimizer={}
    optimizer['gernerator']=optim.Adam(models['generative'].parameters(),
                                       lr=LEARNING_RATE)
    for scale in UPSCALE_FACTOR_LIST:
        optimizer['gernerator'].add_param_group({'params':models['upscale'][scale].parameters(),
                                                 'lr':LEARNING_RATE*0.1})
    optimizer['gernerator'].add_param_group({'params':models['extra'].parameters(),
                                             'lr':LEARNING_RATE*0.1})
    optimizer['discriminator']=optim.Adam(models['discriminator'].parameters(),
                                          lr=LEARNING_RATE)

    #set Meter to calculate the average of loss
    train_loss=meter.AverageMeter()

    #running
    for epoch in range(EPOCH_START,EPOCH_START+EPOCH):
        #select upscale factor
        scale=UPSCALE_FACTOR_LIST[epoch%len(UPSCALE_FACTOR_LIST)]
        datasets.scale_factor=scale

        for iteration in range(ITER_PER_EPOCH):
            train(models,scale,train_data_loader,criterion,optimizer,train_loss,False)
        print('{:0>3d}: train_loss: {:.8f}, scale: {}'.format(epoch+1,train_loss.avg,scale))
        utils.save_model(models,scale,SAVE_PATH,epoch//len(UPSCALE_FACTOR_LIST)+1)
        train_loss.reset()
