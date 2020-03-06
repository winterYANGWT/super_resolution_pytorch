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
UPSCALE_FACTOR_LIST=[2,3,4]
EPOCH_START=10*3
EPOCH=10*3
ITER_PER_EPOCH=10
LEARNING_RATE=0.1**4
SAVE_PATH='./Model/FSRCNN_DIV2K_234'
CONTINUE=(EPOCH_START!=0)
torch.set_num_threads(2)

def train(models,upscale_factor,data_loader,criterion,optimizer,meter,interpolate):
    #extract model
    generative=models['generative']
    upscale=models['upscale'][upscale_factor]
    extra=models['extra']

    #train
    generative.train()
    upscale.train()
    extra.train()
    for inputs,labels in data_loader:
        if interpolate==True:
            inputs=F.interpolate(inputs,
                                 scale_factor=upscale_factor).to(device)    
        else:
            inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=generative(inputs)
        outputs=upscale(outputs)
        outputs=extra(outputs)
        loss=criterion(outputs,labels)
        meter.update(loss.item(),len(inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
if __name__=='__main__':
    #set device 
    cudnn.benchmark=True
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load data
    train_data=datasets.TrainData_DIV2K

    #add model
    models={}
    models['generative']=model.FSRCNN().to(device)
    models['upscale']={}
    for scale in UPSCALE_FACTOR_LIST:
        models['upscale'][scale]=model.Subpixel_Layer(models['generative'].output_channel,scale).to(device)
    models['extra']=model.Extra_Layer().to(device)

    if CONTINUE==True:
        for scale in UPSCALE_FACTOR_LIST:
            utils.load_model(models,scale,SAVE_PATH,EPOCH_START//len(UPSCALE_FACTOR_LIST))
    else:
        for key in models.keys():
            print(models[key])

    #set optimizer and criterion
    criterion=nn.MSELoss()
    optimizer=optim.Adam(models['generative'].parameters(),lr=LEARNING_RATE)
    for scale in UPSCALE_FACTOR_LIST:
        optimizer.add_param_group({'params':models['upscale'][scale].parameters(),'lr':LEARNING_RATE*0.1})
    optimizer.add_param_group({'params':models['extra'].parameters(),'lr':LEARNING_RATE*0.1})

    #set Meter to calculate the average of loss
    train_loss=meter.AverageMeter()

    #running
    for epoch in range(EPOCH_START,EPOCH_START+EPOCH):
        #select upscale factor
        scale=UPSCALE_FACTOR_LIST[epoch%len(UPSCALE_FACTOR_LIST)]
        datasets.scale_factor=scale
        #load data
        train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=4)
        for iteration in range(ITER_PER_EPOCH):
            train(models,scale,train_data_loader,criterion,optimizer,train_loss,False)
        print('{:0>3d}: train_loss: {:.8f}, scale: {}'.format(epoch+1,train_loss.avg,scale))
        utils.save_model(models,scale,SAVE_PATH,epoch//len(UPSCALE_FACTOR_LIST)+1)
        train_loss.reset()
