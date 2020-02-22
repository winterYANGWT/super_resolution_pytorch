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

BATCH_SIZE=8
UPSCALE_FACTOR_LIST=[2,3,4]
EPOCH_START=0
EPOCH=70*len(UPSCALE_FACTOR_LIST)
ITER_PER_EPOCH=400
LEARNING_RATE=0.1**3
SAVE_PATH='./Model/FSRCNN_234'
LEARNING_DECAY_LIST=[0.8,0.9,1]
CONTINUE=(EPOCH_START==0)

class Best(object):
    def __init__(self):
        super().__init__()
        self.best_psnr=0
        self.best_model=None

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
    
def test(models,upscale_factor,data_loader,criterion,PSNR_meter,SSIM_meter,interpolate):
    #extract model
    generative=models['generative']
    upscale=models['upscale'][upscale_factor]
    extra=models['extra']

    #test
    generative.eval()
    upscale.eval()
    extra.eval()
    with torch.no_grad():
        for images in data_loader:
            if interpolate==True:
                inputs=F.interpolate(images['LR'],
                                     scale_factor=upscale_factor).to(device)
            else:
                inputs=images['LR'].to(device)
            labels=images['HR'].to(device)
            outputs=generative(inputs)
            outputs=upscale(outputs)
            outputs=extra(outputs)
            loss=criterion(outputs,labels)
            SSIM_meter.update(utils.calc_SSIM(outputs,labels),len(inputs))
            PSNR_meter.update(utils.calc_PSNR(loss.item()),len(inputs))

if __name__=='__main__':
    #set device 
    cudnn.benchmark=True
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load data
    train_data=datasets.TrainData_T91
    eval_data=datasets.TestData_Set5

    #add model
    models={}
    models['generative']=model.FSRCNN().to(device)
    models['upscale']={}
    for scale in UPSCALE_FACTOR_LIST:
        models['upscale'][scale]=model.Subpixel_Layer(models['generative'].output_channel,scale).to(device)
    models['extra']=model.Extra_Layer().to(device)
    for key in models.keys():
        print(models[key])

    if CONTINUE==True:
        for scale in UPSCALE_FACTOR_LIST:
            utils.load_model(models,scale,SAVE_PATH)

    best={}
    for scale in UPSCALE_FACTOR_LIST:
        best[scale]=Best()

    #set optimizer and criterion
    criterion=nn.MSELoss()
    optimizer=optim.Adam(models['generative'].parameters(),lr=LEARNING_RATE)
    for scale in UPSCALE_FACTOR_LIST:
        optimizer.add_param_group({'params':models['upscale'][scale].parameters(),'lr':LEARNING_RATE*0.1})
    optimizer.add_param_group({'params':models['extra'].parameters(),'lr':LEARNING_RATE*0.1})

    #set Meter to calculate the average of loss
    train_loss=meter.AverageMeter()
    PSNR=meter.AverageMeter()
    SSIM=meter.AverageMeter()
    decay=0

    #running
    for epoch in range(EPOCH_START,EPOCH_START+EPOCH):
        #update learning rate
        if((epoch+1)==(EPOCH*LEARNING_DECAY_LIST[decay])):
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.5
            decay=decay+1
        #select upscale factor
        scale=UPSCALE_FACTOR_LIST[epoch%len(UPSCALE_FACTOR_LIST)]
        datasets.scale_factor=scale
        #load data
        train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=8)
        eval_data_loader=torch.utils.data.DataLoader(dataset=eval_data[scale],
                                                     batch_size=1,
                                                     shuffle=False)
        for iteration in range(ITER_PER_EPOCH):
            train(models,scale,train_data_loader,criterion,optimizer,train_loss,False)
        test(models,scale,eval_data_loader,criterion,PSNR,SSIM,False)
        if best[scale].best_psnr<PSNR.avg:
            best[scale].best_psnr=PSNR.avg
            best[scale].best_model=copy.deepcopy(models)
        #report loss and PSNR and save model
        print('{:0>3d}: train_loss: {:.8f}, PSNR: {:.3f} SSIM: {:.3f}, scale: {}'.format(epoch+1,train_loss.avg,PSNR.avg,SSIM.avg,scale))
        utils.save_model(models,scale,SAVE_PATH,epoch//len(UPSCALE_FACTOR_LIST))
        train_loss.reset()
        PSNR.reset()
        SSIM.reset()

    #save best model
    for scale in UPSCALE_FACTOR_LIST:
        utils.save_model(best[scale].best_model,scale,SAVE_PATH,-1)

