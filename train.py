import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import model
import datasets
import meter
import utils

BATCH_SIZE=8
UPSCALE_FACTOR_LIST=[3]
EPOCH=800
ITER_PER_EPOCH=400
LEARNING_RATE=0.1**3
SAVE_PATH='./model/FSRCNN'
LEARNING_DECAY_LIST=[0.8,0.9,1]

def train(model_dict,upscale_factor,data_loader,criterion,optimizer,meter,interpolate):
    #extract model
    generative_net=model_dict['generative_net']
    upscale_net=model_dict['upscale_net'][upscale_factor]
    extra_net=model_dict['extra_net']

    #train
    generative_net.train()
    upscale_net.train()
    extra_net.train()
    for inputs,labels in data_loader:
        if interpolate==True:
            inputs=F.interpolate(inputs,
                                 scale_factor=UPSCALE_FACTOR).to(device)    
        else:
            inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=generative_net(inputs)
        outputs=upscale_net(outputs)
        outputs=extra_net(outputs)
        loss=criterion(outputs,labels)
        meter.update(loss.item(),len(inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def eval(model_dict,upscale_factor,data_loader,criterion,optimizer,meter,interpolate):
    #extract model
    generative_net=model_dict['generative_net']
    upscale_net=model_dict['upscale_net'][upscale_factor]
    extra_net=model_dict['extra_net']

    #eval
    generative_net.eval()
    upscale_net.eval()
    extra_net.eval()
    with torch.no_grad():
#       for inputs,labels in data_loader:
#            inputs=inputs.to(device)
#            labels=labels.to(device)
        for images in data_loader:
            if interpolate==True:
                inputs=F.interpolate(images['LR'],scale_factor=UPSCALE_FACTOR).to(device)
            else:
                inputs=images['LR'].to(device)
            labels=images['HR'].to(device)
            outputs=generative_net(inputs)
            outputs=upscale_net(outputs)
            outputs=extra_net(outputs)
            loss=criterion(outputs,labels)
            meter.update(utils.calc_PSNR(loss.item()),len(inputs))

def save_model(model_dict,scale,output_dir,epoch):
    #check if output dir exists
    save_path=os.path.join(output_dir,str(scale),str(epoch+1))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #save
    generative_net=model_dict['generative_net']
    upscale_net=model_dict['upscale_net'][scale]
    extra_net=model_dict['extra_net']
    torch.save(generative_net.state_dict(),os.path.join(save_path,'generative_net'))
    torch.save(upscale_net.state_dict(),os.path.join(save_path,'upscale_net'))
    torch.save(extra_net.state_dict(),os.path.join(save_path,'extra_net'))


if __name__=='__main__':
    #set device 
    cudnn.benchmark=True
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load data
    train_data=datasets.TrainData_T91
    eval_data=datasets.TestData_Set5

    #add model
    models={}
    models['generative_net']=model.FSRCNN().to(device)
    models['upscale_net']={}
    for scale in UPSCALE_FACTOR_LIST:
        models['upscale_net'][scale]=model.Subpixel_Layer(models['generative_net'].output_channel,scale).to(device)
    models['extra_net']=model.Extra_Layer().to(device)
    for key in models.keys():
        print(models[key])

    #set optimizer and criterion
    criterion=nn.MSELoss()
    optimizer=optim.Adam(models['generative_net'].parameters(),lr=LEARNING_RATE)
    for scale in UPSCALE_FACTOR_LIST:
        optimizer.add_param_group({'params':models['upscale_net'][scale].parameters(),'lr':LEARNING_RATE*0.1})
    optimizer.add_param_group({'params':models['extra_net'].parameters(),'lr':LEARNING_RATE*0.1})

    #set Meter to calculate the average of loss
    train_loss=meter.AverageMeter()
    PSNR=meter.AverageMeter()
    decay=0

    #running
    for epoch in range(EPOCH):
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
                                                      num_workers=8,
                                                      pin_memory=True)
        eval_data_loader=torch.utils.data.DataLoader(dataset=eval_data[scale],
                                                     batch_size=1,
                                                     shuffle=False)
        for iteration in range(ITER_PER_EPOCH):
            train(models,scale,train_data_loader,criterion,optimizer,train_loss,False)
            eval(models,scale,eval_data_loader,criterion,optimizer,PSNR,False)
        #report loss and PSNR and save model
        print('{:0>3d}: train_loss: {:.8f}, PSNR: {:.3f}'.format(epoch+1,train_loss.avg,PSNR.avg))
        save_model(models,scale,SAVE_PATH,epoch)
        train_loss.reset()
        PSNR.reset()

