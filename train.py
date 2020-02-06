import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import model
import datasets
import meter
from tqdm import tqdm

BATCH_SIZE=1
UPSCALE_FACTOR=3
EPOCH=800*30
EVAL_INTER=1*30
LEARNING_RATE=0.1**3
SAVE_PATH='./model/model1/'+str(UPSCALE_FACTOR)

if __name__=='__main__':
    #load data
    train_data=datasets.TrainData_T91
    eval_data=datasets.TestData_Set5[UPSCALE_FACTOR]
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=4)
    eval_data_loader=torch.utils.data.DataLoader(dataset=eval_data,
                                                 batch_size=1,
                                                 shuffle=False)
    #add model
    model=model.model1(UPSCALE_FACTOR)
    print(model)
    #set optimizer and criterion
    criterion=nn.MSELoss()
    optimizer=optim.Adam([{'params':model.conv1.parameters()},
                          {'params':model.conv2.parameters()},
                          {'params':model.conv3.parameters(),'lr':LEARNING_RATE*0.1}],lr=LEARNING_RATE)
    
    train_loss=meter.LossMeter()
    eval_loss=meter.LossMeter()
    for epoch in range(EPOCH):
        for param_group in optimizer.param_groups:
            param_group['lr']=LEARNING_RATE*(0.1**(epoch//int(EPOCH*0.8)))
    #train
        model.train()
        with tqdm(total=(len(train_data)-len(train_data)%BATCH_SIZE),ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch,EPOCH-1))
            for HR_image in train_data_loader:
                optimizer.zero_grad()
                LR_image=nn.functional.interpolate(HR_image,
                                                   scale_factor=1/UPSCALE_FACTOR,
                                                   mode='bicubic',
                                                   align_corners=True)
                outputs=model(LR_image)
                loss=criterion(outputs,HR_image)
                loss.backward()
                optimizer.step()
                train_loss.update(loss.item(),len(LR_image))
                t.set_postfix(loss='{:.6f}'.format(train_loss.avg))
                t.update(len(HR_image))
    #eval
        model.eval()
        with torch.no_grad():
            for images_Y,_,_ in eval_data_loader:
                LR_image_Y=images_Y['LR']
                HR_image_Y=images_Y['HR']
                outputs=model(LR_image_Y)
                loss=torch.mean((outputs-HR_image_Y)**2)
                eval_loss.update(loss.item(),len(LR_image_Y))

        if((epoch+1)%(EVAL_INTER)==0):
            print(epoch+1,': train_loss: ',train_loss.avg,', eval_loss: ',eval_loss.avg,', PSNR: ',meter.calc_PSNR(eval_loss.avg))
            torch.save(model.state_dict(),
                       os.path.join(SAVE_PATH,str(epoch+1)+'.mdl'))            
            train_loss.reset()
            eval_loss.reset()
