import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import model
import datasets

BATCH_SIZE=91
UPSCALE_FACTOR=4
EPOCH=2*10**5
EVAL_INTER=2*10**3
LEARNING_RATE=0.1**3
SAVE_PATH='./model/model1/'+str(UPSCALE_FACTOR)

if __name__=='__main__':
    #load data
    train_data=datasets.TrainData_T91
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=4)
    #add model
    ESPCN=model.model1(UPSCALE_FACTOR)
    print(ESPCN)
    #set optimizer and criterion
    criterion=nn.MSELoss()
    optimizer=optim.Adam([{'params':ESPCN.conv1.parameters()},
                          {'params':ESPCN.conv2.parameters()},
                          {'params':ESPCN.conv3.parameters(),'lr':LEARNING_RATE*0.1}],lr=LEARNING_RATE)
    #train
    running_loss=0.0
    iterations=0
    for epoch in range(EPOCH):
        for HR_image in train_data_loader:
            optimizer.zero_grad()
            LR_image=nn.functional.interpolate(HR_image,
                                               scale_factor=1/UPSCALE_FACTOR,
                                               mode='bicubic',
                                               align_corners=True)
            outputs=ESPCN(LR_image)
            loss=criterion(outputs,HR_image)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            iterations+=1
    #eval and save model
        if((epoch+1)%(EVAL_INTER)==0):
            print(epoch+1,': ',running_loss/iterations)
            torch.save(ESPCN.state_dict(),
                       os.path.join(SAVE_PATH,str(epoch+1)+'.mdl'))            
            running_loss=0.0
            iterations=0
