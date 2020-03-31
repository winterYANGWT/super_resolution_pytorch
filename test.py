import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import model
import datasets
import meter
import utils
import assessment

UPSCALE_FACTOR=2
LOAD_PATH='./Model/FSRCNN_DIV2K_234'
SAVE_PATH='./Result/FSRCNN_2'
EPOCH=90

def test(models,upscale_factor,data_loader,criterion,PSNR_meter,SSIM_meter,interpolate):
    #extract model
    generative=models['generative']
    upscale=models['upscale'][upscale_factor]
    extra=models['extra']

    #test
    generative.eval()
    upscale.eval()
    extra.eval()
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
        SSIM_meter.update(assessment.calc_SSIM(outputs,labels),len(inputs))
        del generative
        del outputs
        PSNR_meter.update(assessment.calc_PSNR(loss.item()),len(inputs))

if __name__=='__main__':
    #set device
    cudnn.benchmark=True
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #load data
    test_datas=datasets.TestData[UPSCALE_FACTOR]
    #add model
    models={}
    models['generative']=model.FSRCNN().to(device)
    models['upscale']={}
    models['upscale'][UPSCALE_FACTOR]=model.Subpixel_Layer(models['generative'].output_channel,UPSCALE_FACTOR).to(device)
    models['extra']=model.Extra_Layer().to(device)
    interpolate=False

    #set criterion
    criterion=nn.MSELoss()

    #set meter
    PSNR=meter.AverageMeter()
    SSIM=meter.AverageMeter()
    
    for key in test_datas.keys():
        print(key)
        test_data=test_datas[key]
        test_data_loader=torch.utils.data.DataLoader(dataset=test_data,
                                                     batch_size=1,
                                                     shuffle=False)
        PSNR_array=[]
        SSIM_array=[]
        best=0
        for epoch in range(EPOCH):
            utils.load_model(models,UPSCALE_FACTOR,LOAD_PATH,epoch+1)
            with torch.no_grad():
                test(models,UPSCALE_FACTOR,test_data_loader,criterion,PSNR,SSIM,False)
                if best<PSNR.avg:
                    best=PSNR.avg
                PSNR_array.append(PSNR.avg)
                SSIM_array.append(SSIM.avg)
            PSNR.reset()
            SSIM.reset()
        utils.save_result(PSNR_array,SSIM_array,best,SAVE_PATH,key)

