import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import model
import utils

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image-path',type=str,required=True)
    parser.add_argument('--output-path',type=str,required=True)
    parser.add_argument('--model-path',type=str,required=True)
    parser.add_argument('--scale',type=int,required=True)
    parser.add_argument('--epoch',type=int,default=0)
    args=parser.parse_args()

    #add model
    models={}
    models['generative']=model.FSRCNN()#.to(device)
    models['upscale']={}
    models['upscale'][args.scale]=model.Subpixel_Layer(models['generative'].output_channel,args.scale)#.to(device)
    models['extra']=model.Extra_Layer()#.to(device)
    interpolate=False

    #load model
    utils.load_model(models,args.scale,args.model_path,args.epoch-1)
    
    #transform
    transform=transforms.ToPILImage()
    img=Image.open(args.image_path).convert('YCbCr')
    width,height=img.size[0]*args.scale,img.size[1]*args.scale
    YCbCr=img.split()
    YCbCr=[utils.transform_Tensor(channel) for channel in YCbCr]
    YCbCr[0]=torch.unsqueeze(YCbCr[0],0)
    with torch.no_grad():
        YCbCr[0]=models['generative'](YCbCr[0])
        YCbCr[0]=models['upscale'][args.scale](YCbCr[0])
        YCbCr[0]=models['extra'](YCbCr[0])
        YCbCr[0]=torch.squeeze(YCbCr[0],0)
        YCbCr[0]=transform(YCbCr[0])
        YCbCr[1]=transform(YCbCr[1])
        YCbCr[2]=transform(YCbCr[2])
        YCbCr[1]=YCbCr[1].resize((width,height),resample=Image.BICUBIC)
        YCbCr[2]=YCbCr[2].resize((width,height),resample=Image.BICUBIC)
        YCbCr=Image.merge('YCbCr',[YCbCr[0],YCbCr[1],YCbCr[2]])
    YCbCr.save(args.output_path)
