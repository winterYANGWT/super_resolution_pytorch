import torch
from PIL import Image
import sys
import torchvision.transforms as transforms
import model

if __name__=='__main__':
    transfrom1=transforms.ToTensor()
    transfrom2=transforms.ToPILImage()
    input_image=transfrom1(Image.open(sys.argv[1]).convert('YCbCr'))
    input_image=input_image.view(1,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    upscale_factor=int(sys.argv[2])
    assert upscale_factor>=2 and upscale_factor<=4
    ESPCN=model.model1(upscale_factor)
    ESPCN.load_state_dict(torch.load(sys.argv[3]))
    output_image=ESPCN(input_image)
    print(output_image)
    output_image=output_image.view([output_image.shape[1],output_image.shape[2],output_image.shape[3]])
    output_image=transfrom2(output_image)
    output_image.save(sys.argv[4])
