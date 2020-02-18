import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import h5py
import utils

scale_factor=0

class TrainData(torch.utils.data.Dataset):
    def __init__(self,h5_file):
        super(TrainData,self).__init__()
        self.h5_file=h5_file

    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return np.expand_dims(f['lr'][idx]/255.,0),np.expand_dims(f['hr'][idx]/255.,0)

    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr'])
class EvalData(torch.utils.data.Dataset):
    def __init__(self, h5_file):
        super(EvalData, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform=None):
        super(TrainDataset,self).__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.transform=transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        image_path=self.data_frame.loc[index,'path']
        with Image.open(image_path).convert('RGB') as image_HR:
            width,height=image_HR.size[0]//scale_factor,image_HR.size[1]//scale_factor
            image_LR=image_HR.resize((width,height),resample=Image.BICUBIC)
            image_HR_Y,image_LR_Y=utils.RGB2Y(image_HR),utils.RGB2Y(image_LR)

        if self.transform!=None:
            image_HR_Y=self.transform(image_HR_Y)
            image_LR_Y=self.transform(image_LR_Y)

        return image_LR_Y,image_HR_Y

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform=None):
        super(TestDataset,self).__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.transform=transform
        self.keys=self.data_frame.columns[1:]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()
        
        images_path={key:self.data_frame.loc[index,key] for key in self.keys}
        images_Y={}
        for key in images_path.keys():
            with Image.open(images_path[key]).convert('RGB') as image:
                images_Y[key]=utils.RGB2Y(image)
                
        if self.transform!=None:
            images_Y={key:self.transform(images_Y[key]) for key in images_Y.keys()}

        return images_Y

class RandomSelectedRotation(object):
    def __init__(self,select_list):
        super(RandomSelectedRotation,self).__init__()
        assert isinstance(select_list,list)
        self.select_list=select_list
 
    def __call__(self,sample):
        angle=random.choice(self.select_list)
        return transforms.functional.rotate(sample,angle)

class Normalize(object):
    def __init__(self):
        super(Normalize,self).__init__()

    def __call__(self,sample):
        return sample/255.0

transform=transforms.Compose([transforms.RandomCrop(60),
                              RandomSelectedRotation([0,90,180,270]),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor()])
transform_T91=transforms.Compose([transforms.RandomCrop(24),
#                                  RandomSelectedRotation([0,90,180,270]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()])
transform_DIV2K=transforms.Compose([transforms.RandomCrop(24),
#                                    RandomSelectedRotation([0,90,180,270]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
transform_Test=transforms.Compose([Normalize(),
                                   transforms.ToTensor()])

#train data
TrainData_T91cut=TrainDataset('./Datasets/TrainData_T91cut.csv',transform=transform_Test)
TrainData_T91=TrainDataset('./Datasets/TrainData_T91.csv',transform=transform_T91)
TrainData_BSD500=TrainDataset('./Datasets/TrainData_BSD500.csv',transform=transform_DIV2K)
TrainData_DIV2K=TrainDataset('./Datasets/TrainData_DIV2K.csv',transform=transform)
#eval data
TestData_Urban100={2:TestDataset('./Datasets/TestData_Urban100_2.csv',transform=transform),
                   4:TestDataset('./Datasets/TestData_Urban100_4.csv',transform=transform)}
TestData_BSD100={2:TestDataset('./Datasets/TestData_BSD100_2.csv',transform=transform),
                 3:TestDataset('./Datasets/TestData_BSD100_3.csv',transform=transform),
                 4:TestDataset('./Datasets/TestData_BSD100_4.csv',transform=transform)}
TestData_Set5={2:TestDataset('./Datasets/TestData_Set5_2.csv',transform=transform_Test),
               3:TestDataset('./Datasets/TestData_Set5_3.csv',transform=transform_Test),
               4:TestDataset('./Datasets/TestData_Set5_4.csv',transform=transform_Test)}
TestData_Set14={2:TestDataset('./Datasets/TestData_Set14_2.csv',transform=transform),
                3:TestDataset('./Datasets/TestData_Set14_3.csv',transform=transform),
                4:TestDataset('./Datasets/TestData_Set14_4.csv',transform=transform)}
#test data
TestData_x2={'Urban100':TestData_Urban100[2],
             'BSD100':TestData_BSD100[2],
             'Set5':TestData_Set5[2],
             'Set14':TestData_Set14[2]}
TestData_x3={'BSD100':TestData_BSD100[3],
             'Set5':TestData_Set5[3],
             'Set14':TestData_Set14[3]}
TestData_x4={'Urban100':TestData_Urban100[4],
             'BSD100':TestData_BSD100[4],
             'Set5':TestData_Set5[4],
             'Set14':TestData_Set14[4]}
TestData={2:TestData_x2,
          3:TestData_x3,
          4:TestData_x4}

if __name__=='__main__':
    loader=torch.utils.data.DataLoader(dataset=TrainData_T91,batch_size=8)
    for images in loader:
        print(images)

