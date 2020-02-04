import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import random

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
        image=Image.open(image_path).convert('YCbCr')
        if self.transform!=None:
            image=self.transform(image)

        return image

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform=None):
        super(TestDataset,self).__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.transform=transform
        self.columns=self.data_frame.columns[1:]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()
        
        images_path=[self.data_frame.loc[index,col] for col in self.columns]
        images={col:Image.open(path).convert('YCbCr') for col,path in zip(self.columns,images_path)}
        
        if self.transform!=None:
            images={key:self.transform(images[key]) for key in images.keys()}

        return images

class RandomSelectedRotation(object):
    def __init__(self,select_list):
        super(RandomSelectedRotation,self).__init__()
        assert isinstance(select_list,list)
        self.select_list=select_list
 
    def __call__(self,sample):
        angle=random.choice(self.select_list)
        return transforms.functional.rotate(sample,angle)

transform=transforms.Compose([transforms.RandomCrop(60),
                              RandomSelectedRotation([0,90,180,270]),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor()])
transform_T91=transforms.Compose([transforms.RandomCrop(60),
                                  RandomSelectedRotation([0,90,180,270]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()])
transform_DIV2K=transforms.Compose([transforms.RandomCrop(600),
                                    RandomSelectedRotation([0,90,180,270]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])

TrainData_T91=TrainDataset('./Datasets/TrainData_T91.csv',transform=transform_T91)
TrainData_BSD500=TrainDataset('./Datasets/TrainData_BSD500.csv',transform=transform_DIV2K)
TrainData_DIV2K=TrainDataset('./Datasets/TrainData_DIV2K.csv',transform=transform)
TestData_Urban100_2=TestDataset('./Datasets/TestData_Urban100_2.csv',transform=transform)
TestData_Urban100_4=TestDataset('./Datasets/TestData_Urban100_4.csv',transform=transform)
TestData_BSD100_2=TestDataset('./Datasets/TestData_BSD100_2.csv',transform=transform)
TestData_BSD100_3=TestDataset('./Datasets/TestData_BSD100_3.csv',transform=transform)
TestData_BSD100_4=TestDataset('./Datasets/TestData_BSD100_4.csv',transform=transform)
TestData_Set5_2=TestDataset('./Datasets/TestData_Set5_2.csv',transform=transform)
TestData_Set5_3=TestDataset('./Datasets/TestData_Set5_3.csv',transform=transform)
TestData_Set5_4=TestDataset('./Datasets/TestData_Set5_4.csv',transform=transform)
TestData_Set14_2=TestDataset('./Datasets/TestData_Set14_2.csv',transform=transform)
TestData_Set14_3=TestDataset('./Datasets/TestData_Set14_3.csv',transform=transform)
TestData_Set14_4=TestDataset('./Datasets/TestData_Set14_4.csv',transform=transform)
