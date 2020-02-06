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
        image_Y=Image.open(image_path).convert('YCbCr').split()[0]
        if self.transform!=None:
            image_Y=self.transform(image_Y)

        return image_Y

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
        images_Y={col:Image.open(path).convert('YCbCr').split()[0] for col,path in zip(self.columns,images_path)}
        images_Cb={col:Image.open(path).convert('YCbCr').split()[1] for col,path in zip(self.columns,images_path)}
        images_Cr={col:Image.open(path).convert('YCbCr').split()[2] for col,path in zip(self.columns,images_path)}
        
        if self.transform!=None:
            images_Y={key:self.transform(images_Y[key]) for key in images_Y.keys()}
            images_Cb={key:self.transform(images_Cb[key]) for key in images_Cb.keys()}
            images_Cr={key:self.transform(images_Cr[key]) for key in images_Cr.keys()}

        return images_Y,images_Cb,images_Cr

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
#                                  RandomSelectedRotation([0,90,180,270]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()])
transform_DIV2K=transforms.Compose([transforms.RandomCrop(600),
                                    RandomSelectedRotation([0,90,180,270]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
transform_Test=transforms.Compose([transforms.ToTensor()])

TrainData_T91=TrainDataset('./Datasets/TrainData_T91.csv',transform=transform_T91)
TrainData_BSD500=TrainDataset('./Datasets/TrainData_BSD500.csv',transform=transform_DIV2K)
TrainData_DIV2K=TrainDataset('./Datasets/TrainData_DIV2K.csv',transform=transform)
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

if __name__=='__main__':
    loader=torch.utils.data.DataLoader(dataset=TrainData_T91,batch_size=8)
    for images in loader:
        print(images)
