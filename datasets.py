import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import random
import utils

scale_factor=0

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,pil_transform=None,tensor_transform=None):
        super().__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.pil_transform=pil_transform
        self.tensor_transform=tensor_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        image_path=self.data_frame.loc[index,'path']
        with Image.open(image_path).convert('RGB') as image_HR:
            if self.pil_transform!=None:
                image_HR=self.pil_transform(image_HR)
            width,height=image_HR.size[0]//scale_factor,image_HR.size[1]//scale_factor
            image_LR=image_HR.resize((width,height),resample=Image.BICUBIC)
            image_HR_Y,image_LR_Y=utils.RGB2Y(image_HR),utils.RGB2Y(image_LR)

        if self.tensor_transform!=None:
            image_HR_Y=self.tensor_transform(image_HR_Y)
            image_LR_Y=self.tensor_transform(image_LR_Y)

        return image_LR_Y,image_HR_Y


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform=None):
        super().__init__()
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


#train data
TrainData_T91=TrainDataset('./Datasets/TrainData_T91.csv',
                           pil_transform=utils.transform_PIL,
                           tensor_transform=utils.transform_Tensor)
TrainData_BSD500=TrainDataset('./Datasets/TrainData_BSD500.csv',
                              pil_transform=utils.transform_PIL,
                              tensor_transform=utils.transform_Tensor)
TrainData_DIV2K=TrainDataset('./Datasets/TrainData_DIV2K.csv',
                             pil_transform=utils.transform_PIL,
                             tensor_transform=utils.transform_Tensor)

#eval data
TestData_Urban100={2:TestDataset('./Datasets/TestData_Urban100_2.csv',
                                 transform=utils.transform_Tensor),
                   4:TestDataset('./Datasets/TestData_Urban100_4.csv',
                                 transform=utils.transform_Tensor)}
TestData_BSD100={2:TestDataset('./Datasets/TestData_BSD100_2.csv',
                               transform=utils.transform_Tensor),
                 3:TestDataset('./Datasets/TestData_BSD100_3.csv',
                               transform=utils.transform_Tensor),
                 4:TestDataset('./Datasets/TestData_BSD100_4.csv',
                               transform=utils.transform_Tensor)}
TestData_Set5={2:TestDataset('./Datasets/TestData_Set5_2.csv',
                             transform=utils.transform_Tensor),
               3:TestDataset('./Datasets/TestData_Set5_3.csv',
                             transform=utils.transform_Tensor),
               4:TestDataset('./Datasets/TestData_Set5_4.csv',
                             transform=utils.transform_Tensor)}
TestData_Set14={2:TestDataset('./Datasets/TestData_Set14_2.csv',
                              transform=utils.transform_Tensor),
                3:TestDataset('./Datasets/TestData_Set14_3.csv',
                              transform=utils.transform_Tensor),
                4:TestDataset('./Datasets/TestData_Set14_4.csv',
                              transform=utils.transform_Tensor)}

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

