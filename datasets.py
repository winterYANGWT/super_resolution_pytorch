import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import random
import utils
import transform

scale_factor=0
format_type=None

data_transform={}
data_transform['train']={None:None,
                         'YCbCr':transform.transform_RGB,
                         'RGB':transform.transform_Y}
data_transform['test']={None:None,
                        'YCbCr':transform.transform_RGB2Y,
                        'RGB':transform.transform_Identity}
data_transform['tensor']=transform.transform_Tensor

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file):
        super().__init__()
        self.data_frame=pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        image_path=self.data_frame.loc[index,'path']
        with Image.open(image_path).convert('RGB') as image_HR:
            if data_transform['train'][format_type]!=None:
                image_HR=data_transform['train'][format_type](image_HR)
            width,height=image_HR.size[0]//scale_factor,image_HR.size[1]//scale_factor
            image_LR=image_HR.resize((width,height),resample=Image.BICUBIC)

        if data_transform['tensor']!=None:
            image_HR=data_transform['tensor'](image_HR)
            image_LR=data_transform['tensor'](image_LR)

        return image_LR,image_HR


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file):
        super().__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.keys=self.data_frame.columns[1:]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()
        
        images_path={key:self.data_frame.loc[index,key] for key in self.keys}
        images={}
        for key in images_path.keys():
            with Image.open(images_path[key]).convert('RGB') as image:
                if data_transform['test'][format_type]!=None:
                    images[key]=data_transform['test'][format_type](image)
                
        if data_transform['tensor']!=None:
            images={key:data_transform['tensor'](images[key]) for key in images.keys()}

        return images


#train data
TrainData_T91=TrainDataset('./Datasets/TrainData_T91.csv')
TrainData_BSD500=TrainDataset('./Datasets/TrainData_BSD500.csv')
TrainData_DIV2K=TrainDataset('./Datasets/TrainData_DIV2K.csv')

#eval data
TestData_Urban100={2:TestDataset('./Datasets/TestData_Urban100_2.csv'),
                   4:TestDataset('./Datasets/TestData_Urban100_4.csv')}
TestData_BSD100={2:TestDataset('./Datasets/TestData_BSD100_2.csv'),
                 3:TestDataset('./Datasets/TestData_BSD100_3.csv'),
                 4:TestDataset('./Datasets/TestData_BSD100_4.csv')}
TestData_Set5={2:TestDataset('./Datasets/TestData_Set5_2.csv'),
               3:TestDataset('./Datasets/TestData_Set5_3.csv'),
               4:TestDataset('./Datasets/TestData_Set5_4.csv')}
TestData_Set14={2:TestDataset('./Datasets/TestData_Set14_2.csv'),
                3:TestDataset('./Datasets/TestData_Set14_3.csv'),
                4:TestDataset('./Datasets/TestData_Set14_4.csv')}

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
    scale_factor=4
    format_type='RGB'
    loader=torch.utils.data.DataLoader(dataset=TrainData_DIV2K,batch_size=8)
    for images in loader:
        print(images)

