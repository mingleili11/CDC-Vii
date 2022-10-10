from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import torch
from PIL import Image

#%%读取图片数据
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, filenames, mode='train',transform=None):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        aa = self.filenames[item]
        img = Image.open(self.filenames[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        _, name = os.path.split(self.filenames[item])
        s = name[:-4].split('_')
        label = int(s[-2])*1.0/int(s[-1])

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.filenames[item]

#%%读取.mat数据
class CWTDataset(Dataset):
    def __init__(self, filenames, mode='train'):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data = loadmat(self.filenames[index])
        xin = data['H'][None, ...]
        xin = torch.tensor(xin).float()

        _, name = os.path.split(self.filenames[index])
        s = name[:-4].split('_')
        target = int(s[-2]) * 1.0 / int(s[-1])
        #target = 1-int(s[-2])*1.0/int(s[-1])
        #if target>=0.7:
            #target = 0.7

        return xin, target, self.filenames[index]
