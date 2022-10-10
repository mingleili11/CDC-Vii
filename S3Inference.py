import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from network import *
from trainers import *
from dataset import *
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
#%%
batchSize = 32  # batch size
"""
trainpath = os.path.join('..', 'Dataset', 'cwt_train_cropimage')
trainfiles = utils.all_files_under(trainpath)
testpath = os.path.join('..', 'Dataset', 'cwt_test_cropimage')
testfiles = utils.all_files_under(testpath)
data_transform = {
        "train": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()])}
datasetTrain = MyDataSet(trainfiles, mode='train',transform=data_transform["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=batchSize, shuffle=True)
datasetTest = MyDataSet(testfiles, mode='test',transform=data_transform["test"])
dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=False)
"""
#%%
#trainpath = os.path.join('..', 'Dataset', 'cwt_train_cropimage')
trainpath = os.path.join('..', 'Dataset', 'cwt_train_set_use_1')
trainfiles = utils.all_files_under(trainpath)
#testpath = os.path.join('..', 'Dataset', 'cwt_test_cropimage')
testpath = os.path.join('..', 'Dataset', 'cwt_test_set_use_1')
#testpath = os.path.join('..', 'Dataset', 'test_set')
testfiles = utils.all_files_under(testpath)
"""
data_transform = {
        "train": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()])}
"""
#datasetTrain = MyDataSet(trainfiles, mode='train',transform=data_transform["train"])
datasetTrain = CWTDataset(trainfiles, mode='train')
dataloaderTrain = DataLoader(datasetTrain, batch_size=batchSize, shuffle=True)
#datasetTest = MyDataSet(testfiles, mode='test',transform=data_transform["test"])
datasetTest = CWTDataset(testfiles, mode='test')
dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=False)
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('../results/model.pt')
model.to(device)
model.eval()

#%% 预测训练集的所有结果
filenames, realValues, predValues = [], [], []
for i, (x, y, names) in enumerate(tqdm(dataloaderTrain)):
    x = x.to(device)
    with torch.no_grad():
        pred,attn = model(x)
        #pred = model(x)

    names = [name[name.index('Bear'):-4] for name in names]
    filenames.append(names)
    realValues.append(y.numpy())
    predValues.append(pred.detach().cpu().numpy())


filenames = np.hstack(filenames)
realValues = np.hstack(realValues)
predValues = np.hstack(predValues)

trainDF = pd.DataFrame({'filenames': filenames, 'label':realValues, 'prediction':predValues})
trainDF = trainDF.sort_values('filenames')
trainDF.to_csv('../results/trainPredictions.csv')


#%% 预测测试集的所有结果
filenames, realValues, predValues = [], [], []
for i, (x, y, names) in enumerate(tqdm(dataloaderTest)):
    x = x.to(device)
    with torch.no_grad():
        pred,attn = model(x)
        #pred = model(x)

    names = [name[name.index('Bear'):-4] for name in names]
    filenames.append(names)
    realValues.append(y.numpy())
    predValues.append(pred.detach().cpu().numpy())

filenames = np.hstack(filenames)
realValues = np.hstack(realValues)
predValues = np.hstack(predValues)

testDF = pd.DataFrame({'filenames': filenames, 'label': realValues, 'prediction': predValues})
testDF = testDF.sort_values('filenames')
testDF.to_csv('../results/testPredictions.csv')








