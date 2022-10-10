import seaborn as sns; sns.set()
import os
from torch.utils.data import DataLoader, Dataset
from network import *
from trainers import *
from dataset import *
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch
import joblib

#%%
batchSize = 32  # batch size
"""
#trainpath = os.path.join('..', 'Dataset', 'cwt_train_cropimage')
trainpath = os.path.join('..', 'Dataset', 'cwt_train')
trainfiles = utils.all_files_under(trainpath)
#testpath = os.path.join('..', 'Dataset', 'cwt_test_cropimage')
testpath = os.path.join('..', 'Dataset', 'cwt_test')
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
trainpath = os.path.join('..', 'Dataset', 'cwt_train')
trainfiles = utils.all_files_under(trainpath)
#testpath = os.path.join('..', 'Dataset', 'cwt_test_cropimage')
testpath = os.path.join('..', 'Dataset', 'cwt_test')
testfiles = utils.all_files_under(testpath)
data_transform = {
        "train": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()])}
#datasetTrain = MyDataSet(trainfiles, mode='train',transform=data_transform["train"])
datasetTrain = CWTDataset(trainfiles, mode='train')
dataloaderTrain = DataLoader(datasetTrain, batch_size=batchSize, shuffle=False)
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
        pred ,attn= model(x)


joblib.dump(attn, 'attn.pkl')

attn = joblib.load('attn.pkl')
aa = attn[13, 0, :, :]
f, ax = plt.subplots(figsize=(9, 6))
asaa = aa.max()
fsaf = aa.cpu().detach().numpy()
ax = sns.heatmap(aa.cpu().detach().numpy(), vmin=0, vmax=aa.max())
plt.show()
print('aa')