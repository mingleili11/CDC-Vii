import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from network_for_comparision import VisionTransformer,Net,cascade_VisionTransformer
from trainers import Trainer
from dataset import *
import utils
import seaborn as sns; sns.set()
#%%
epochs = 10 # train epochs
batchSize = 32 # batch size
learning_rate = 2e-4  # learning rate
evalEpoch = 20
#%%
#trainpath = os.path.join('..', 'Dataset', 'cwt_train_cropimage')
trainpath = os.path.join('..', 'Dataset', 'cwt_train_set_use_1')
trainfiles = utils.all_files_under(trainpath)
#testpath = os.path.join('..', 'Dataset', 'cwt_test_cropimage')
testpath = os.path.join('..', 'Dataset', 'cwt_test_set_use_1')
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
#%% train
# initialize model
#model = VisionTransformer(in_c=1)
#model = Net(inchannels=1, size=128)
model = cascade_VisionTransformer(in_c=1)
# initialize trainer
trainer = Trainer(model, learning_rate=learning_rate, epochs=epochs, evalEpoch=evalEpoch,evalEnable=True)
# train the model
trainer.fit(
    trainDataLoader=dataloaderTrain,
    testDataLoader=dataloaderTest,
)
# save train logs and weights
np.save('../results/history.npy', trainer.logger)
torch.save(trainer.model, '../results/model.pt')
#%%
data = trainer.logger
trainloss = data['trainloss']
#testloss = data['testloss']
trainEpochs = np.arange(len(trainloss)) + 1
#testEpochs = np.arange(0, len(trainloss), evalEpoch) + 1

plt.figure()
plt.plot(trainEpochs, trainloss, label='train')
#plt.plot(testEpochs, testloss, label='test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('../results/loss.png')
plt.show()
