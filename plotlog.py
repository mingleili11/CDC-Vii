import numpy as np
import matplotlib.pyplot as plt
import os


data = np.load(os.path.join('..', 'results', 'history.npy'), allow_pickle=True).item()
trainloss = data['trainloss']
testloss = data['testloss']
trainEpochs = np.arange(len(trainloss))+1
testEpochs = np.arange(0, len(trainloss), 5) + 1

plt.figure()
plt.plot(trainEpochs, trainloss, label='train')
plt.plot(testEpochs, testloss, label='test')
plt.legend()
plt.xlabel('Epochs')
plt.xlabel('Loss')
plt.savefig('../results/loss.png')
