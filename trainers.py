import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from numpy import *



class Trainer():
    """general trainer
    """
    def __init__(self, model, learning_rate=2e-4, epochs=100, logCols=140, evalEpoch=5, evalEnable=True):
        self.learning_rate = learning_rate  # learning rate
        self.epochs = epochs  # epochs

        # Build Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        # adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # loss function
        self.criterion = nn.MSELoss()
        # other
        self.logCols = logCols
        self.evalEpoch = evalEpoch
        self.evalEnable = evalEnable
        self.logger = {'trainloss': [], 'testloss': []}
        self.logger1 = {'trainloss': [], 'testloss': []}
    def fit(self, trainDataLoader, testDataLoader):
        for epoch in range(self.epochs + 1):
            self.model.train()
            pbar = tqdm(trainDataLoader, ncols=self.logCols)  # initialize pbar,logCols?????????
            pbar.set_description('Epoch {:2d}'.format(epoch))#:2d??epoch???????
            # train period
            temp, NtrueTrain, NtrueReal, N = {}, 0, 0, 0
            for n_step, batch_data in enumerate(pbar):
                # get data
                xData = batch_data[0].float().to(self.device)
                yTrain = batch_data[1].float().to(self.device)
                # forward and backward
                yPred,attns = self.model(xData)
                #yPred = self.model(xData)
                loss = self.criterion(yPred, yTrain)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # metric and log data
                temp['MSE'] = loss.item()
                pbar.set_postfix(temp)
                self.logger1['trainloss'].append(loss.item())
            train_mean = mean(self.logger1['trainloss'])
            self.logger['trainloss'].append(train_mean)
            if self.evalEnable and epoch % self.evalEpoch == 0:
                # test period
                print('*' * self.logCols)
                self.evaluation(testDataLoader)
                print('*' * self.logCols)
        return self
    def evaluation(self, testDataLoader):
        self.model.eval() # set model mode
        pbar = tqdm(testDataLoader, ncols=self.logCols)
        pbar.set_description('Test')
        temp = {}
        for n_step, batch_data in enumerate(pbar):
            # get data
            xData = batch_data[0].float().to(self.device)
            yTrue = batch_data[1].float().to(self.device)
            # forward
            with torch.no_grad():
                yPred ,attns= self.model(xData)
                #yPred = self.model(xData)
            loss = self.criterion(yPred, yTrue)
            # metric and log result
            temp['MSE'] = loss.item()
            pbar.set_postfix(temp)
            self.logger1['testloss'].append(loss.item())
        test_mean = mean(self.logger1['testloss'])
        self.logger['testloss'].append(test_mean)
        return self