import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from dataset import *
import utils
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")
plt.close('all')

data = pd.read_csv(os.path.join('..','results','testPredictions.csv'), index_col=0)
filenames = data.iloc[:, 0].to_list()
filenames = [f[:10] for f in filenames]
Names = np.unique(filenames)


for name in Names:
    #name = 'Bearing1_4'
    # 真实Fault时间
    #total = utils.all_files_under_deletetemp(os.path.join('/home/ubuntu/vision_transformer/forpaper/Dataset/Full_Test_Set', name))
    #total = open('..','Dataset','record.txt', "r")
    total = np.loadtxt('../Dataset/record.txt',delimiter=',')
    #actualTime = len(total)*10

    actualTime = total[0] * 10
    label= np.arange(actualTime)/actualTime
    # 加载vit模型预测的HI
    index = [name in f for f in filenames]
    filedata = data.loc[index,:]
    filedata = filedata.sort_values('label')
    time = np.arange(filedata.shape[0])*10
    predHI = filedata.prediction.values
    curtime = np.max(time)
    
    # 基于GPR预测未来HI
    kernel = DotProduct() + Exponentiation(RationalQuadratic(), exponent=1.1)
    gpr = GaussianProcessRegressor(alpha=1e-2, kernel=kernel)
    gpr.fit(time[:, None], predHI)
    
    timeAll = np.arange(actualTime+100)
    predHIAll, predstd = gpr.predict(timeAll[:, None], return_std=True)
    conf = np.vstack([predHIAll-2*predstd, predHIAll+2*predstd]).T
    
    # 预测Fault时间
    index = np.argmin(np.abs(predHIAll-1))
    predTime = timeAll[index]
    # 真实Fault时刻，预测的HI
    faultHI = predHIAll[timeAll==actualTime]
    
    # RUL 误差
    actualRUL = actualTime - curtime
    predictedRUL = predTime - curtime
    print(f'{name}, current time: {curtime}s, actural RUL:{actualRUL}s, predicted RUL:{predictedRUL}s')
    
    #%%
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(time, predHI, 'b.', label='Data')
    plt.plot(np.arange(actualTime), label, 'm-')
    plt.plot(timeAll[timeAll<=curtime], predHIAll[timeAll<=curtime], 'r-', label='Estimation')
    plt.plot(timeAll[timeAll>curtime], predHIAll[timeAll>curtime], 'g-', label='Prediction')
    plt.plot(timeAll, conf, 'k--', label='95% confidence interval')
    plt.plot(timeAll, timeAll*0 + faultHI, 'k-', linewidth=2, label='Failure Treshold')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('HI')
    plt.title(name)
    savepath = os.path.join( 'results', 'gengxin')
    plt.savefig('../results/testresult/'+name + '.png')
    plt.show()
    plt.close()

    """
    ######预测存储
        batchSize=32
        testpath = os.path.join('..', 'Dataset', 'cwt_test_set_use_1')
        testfiles = utils.all_files_under(testpath)
        datasetTest = CWTDataset(testfiles, mode='test')
        dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=False)
        filenames, realValues = [], []
        for i, (x, y, names) in enumerate(tqdm(dataloaderTest)):
            names = [name[name.index('Bear'):-4] for name in names]
            filenames.append(names)
            realValues.append(y.numpy())

        filenames = np.hstack(filenames)
        realValues = np.hstack(realValues)

        pred_csv = pd.DataFrame({'filenames': filenames, 'label': realValues, 'prediction': predHIAll})
        pred_csv.to_csv('../results/predHIAll.csv')
    ######
    """

