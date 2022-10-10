import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
#plt.close('all')
data = pd.read_csv(os.path.join('..','results','trainPredictions.csv'), index_col=0)
#data = pd.read_csv(os.path.join('..','results','testPredictions.csv'), index_col=0)
filenames = data.iloc[:, 0].to_list()
filenames = [f[:10] for f in filenames]
Names = np.unique(filenames)
ax = plt.axes()
colors = ['b' ,'lightcoral', 'g', 'r', 'c', 'm', 'y' ]
for i, name in enumerate(Names):
    index = [name in f for f in filenames]
    filedata = data.loc[index,:]
    #filedata = filedata.sort_values('label') predict future fault trend use this
    time = np.arange(filedata.shape[0])*10
    plt.figure(figsize=(10, 10), dpi=100)
    #plt.plot(time, filedata.prediction, color="lightcoral", linewidth=3.0, linestyle="-", label=name)
    #plt.plot(time, filedata.prediction, color=colors[i] + '.', label=name)
    #plt.plot(time, filedata.prediction, color=colors[i] ,marker='.', label=name)
    plt.scatter(time, filedata.prediction, linewidths=1, color=colors[i], marker='.',label=name)
    #p1 = ax.scatter(time, filedata.prediction, marker='.', color=colors[i], s=8)
    plt.plot(time, filedata.label, 'k-')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('HI')
    plt.title('Result on train set')
    plt.savefig('../results/trainresult/' + name + '.png')
    plt.show()
    plt.close()
    #plt.savefig('../results/train.png')

