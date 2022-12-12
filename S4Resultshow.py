import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


data = pd.read_csv(os.path.join('..','results','testPredictions.csv'), index_col=0)
filenames = data.iloc[:, 0].to_list()
filenames = [f[:10] for f in filenames]
Names = np.unique(filenames)
ax = plt.axes()
colors = ['b' ,'lightcoral', 'g', 'r', 'c', 'm', 'y' ]


for i, name in enumerate(Names):
    index = [name in f for f in filenames]
    filedata = data.loc[index, :]
    filedata = filedata.sort_values('label',ascending =[False])
    error = filedata.label-filedata.prediction

    time = np.arange(filedata.shape[0]) * 1
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    p1, = ax.plot(time, filedata.prediction, lw=1, color=colors[i], marker='.', label=name)
    #p1, = ax.plot(time, filedata.prediction, lw=0.5, color=colors[i],  label=name)

    p2, = ax.plot(time, filedata.label, lw=3, color='k')
    p3 = ax.bar(time, error, width=1, color='r')
    #     ax1 = ax.twiny()
    #     ax2 = ax1.twinx()
    #     ax2.bar(time,error,width=10,color='r')
    #     ax2.set_ylim(-1,1)
    ax.legend(handles=[p1, p2, p3], labels=['prediction', 'true', 'error'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('HI')
    ax.set_title('Result on test set')
    plt.savefig('../results/testresult/' + name + '.png')
    plt.show()
    plt.close()
