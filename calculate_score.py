import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filename = '../results/testPredictions.csv'
df = pd.read_csv(filename)
df = df.drop(labels='Unnamed: 0',axis=1)

""""
###plt
predHI = df.prediction.to_numpy()
label = df.label.to_numpy()
actualTime=230
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(np.arange(actualTime), predHI, 'b.', label='Data')
plt.plot(np.arange(actualTime), label, 'm-')
plt.show()
"""

#'Bearing2_3',#'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7'
##Dataset_1
names=['Bearing1_3']#'Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7'
for name in names:
    resault = df['filenames'].str.contains(name)
    resault.fillna(value=False,inplace = True)
    df1=df[resault]
    label = df1.label.to_numpy()
    prediction = df1.prediction.to_numpy()
    ##MAE
    Dif_MAE=label-prediction
    MAE=np.sum(np.absolute(Dif_MAE))/len(label)
    ##RMSE
    Dif_RMSE = (label - prediction)**2
    RMSE = np.sqrt(np.sum(np.absolute(Dif_RMSE)) / len(label))

    t = dict(zip(label, prediction))
    ##SCORE
    scoreall = []
    for label_s, prediction_s in t.items():
        er=prediction_s-label_s
        if label_s >= 0.5:
            if er<=0:
                score=np.exp(-np.log(0.6)*(er)/0.1)
            else:
                score = np.exp(np.log(0.6) * (er) / 0.4)
            scoreall.append(0.6*score)
        else:
            if er <= 0:
                score = np.exp(-np.log(0.6) * (er) / 0.1)
            else:
                score = np.exp(np.log(0.6) * (er) / 0.4)
            scoreall.append(0.4*score)
    aa=np.sum(scoreall)
    bb=len(label)
    score = np.sum(scoreall)/len(label)
    print(score)
##Dataset_2
names=['Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7']
for name in names:
    resault = df['filenames'].str.contains(name)
    resault.fillna(value=False,inplace = True)
    df1=df[resault]

##Dataset_3
names=['Bearing3_3']
for name in names:
    resault = df['filenames'].str.contains(name)
    resault.fillna(value=False,inplace = True)
    df1=df[resault]