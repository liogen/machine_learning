import numpy as np
import math
from numpy.linalg import inv
import pandas as pd

import matplotlib.pyplot as plt
from utils import splinify, get_jfk_mu, get_y_reg

nbSamples = 1000

cols = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime',
        'tpep_dropoff_datetime', 'trip_distance']

dfJ = pd.read_csv('dataset/yellow_tripdata_2017-01.csv', usecols=cols)
dfF = pd.read_csv('dataset/yellow_tripdata_2017-02.csv', usecols=cols)
dfM = pd.read_csv('dataset/yellow_tripdata_2017-03.csv', usecols=cols)
# dfA = pd.read_csv('dataset/yellow_tripdata_2017-04.csv', usecols=cols)
# dfMy = pd.read_csv('dataset/yellow_tripdata_2017-05.csv', usecols=cols)

df = dfJ.append(dfF).append(dfM)  # .append(dfA).append(dfMy)

# 236 manhattan upper east side
startTime, dur = get_jfk_mu(df)

plt.scatter(startTime, dur)
plt.show()

X = startTime
Y = dur

# Find a and b
Xm = np.matrix([splinify(np.min(X), np.max(X), 1.0, x) for x in X])
A = inv(Xm.transpose() * Xm) * Xm.transpose() * np.matrix(Y).transpose()
Yreg = get_y_reg(A, Xm)

Yfiltered = [Y[i] for i in range(len(Y)) if (
        (math.fabs((Y[i]-Yreg[i]) / Y[i]) < 0.9) and
        (Y[i] > 0.2) and
        (Y[i] < 2.5))]
Xfiltered = [X[i] for i in range(len(Y)) if (
        (math.fabs((Y[i]-Yreg[i]) / Y[i]) < 0.9) and
        (Y[i] > 0.2) and
        (Y[i] < 2.5))]

Xm = np.matrix([splinify(
    np.min(Xfiltered), np.max(X), 1.0, x) for x in Xfiltered])
A = inv(Xm.transpose() * Xm) * Xm.transpose() * np.matrix(
    Yfiltered).transpose()
Yfilteredreg = get_y_reg(A, Xm)

plt.xlabel('Heure Depart (h)')
plt.ylabel('Duree Trajet (h)')

plt.scatter(X, np.asarray(Y))
plt.scatter(Xfiltered, np.asarray(Yfilteredreg))
plt.show()
