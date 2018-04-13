import numpy as np
import math
from numpy.linalg import inv
import pandas as pd
from dateutil import parser

import matplotlib.pyplot as plt

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
JFK_MU = df[(df['PULocationID'] == 132) & (df['DOLocationID'] == 236)]

JFK_MU['weekday'] = JFK_MU['tpep_pickup_datetime'].apply(
    lambda x: parser.parse(x).weekday())

JFK_MU = JFK_MU[JFK_MU['weekday'] < 5]

pu = [parser.parse(dt) for dt in JFK_MU['tpep_pickup_datetime'].values]
do = [parser.parse(dt) for dt in JFK_MU['tpep_dropoff_datetime'].values]
dur = [(b - a).total_seconds() / 3600.0 for a, b in zip(pu, do)]
startTime = [dt.hour + dt.minute / 60.0 for dt in pu]

plt.scatter(startTime, dur)
plt.show()

X = startTime
Y = dur


def i_plus(xi, x):
    if x >= xi:
        return x - xi
    return 0.0


def splinify(x_min, x_max, step, x):
    a = [i_plus(x_min + i * step, x) for i in range(int(
        (x_max - x_min) / step))]
    a.reverse()
    return a + [1]


# Find a and b
Xm = np.matrix([splinify(np.min(X), np.max(X), 1.0, x) for x in X])
A = inv(Xm.transpose() * Xm) * Xm.transpose() * np.matrix(Y).transpose()
Yreg = np.matrix([[np.dot(x, A).item(0)] for x in Xm])

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
Yfilteredreg = np.matrix([[np.dot(x, A).item(0)] for x in Xm])

plt.xlabel('Heure Depart (h)')
plt.ylabel('Duree Trajet (h)')

plt.scatter(X, np.asarray(Y))
plt.scatter(Xfiltered, np.asarray(Yfilteredreg))
plt.show()
