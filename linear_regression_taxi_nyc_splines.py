import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_jfk_mu, get_y_reg, get_filtered, get_a, get_xm

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
Xm = get_xm(X, X)
A = get_a(Xm, np.matrix(Y).transpose())
Yreg = get_y_reg(A, Xm)

Yfiltered = get_filtered(Y, Y, Yreg)
Xfiltered = get_filtered(X, Y, Yreg)

Xm = get_xm(Xfiltered, X)
A = get_a(Xm, np.matrix(Yfiltered).transpose())
Yfilteredreg = get_y_reg(A, Xm)

plt.xlabel('Heure Depart (h)')
plt.ylabel('Duree Trajet (h)')

plt.scatter(X, np.asarray(Y))
plt.scatter(Xfiltered, np.asarray(Yfilteredreg))
plt.show()
