import pandas as pd
import matplotlib.pyplot as plt
from utils import get_jfk_mu

cols = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime',
        'tpep_dropoff_datetime', 'trip_distance']

dfJ = pd.read_csv('dataset/yellow_tripdata_2017-01.csv', usecols=cols)
dfF = pd.read_csv('dataset/yellow_tripdata_2017-02.csv', usecols=cols)
dfM = pd.read_csv('dataset/yellow_tripdata_2017-03.csv', usecols=cols)
# dfA = pd.read_csv('dataset/yellow_tripdata_2017-04.csv', usecols=cols)
# dfMy = pd.read_csv('dataset/yellow_tripdata_2017-05.csv', usecols=cols)

df = dfJ.append(dfF).append(dfM)  # .append(dfA).append(dfMy)
startTime, dur = get_jfk_mu(df)

plt.scatter(startTime, dur)
plt.show()
