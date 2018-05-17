import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/04cars.dat.txt')
cols = ['price', 'invoice price', 'dealer cost', 'engine', 'cylinders',
        'horsepower', 'weight', 'wheel', 'length', 'width', 'cm per gallons',
        'hm per gallons']

X_scaled = preprocessing.scale(
    df[cols].replace('*', float('nan')).dropna().as_matrix())
pe = df[df['price'] > 1000][df['engine'] < 10][['price', 'engine']]
pe_scaled = preprocessing.scale(pe)
plt.scatter(pe['price'], pe['engine'])
plt.xlabel('price')
plt.ylabel('engine')
plt.show()

plt.scatter(pe_scaled[:, 0], pe_scaled[:, 1])
plt.xlabel('price')
plt.ylabel('engine')
plt.show()

pca = PCA(n_components=2)
pca.fit(pe)
print(pca.explained_variance_)
pca.fit(pe_scaled)
print(pca.explained_variance_)

C = pe_scaled.transpose() * pe_scaled
