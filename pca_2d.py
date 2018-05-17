import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import random
import numpy as np

nbSamples = 1000
X0 = [random() for x in range(nbSamples)]
X1 = [3.1416 * x for x in X0]

plt.scatter(X0, X1)
plt.show()

X = np.matrix((X0, X1)).transpose()
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print(pca.components_[0])
print(pca.singular_values_)
print(pca.explained_variance_)

print(pca.components_[0][1] / pca.components_[0][0])
print(np.dot(pca.components_[0], pca.components_[1]))
print(np.linalg.norm(pca.components_[0]))
print(np.linalg.norm(pca.components_[1]))

plt.scatter(X_r[:, 0], X_r[:, 1])
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.show()
