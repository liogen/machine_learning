import numpy as np
import math
import random
from numpy.linalg import inv

import matplotlib.pyplot as plt

nbSamples = 1000

X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([3 * x[0].item(0) + 0.666 for x in X]).transpose()
Gnoise = np.random.normal(0.0, 0.1, len(Y))
Ynoisy = np.matrix([Y[i].item(0) + Gnoise[i] for i in range(len(Y))]).transpose()

# Find a and b
A = inv(X.transpose() * X) * X.transpose() * Ynoisy
print(A)

plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))

x = [0, 1]
y = [[x[0], 1], [x[1], 1]] * A
plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))
plt.plot(x, y, color='r')

plt.show()
