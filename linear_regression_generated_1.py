import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt
from utils import X, Y

Gnoise = np.random.normal(0.0, 0.1, len(Y))
Ynoisy = np.matrix([Y[i].item(0) + Gnoise[i] for i in range(
    len(Y))]).transpose()

# Find a and b
A = inv(X.transpose() * X) * X.transpose() * Ynoisy
print(A)

plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))

x = [0, 1]
y = [[x[0], 1], [x[1], 1]] * A
plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))
plt.plot(x, y, color='r')

plt.show()
