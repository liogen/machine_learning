import numpy as np
import math
import random
from numpy.linalg import inv

import matplotlib.pyplot as plt
from utils import splinify, get_y_reg

nbSamples = 1000

X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([math.log(x[0].item(0)) for x in X]).transpose()

Xsplines = np.matrix([splinify(0.0, 1.0, 0.05, x[0].item(0)) for x in X])
A = inv(Xsplines.transpose() * Xsplines) * Xsplines.transpose() * Y
Yreg = get_y_reg(A, Xsplines)

plt.scatter(np.asarray(X[:, 0]), np.asarray(Y))
plt.scatter(np.asarray(X[:, 0]), np.asarray(Yreg))
plt.show()
