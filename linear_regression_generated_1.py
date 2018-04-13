import numpy as np
import matplotlib.pyplot as plt
from utils import X, Y, get_y, get_a

Gnoise = np.random.normal(0.0, 0.1, len(Y))
Ynoisy = np.matrix([Y[i].item(0) + Gnoise[i] for i in range(
    len(Y))]).transpose()

# Find a and b
A = get_a(X, Ynoisy)
print(A)

plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))

x = [0, 1]
y = get_y(x, A)

plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))
plt.plot(x, y, color='r')

plt.show()
