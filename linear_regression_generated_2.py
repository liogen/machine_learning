import autograd.numpy as np
from autograd import grad
import random
import matplotlib.pyplot as plt

nbSamples = 1000
X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([3 * x[0].item(0) + 0.666 for x in X]).transpose()


def error(x_var, y_var, a):
    a = np.matrix([[a], [0.666]])
    e = x_var * a - y_var
    return (e.transpose() * e).item(0)


def gen_error(x_var, y_var):
    return lambda a: error(x_var, y_var, a)


err = gen_error(X, Y)

xs = [x * 6.0 / nbSamples for x in range(nbSamples)]
e = [err(x) for x in xs]
plt.plot(xs, e)

plt.show()

# Newton resolution

grad_err = grad(err, 0)


def newton_step(f0, df, x0):
    df0 = df(x0)
    x1 = x0 - f0 / df0
    return x1


def newton_solver(f, df, x0):
    count = 0
    f0 = f(x0)
    while True:
        x0 = newton_step(f0, df, x0)
        print("iter %d: %f" % (count, x0))
        count += 1
        f0 = f(x0)
        if f0 < 1e-6:
            break
    return x0


# newton_solver(err, grad_err, 0)
