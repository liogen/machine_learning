import math
import random
from dateutil import parser
import autograd.numpy as np
from numpy.linalg import inv


nbSamples = 1000
X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([3 * x[0].item(0) + 0.666 for x in X]).transpose()


def get_y(x, a):
    return [[x[0], 1], [x[1], 1]] * a


def i_plus(xi, x):
    if x >= xi:
        return x - xi
    return 0.0


def splinify(x_min, x_max, step, x):
    a = [i_plus(x_min + i * step, x) for i in range(
        int((x_max - x_min) / step))]
    a.reverse()
    return a + [1]


def error(x_var, y_var, a):
    a = np.matrix([[a], [0.666]])
    e = x_var * a - y_var
    return (e.transpose() * e).item(0)


def gen_error(x_var, y_var):
    return lambda a: error(x_var, y_var, a)


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


def get_jfk_mu(df):
    # 236 manhattan upper east side
    jfk_mu = df[(df['PULocationID'] == 132) & (df['DOLocationID'] == 236)]

    jfk_mu['weekday'] = jfk_mu['tpep_pickup_datetime'].apply(
        lambda x: parser.parse(x).weekday())

    jfk_mu = jfk_mu[jfk_mu['weekday'] < 5]

    pu = [parser.parse(dt) for dt in jfk_mu['tpep_pickup_datetime'].values]
    do = [parser.parse(dt) for dt in jfk_mu['tpep_dropoff_datetime'].values]
    dur = [(b - a).total_seconds() / 3600.0 for a, b in zip(pu, do)]
    start_time = [dt.hour + dt.minute / 60.0 for dt in pu]

    return start_time, dur


def get_y_reg(a, x_spline):
    return np.matrix([[np.dot(x, a).item(0)] for x in x_spline])


def get_filtered(val, y, y_reg):
    return [val[i] for i in range(len(y)) if (
        (math.fabs((y[i]-y_reg[i]) / y[i]) < 0.9) and
        (y[i] > 0.2) and
        (y[i] < 2.5))]


def get_a(x, y):
    return inv(x.transpose() * x) * x.transpose() * y
