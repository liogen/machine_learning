from autograd import grad
import matplotlib.pyplot as plt
from utils import X, Y, nbSamples, gen_error  # , newton_solver

err = gen_error(X, Y)

xs = [x * 6.0 / nbSamples for x in range(nbSamples)]
e = [err(x) for x in xs]
plt.plot(xs, e)

plt.show()

# Newton resolution

grad_err = grad(err, 0)

# newton_solver(err, grad_err, 0)
