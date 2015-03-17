import numpy as np


def minimize(objfun, x, maxiter, alpha0=None, alpha_decay=10):

    objfun.begin_progress()

    fx, g = objfun.evaluate(x)
    objfun.progress(x, g, fx, 0, 1, 0)

    if not alpha0:
        alpha0 = 1 / np.sqrt(np.sum(g * g))

    for i in range(maxiter):
        alpha = alpha0 / (1 + i / alpha_decay)
        fx, g = objfun.evaluate(x)

        objfun.progress(x, g, fx, i + 1, 1, alpha)

        x -= alpha * g


    return fx, x
