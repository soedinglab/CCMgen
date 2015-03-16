import numpy as np


def minimize(objfun, x, maxiter, alpha0=None, alpha_decay=10):

    objfun.begin_progress()

    for i in range(maxiter):
        fx, g = objfun.evaluate(x)

        if not alpha0:
            alpha0 = 1 / np.sqrt(np.sum(g * g))

        alpha = alpha0 / (1 + i / alpha_decay)

        objfun.progress(x, g, fx, i, 1, alpha)

        x -= alpha * g


    return fx, x
