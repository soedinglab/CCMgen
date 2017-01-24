import numpy as np


class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxiter=100, alpha0=None, alpha_decay=10):
        self.maxiter = maxiter
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay


    def minimize(self, objfun, x):

        objfun.begin_progress()

        fx, g = objfun.evaluate(x)
        objfun.progress(x, g, fx, 0, 1, 0)

        if not self.alpha0:
            self.alpha0 = 1 / np.sqrt(np.sum(g * g))

        for i in range(self.maxiter):
            alpha = self.alpha0 / (1 + i / self.alpha_decay)
            fx, g = objfun.evaluate(x)

            objfun.progress(x, g, fx, i + 1, 1, alpha)

            x -= alpha * g

        ret = {
            "code": 2,
            "message": "Reached number of iterations"
        }

        return fx, x, ret
