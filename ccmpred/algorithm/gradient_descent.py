import numpy as np


class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxiter=100, alpha0=1e-3, alpha_decay=1):
        self.maxiter = maxiter
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay

    def __repr__(self):
        return "Gradient descent optimization (alpha0={0} alpha_decay={1} maxiter={2})".format(
            self.alpha0, self.alpha_decay, self.maxiter)


    def minimize(self, objfun, x):

        objfun.begin_progress()

        fx, g = objfun.evaluate(x)
        objfun.progress(x, g, fx, 0, 1, 0)

        for i in range(self.maxiter):
            alpha = self.alpha0 / (1 + i / self.alpha_decay)

            #gradient for gap potentials is set to 0 --> gap states will stay 0
            x -= alpha * g

            fx, g = objfun.evaluate(x)
            objfun.progress(x, g, fx, i + 1, 1, alpha)

        ret = {
            "code": 2,
            "message": "Reached number of iterations"
        }

        return fx, x, ret
