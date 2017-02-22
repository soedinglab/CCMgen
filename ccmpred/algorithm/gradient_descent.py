import numpy as np
import ccmpred.logo
import sys
from collections import deque


class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxiter=100, alpha0=5e-3, alpha_decay=10):
        self.maxiter = maxiter
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay

        self.lastg = np.array([])
        self.g_sign = deque([])

    def __repr__(self):
        return "Gradient descent optimization (alpha0={0} alpha_decay={1} maxiter={2})".format(
            self.alpha0, self.alpha_decay, self.maxiter)


    def begin_progress(self):

        header_tokens = [('iter', 8),
                         ('|x|', 12), ('|x_single|', 12), ('|x_pair|', 12),
                         ('|g|', 12), ('|g_single|', 12), ('|g_pair|', 12),
                         ('sign_g_t10', 12), ('sign_g_t8', 12), ('sign_g_t7', 12), ('sign_g_t6', 12),
                         ('step', 12)
                         ]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, n_iter, x_single, x_pair, g, g_single, g_pair, step):

        # xnorm = np.sum(x * x)
        # gnorm = np.sum(g * g)

        xnorm_single = np.sum(x_single * x_single)
        xnorm_pair = np.sum(x_pair * x_pair)

        gnorm_single = np.sum(g_single * g_single)
        gnorm_pair = np.sum(g_pair * g_pair)

        if len(self.lastg) != 0:
            self.g_sign.append(np.sum(np.sign(self.lastg * g.copy())))
        self.lastg = g.copy()

        sign_g_t10, sign_g_t8, sign_g_t7, sign_g_t6 = [0,0,0,0]
        if len(self.g_sign) > 10:
            self.g_sign.popleft()
            sign_g_t10 = np.sum(self.g_sign)
            sign_g_t8 = np.sum(list(self.g_sign)[1:])
            sign_g_t7 = np.sum(list(self.g_sign)[2:])
            sign_g_t6 = np.sum(list(self.g_sign)[3:])


        data_tokens = [(n_iter, '8d'),
                       (xnorm_single+xnorm_pair, '12g'), (xnorm_single, '12g'), (xnorm_pair, '12g'),
                       (gnorm_single+gnorm_pair, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g'),
                       (sign_g_t10, '12g'), (sign_g_t8, '12g'), (sign_g_t7, '12g'), (sign_g_t6, '12g'),
                       (step, '12g')
                       ]


        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))


        sys.stdout.flush()


    def minimize(self, objfun, x):

        self.begin_progress()

        fx, g = objfun.evaluate(x)

        x_single, x_pair = objfun.linear_to_structured(x)
        g_single, g_pair = objfun.linear_to_structured(g)
        self.progress(0, x_single, x_pair, g, g_single, g_pair, 0)

        for i in range(self.maxiter):
            alpha = self.alpha0 / (1 + i / self.alpha_decay)

            #gradient for gap potentials is set to 0 --> gap states will stay 0
            x -= alpha * g

            fx, g = objfun.evaluate(x)

            x_single, x_pair = objfun.linear_to_structured(x)
            g_single, g_pair = objfun.linear_to_structured(g)
            self.progress(i + 1, x_single, x_pair, g, g_single, g_pair, alpha)

        ret = {
            "code": 2,
            "message": "Reached number of iterations"
        }

        return fx, x, ret
