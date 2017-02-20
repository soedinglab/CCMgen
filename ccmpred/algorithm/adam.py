import numpy as np
import ccmpred.logo
import sys
from collections import deque

class Adam():
    """
    Optimize objective function using Adam

    This is an implementation of the Adam algorithm:
        Kingma, D. P., & Ba, J. L. (2015)
        Adam: a Method for Stochastic Optimization. International Conference on Learning Representations

    Adaptive Moment Estimation (Adam) computes adaptive learning rates for each parameter.
    In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop,
    Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum

    """

    def __init__(self, maxiter=100, learning_rate=1e-3, momentum_estimate1=0.9, momentum_estimate2=0.999, noise=1e-7):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.momentum_estimate1 = momentum_estimate1
        self.momentum_estimate2 = momentum_estimate2
        self.noise = noise
        self.g_hist = deque([])


    def __repr__(self):
        return "Adam stochastic optimization (learning_rate={0} momentum_estimate1={1} momentum_estimate2={2} noise={3} maxiter={4})".format(
            self.learning_rate, self.momentum_estimate1, self.momentum_estimate2, self.noise, self.maxiter)

    def begin_process(self):

        header_tokens = [('iter', 8),
                         ('|x|', 12), ('|x_single|', 12), ('|x_pair|', 12),
                         ('|g|', 12), ('|g_single|', 12), ('|g_pair|', 12),
                         ('|first moment|', 12), ('|second moment|', 12),
                         ('sum sign g', 12), ('gnorm diff', 12)
                         ]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, n_iter, x, x_single, x_pair, g, g_single, g_pair, first_moment, second_moment):

        xnorm = np.sum(x * x)
        gnorm = np.sum(g * g)

        first_moment_norm = np.sum(first_moment * first_moment)
        second_moment_norm = np.sum(second_moment * second_moment)

        xnorm_single = np.sum(x_single * x_single)
        xnorm_pair = np.sum(x_pair * x_pair)

        gnorm_single = np.sum(g_single * g_single)
        gnorm_pair = np.sum(g_pair * g_pair)

        #possible stopping criteria
        sign_g = 0
        gnorm_prev=1
        xnorm_prev=1
        if len(self.g_hist) > 7:
            self.g_hist.popleft()
            for t in range(len(self.g_hist) - 1):
                sign_g += np.sum(np.sign(self.g_hist[t + 1] * self.g_hist[t]))
            gnorm_prev = np.sum(self.g_hist[0] * self.g_hist[0])
        self.g_hist.append(g)

        g_norm_diff = (gnorm_prev - gnorm) / gnorm_prev
        #========================


        data_tokens = [(n_iter, '8d'),
                       (xnorm, '12g'), (xnorm_single, '12g'), (xnorm_pair, '12g'),
                       (gnorm, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g'),
                       (first_moment_norm, '12g'), (second_moment_norm, '12g'),
                       (sign_g, '12g'), (g_norm_diff, '12g')
                       ]


        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))


        sys.stdout.flush()

    def minimize(self, objfun, x):

        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)

        #objfun.begin_progress()
        self.begin_process()

        fx, g = objfun.evaluate(x)
        x_single, x_pair = objfun.linear_to_structured(x)
        g_single, g_pair = objfun.linear_to_structured(g)
        self.progress(0, x, x_single, x_pair, g, g_single, g_pair, first_moment, second_moment)
        #objfun.progress(x, g, fx, 0, 1, 0)

        for i in range(self.maxiter):
            #update moment vectors
            first_moment    = self.momentum_estimate1 * first_moment  +  (1-self.momentum_estimate1)  * (g)
            second_moment   = self.momentum_estimate2 * second_moment +  (1-self.momentum_estimate2)  * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment  / (1 - np.power(self.momentum_estimate1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.momentum_estimate2, i+1))

            #apply update
            x -= self.learning_rate * first_moment_corrected / np.sqrt(second_moment_corrected + self.noise)

            fx, g = objfun.evaluate(x)

            x_single, x_pair = objfun.linear_to_structured(x)
            g_single, g_pair = objfun.linear_to_structured(g)
            #objfun.progress(x, g, fx, i + 1, 1, self.learning_rate)
            self.progress(i + 1, x, x_single, x_pair, g, g_single, g_pair, first_moment_corrected, second_moment_corrected)


        ret = {
            "code": 2,
            "message": "Reached number of iterations"
        }

        return fx, x, ret
