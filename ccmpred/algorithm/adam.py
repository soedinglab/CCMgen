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

    def __init__(self, maxit=100, learning_rate=1e-3, momentum_estimate1=0.9, momentum_estimate2=0.999, noise=1e-7, epsilon=1e-5, convergence_prev=5, early_stopping=False):
        self.maxit = maxit
        self.learning_rate = learning_rate
        self.momentum_estimate1 = momentum_estimate1
        self.momentum_estimate2 = momentum_estimate2
        self.noise = noise

        self.g_hist = deque([])
        self.g_sign = deque([])
        self.x_hist = deque([])

        self.lastg = np.array([])
        self.neg_g_sign = 0

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

    def __repr__(self):
        return "Adam stochastic optimization (learning_rate={0} momentum_estimate1={1} momentum_estimate2={2} noise={3}) \n" \
               "convergence criteria: maxit={4} early_stopping={5} epsilon={6} prev={7}".format(
            self.learning_rate, self.momentum_estimate1, self.momentum_estimate2, self.noise,
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)

    def begin_process(self):

        header_tokens = [('iter', 8),
                         ('|x|', 12), ('|x_single|', 12), ('|x_pair|', 12),
                         ('|g|', 12), ('|g_single|', 12), ('|g_pair|', 12),
                         #('|first moment|', 12), ('|second moment|', 12),
                         ('xnorm_diff', 12), ('gnorm_diff', 12),
                         ('sign_g_t10', 12), ('sign_g_t8', 12)
                         ]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, n_iter, xnorm_single, xnorm_pair, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8 ):

        xnorm = xnorm_single + xnorm_pair
        gnorm = gnorm_single+gnorm_pair

        data_tokens = [(n_iter, '8d'),
                       (xnorm, '12g'), (xnorm_single, '12g'), (xnorm_pair, '12g'),
                       (gnorm, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g'),
                       (xnorm_diff, '12g'), (gnorm_diff, '12g'),
                       (sign_g_t10, '12g'), (sign_g_t8, '12g')
                       ]


        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))


        sys.stdout.flush()

    def minimize(self, objfun, x):

        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)

        self.begin_process()

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }
        fx = -1


        for i in range(self.maxit):

            fx, g = objfun.evaluate(x)

            #update moment vectors
            first_moment    = self.momentum_estimate1 * first_moment + (1-self.momentum_estimate1) * (g)
            second_moment   = self.momentum_estimate2 * second_moment + (1-self.momentum_estimate2) * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(self.momentum_estimate1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.momentum_estimate2, i+1))

            # ========================================================================================
            x_single, x_pair = objfun.linear_to_structured(x)
            g_single, g_pair = objfun.linear_to_structured(g)

            xnorm_single = np.sum(x_single * x_single)
            xnorm_pair = np.sum(x_pair * x_pair)
            xnorm = xnorm_single + xnorm_pair

            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)
            gnorm = gnorm_single + gnorm_pair

            # possible stopping criteria
            if len(self.lastg) != 0:
                self.g_sign.append(np.mean(np.sign(self.lastg * g.copy())))
            self.lastg = g.copy()

            self.x_hist.append(xnorm)
            self.g_hist.append(gnorm)

            xnorm_diff = 1
            gnorm_diff = 1
            sign_g_t10, sign_g_t8, sign_g_t7, sign_g_t6 = [0, 0, 0, 0]
            if len(self.g_sign) > 10:
                self.g_sign.popleft()
                sign_g_t10 = np.sum(self.g_sign) / 10.0
                sign_g_t8 = np.sum(list(self.g_sign)[2:]) / 8.0

                self.x_hist.popleft()
                xnorm_prev = self.x_hist[-self.convergence_prev-1]
                xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev

                self.g_hist.popleft()
                gnorm_prev = self.g_hist[-self.convergence_prev-1]
                gnorm_diff = np.abs((gnorm_prev - gnorm)) / gnorm_prev


            if sign_g_t8 < 0:
                self.neg_g_sign += 1
            else:
                self.neg_g_sign = 0

            # ====================================================================================


            #print out progress
            self.progress(i + 1, xnorm_single, xnorm_pair, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8)


            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:
                    ret = {
                        "code": 0,
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                    }
                    return fx, x, ret

                # if self.neg_g_sign > 10:
                #     ret = {
                #         "code": 0,
                #         "message": "Stopping condition (change of gradient direction) successfull."
                #     }
                #
                #     return fx, x, ret


            #update parameters
            x -= self.learning_rate * first_moment_corrected / np.sqrt(second_moment_corrected + self.noise)


        return fx, x, ret
