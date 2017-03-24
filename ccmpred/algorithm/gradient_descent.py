import numpy as np
import ccmpred.logo
import sys
from collections import deque
import ccmpred.monitor.progress as pr


class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxit=100, alpha0=5e-3, alpha_decay=10, epsilon=1e-5, convergence_prev=5, early_stopping=False):

        self.maxit = maxit
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

        self.progress = pr.Progress(plotfile=None,
                            xnorm_diff=[], max_g=[], gnorm_diff=[], alpha=[])

    def __repr__(self):
        return "Gradient descent optimization (alpha0={0} alpha_decay={1}) \n" \
               "convergence criteria: maxiter={2} early_stopping={3} epsilon={4} prev={5}".format(
            self.alpha0, self.alpha_decay, self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


    def minimize(self, objfun, x, plotfile):

        subtitle = "L={0} N={1} Neff={2}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            ['||x||', '||x_single||', '||x_pair||', '||g||', '||g_single||', '||g_pair||','max_g', 'alpha'],
            subtitle
        )
        self.progress.begin_process()



        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1

        for i in range(self.maxit):

            fx, g = objfun.evaluate(x)

            #new step size
            alpha = self.alpha0 / (1 + i / self.alpha_decay)

            # ========================================================================================
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
            max_g = np.max(np.abs(g))

            xnorm_single = np.sum(x_single * x_single)
            xnorm_pair = np.sum(x_pair * x_pair)
            xnorm = xnorm_single + xnorm_pair

            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)
            gnorm = gnorm_single + gnorm_pair

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||x||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev
                gnorm_prev = self.progress.optimization_log['||g||'][-self.convergence_prev]
                gnorm_diff = np.abs((gnorm_prev - gnorm)) / gnorm_prev
            else:
                xnorm_diff = 1
                gnorm_diff = 1


            #print out progress
            self.progress.log_progress(i + 1,
                                       xnorm_single, xnorm_pair,
                                       gnorm_single, gnorm_pair,
                                       xnorm_diff=xnorm_diff, max_g=max_g, gnorm_diff=gnorm_diff, alpha=alpha)
            # ========================================================================================

            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:
                    ret = {
                        "code": 0,
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                    }
                    return fx, x, ret


            #update parameters
            x -= alpha * g




        return fx, x, ret
