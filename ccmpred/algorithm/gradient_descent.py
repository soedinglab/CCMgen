import numpy as np
import ccmpred.logo
import sys
from collections import deque
import ccmpred.monitor.progress as pr
import ccmpred.model_probabilities

class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxit=100, alpha0=5e-3, decay=True,  start_decay=1e-3, alpha_decay=10, fix_v=False,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False):

        self.maxit = maxit
        self.decay=decay
        self.start_decay = start_decay
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay

        self.fix_v=fix_v

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

        self.progress = pr.Progress(plotfile=None,
                            xnorm_diff=[], max_g=[], alpha=[],
                            sum_qij_uneq_1=[], neg_qijab=[], sum_wij_uneq_0=[], sum_deviation_wij=[], mean_deviation_wij=[]
                                    )

    def __repr__(self):
        return "Gradient descent optimization (alpha0={0} alpha_decay={1}) \n" \
               "convergence criteria: maxiter={2} early_stopping={3} epsilon={4} prev={5}".format(
            self.alpha0, self.alpha_decay, self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


    def minimize(self, objfun, x, plotfile):

        subtitle = "L={0} N={1} Neff={2}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            ['||x||', '||x_single||', '||x_pair||', '||g||', '||g_single||', '||g_pair||',
             'sum_qij_uneq_1', 'neg_qijab', 'sum_wij_uneq_0', 'sum_deviation_wij', 'mean_deviation_wij',
             'max_g', 'alpha'],
            subtitle
        )
        self.progress.begin_process()



        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1
        alpha = self.alpha0
        for i in range(self.maxit):

            fx, gplot, greg = objfun.evaluate(x)
            g = gplot + greg



            # ========================================================================================
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
            max_g = np.max(np.abs(g))

            xnorm_single = np.sum(x_single * x_single)
            xnorm_pair = np.sum(x_pair * x_pair)
            xnorm = np.sqrt(xnorm_single + xnorm_pair)

            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)
            gnorm = np.sqrt(gnorm_single + gnorm_pair)

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||x||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev
            else:
                xnorm_diff = 1

            #new step size
            if self.decay and xnorm_diff < self.start_decay:
                alpha = self.alpha0 / (1 + i / self.alpha_decay)

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            #print out progress
            self.progress.log_progress(i + 1,
                                       xnorm, np.sqrt(xnorm_single), np.sqrt(xnorm_pair),
                                       gnorm, np.sqrt(gnorm_single), np.sqrt(gnorm_pair),
                                       xnorm_diff=xnorm_diff, max_g=max_g, alpha=alpha,
                                       sum_qij_uneq_1=problems['sum_qij_uneq_1'],
                                       neg_qijab=problems['neg_qijab'],
                                       sum_wij_uneq_0=problems['sum_wij_uneq_0'],
                                       sum_deviation_wij=problems['sum_deviation_wij'],
                                       mean_deviation_wij=problems['mean_deviation_wij'],
                                       )
            # ========================================================================================

            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:
                    ret = {
                        "code": 0,
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                    }
                    return fx, x, ret

            # update parameters
            if not self.fix_v:
                x_single -= alpha * g_single  # x_single - alpha * step_single#
            x_pair -=  alpha * g_pair  # x_pair - alpha * step_pair#

            x = objfun.structured_to_linear(x_single, x_pair)






        return fx, x, ret
