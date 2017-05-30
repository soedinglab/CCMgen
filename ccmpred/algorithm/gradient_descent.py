import numpy as np
import ccmpred.logo
import sys
from collections import deque
import ccmpred.monitor.progress as pr
import ccmpred.model_probabilities

class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxit=100, alpha0=5e-3, decay=True,  decay_start=1e-3, decay_rate=10, fix_v=False,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False):

        self.maxit = maxit
        self.alpha0 = alpha0

        #decay settings
        self.decay=decay
        self.decay_start = decay_start
        self.decay_rate = np.float(decay_rate)
        self.it_succesfull_stop_condition=-1

        self.fix_v=fix_v

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev


        self.progress = pr.Progress()



    def __repr__(self):
        rep_str="Gradient descent optimization (alpha0={0})\n".format(self.alpha0)


        rep_str+="convergence criteria: maxit={0} early_stopping={1} epsilon={2} prev={3}\n".format(
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


        if self.decay:
            rep_str+="\tdecay: decay={0} decay_rate={1} decay_start={2} \n".format(
                self.decay, np.round(self.decay_rate, decimals=3), self.decay_start
            )
        else:
            rep_str+="\tdecay: decay={0}\n".format(
              self.decay
            )

        return rep_str

    def minimize(self, objfun, x, plotfile):

        diversity = np.sqrt(objfun.nrow)/objfun.ncol

        if self.decay_rate == 0:
            self.decay_rate = 100.0*diversity

        subtitle = "L={0} N={1} Neff={2} Diversity={3}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3), np.round(diversity,decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_options(plotfile, subtitle)



        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1
        alpha = self.alpha0
        for i in range(self.maxit):

            fx, gx, greg = objfun.evaluate(x)
            g = gx + greg


            #decompose gradients and parameters
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
            gx_single, gx_pair = objfun.linear_to_structured(gx, objfun.ncol)
            g_reg_single, g_reg_pair = objfun.linear_to_structured(greg, objfun.ncol)

            #compute norm of coupling parameters
            xnorm_pair      = np.sqrt(np.sum(x_pair * x_pair))

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||w||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm_pair)) / xnorm_prev
            else:
                xnorm_diff = 1

            #start decay at iteration i
            if self.decay and xnorm_diff < self.decay_start and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i

            #new step size
            if self.it_succesfull_stop_condition > 0:
                alpha = self.alpha0 / (1 + (i - self.it_succesfull_stop_condition) /self.decay_rate)

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            #print out progress
            log_metrics={}
            log_metrics['||w||'] = xnorm_pair
            log_metrics['||g_w||'] = np.sqrt(np.sum(gx_pair * gx_pair))
            log_metrics['||greg_w||'] = np.sqrt(np.sum(g_reg_pair * g_reg_pair))
            log_metrics['xnorm_diff'] = xnorm_diff
            log_metrics['max_g'] = np.max(np.abs(gx))
            log_metrics['alpha'] = alpha
            log_metrics['sum_w'] = problems['sum_deviation_wij']

            if not self.fix_v:
                log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
                log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
                log_metrics['||g_v||'] = np.sqrt(np.sum(gx_single * gx_single))
                log_metrics['||g||'] = np.sqrt(np.sum(gx * gx))
                log_metrics['||g_reg_v||'] = np.sqrt(np.sum(g_reg_single * g_reg_single))

            self.progress.log_progress(i + 1, **log_metrics)


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
                x_single -= alpha * g_single
            x_pair -=  alpha * g_pair

            x = objfun.structured_to_linear(x_single, x_pair)



        return fx, x, ret
