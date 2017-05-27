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
        self.decay_rate = decay_rate
        self.it_succesfull_stop_condition=-1

        self.fix_v=fix_v

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev


        metrics=['xnorm_pair', 'gnorm_pair', 'gnorm_reg_pair', 'xnorm_diff', 'max_g', 'alpha', 'sum_deviation_wij']
        if not self.fix_v:
            metrics += ['xnorm', 'xnorm_single', 'gnorm', 'gnrom_single||']

        self.progress = pr.Progress(metrics=metrics)



    def __repr__(self):
        rep_str="Gradient descent optimization (alpha0={0})\n".format(self.alpha0)


        rep_str+="convergence criteria: maxit={0} early_stopping={1} epsilon={2} prev={3}\n".format(
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


        if self.decay:
            rep_str+="decay: decay={0} decay_rate={1} decay_start={2} \n".format(
                self.decay, np.round(self.decay_rate, decimals=3), self.decay_start
            )
        else:
            rep_str+="decay: decay={0}\n".format(
              self.decay
            )

        return rep_str

    def minimize(self, objfun, x, plotfile):

        subtitle = "L={0} N={1} Neff={2}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_options(plotfile, subtitle)
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

            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)
            gnorm = np.sqrt(gnorm_single + gnorm_pair)


            g_plot_single, g_plot_pair = objfun.linear_to_structured(gplot, objfun.ncol)
            gnorm_plot_single = np.sum(g_plot_single * g_plot_single)
            gnorm_plot_pair = np.sum(g_plot_pair * g_plot_pair)
            gnorm_plot = np.sqrt(gnorm_plot_single + gnorm_plot_pair)

            g_reg_plot_single, g_reg_plot_pair = objfun.linear_to_structured(greg, objfun.ncol)
            gnorm_reg_plot_pair = np.sum(g_reg_plot_pair * g_reg_plot_pair)

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['xnorm_pair'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - np.sqrt(xnorm_pair))) / xnorm_prev
            else:
                xnorm_diff = 1

            #start decay at iteration i
            if self.decay and xnorm_diff < self.decay_start and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i

            #new step size
            if self.decay and xnorm_diff < self.decay_start:
                alpha = self.alpha0 / (1 + (i - self.it_succesfull_stop_condition) / np.float(self.decay_rate))

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            #print out progress
            self.progress.log_progress(i + 1,
                                       xnorm_pair= np.sqrt(xnorm_pair),
                                       gnorm_pair=np.sqrt(gnorm_pair),
                                       gnorm_reg_pair=np.sqrt(gnorm_reg_plot_pair),
                                       xnorm_diff=xnorm_diff, max_g=max_g, alpha=alpha,
                                       sum_deviation_wij=problems['sum_deviation_wij']
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
