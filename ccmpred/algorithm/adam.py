import numpy as np
import ccmpred.logo
import ccmpred.monitor.progress as pr
import sys
from collections import deque
import ccmpred.model_probabilities
import ccmpred.sanity_check

class Adam():
    """
    Optimize objective function using Adam

    This is an implementation of the Adam algorithm:
        Kingma, D. P., & Ba, J. L. (2015)
        Adam: a Method for Stochastic Optimization. International Conference on Learning Representations

    Adaptive Moment Estimation (Adam) computes adaptive learning rates for each parameter.
    In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop,
    Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum

    m = mom1*m + (1-mom1)*dx
    v = mom2*v + (1-mom2)*(dx**2)
    x += - learning_rate * m / (np.sqrt(v) + eps)


    If setting mom1=0, then Adam becomes RMSProp as described by Hinton here:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    v = mom2*v + (1-mom2)*dx**2
    x += - learning_rate * dx / (np.sqrt(v) + eps)

    """

    def __init__(self, maxit=100, alpha0=1e-3, decay_rate=1e1, beta1=0.9, beta2=0.999, beta3=0.9, noise=1e-8,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False, decay_type="step",
                 decay=False, decay_start=1e-4, fix_v=False, group_alpha=False, qij_condition=False):

        self.alpha0 = alpha0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.noise = noise

        self.decay=decay
        self.it_succesfull_stop_condition=-1
        self.decay_rate = decay_rate
        self.decay_start = decay_start
        self.decay_type  = decay_type

        self.refinement = False

        self.fix_v = fix_v
        self.group_alpha = group_alpha

        self.maxit = maxit
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev
        self.qij_condition = qij_condition


        metrics=['xnorm_pair', 'gnorm_pair', 'norm_g_reg_pair', 'xnorm_diff', 'max_g', 'alpha',
                 'sum_qij_uneq_1', 'neg_qijab', 'sum_wij_uneq_0', 'sum_deviation_wij']
        if not self.fix_v:
            metrics += ['xnorm', 'xnorm_single', 'gnorm', 'gnrom_single||']

        self.progress = pr.Progress(metrics=metrics)

    def __repr__(self):

        rep_str="Adam (beta1={0} beta2={1} beta3={2} alpha0={3} noise={4} fix_v={5}) \n ".format(
            self.beta1, self.beta2, self.beta3, np.round(self.alpha0, decimals=3), self.noise, self.fix_v
        )

        if self.decay:
            rep_str+="decay: decay={0} decay_rate={1} decay_start={2} decay_type={3}\n".format(
                self.decay, np.round(self.decay_rate, decimals=3), self.decay_start, self.decay_type
            )
        else:
            rep_str+="decay: decay={0}\n".format(
              self.decay
            )

        if self.early_stopping:
            rep_str+="convergence criteria: maxit={0} early_stopping={1} epsilon={2} prev={3}\n".format(
                self.maxit, self.early_stopping, self.epsilon, self.convergence_prev
            )
        else:
            rep_str+="convergence criteria: maxit={0} early_stopping={1}\n".format(
                self.maxit, self.early_stopping
            )

        return rep_str




    def minimize(self, objfun, x, plotfile):

        diversity = np.sqrt(objfun.nrow)/objfun.ncol
        L = objfun.ncol

        #scale learning rate with diversity of alignment
        #self.alpha0 = diversity/1 # 1.0 / np.sqrt(objfun.neff)
        alpha=self.alpha0
        #self.decay_rate = 10.0*diversity #L/2.0 #np.sqrt(objfun.neff)


        subtitle = "L={0} N={1} Neff={2} Diversity={3}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3), np.round(diversity,decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_options(plotfile, subtitle)
        self.progress.begin_process()


        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)
        x_moment = np.zeros(objfun.nvar)

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1
        for i in range(self.maxit):

            fx, gplot, greg = objfun.evaluate(x)
            g = gplot + greg

            #update moment vectors
            first_moment    = self.beta1 * first_moment + (1-self.beta1) * (g)
            second_moment   = self.beta2 * second_moment + (1-self.beta2) * (g*g)
            x_moment        = self.beta3 * x_moment + (1-self.beta3) * (x)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(self.beta1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.beta2, i+1))
            x_moment_corrected = x_moment / (1 - np.power(self.beta3, i+1))

            first_moment_corrected_single, first_moment_corrected_pair = objfun.linear_to_structured(first_moment_corrected, objfun.ncol)
            second_moment_corrected_single, second_moment_corrected_pair = objfun.linear_to_structured(second_moment_corrected, objfun.ncol)
            x_moment_corrected_single, x_moment_corrected_pair = objfun.linear_to_structured(x_moment_corrected, objfun.ncol)


            #use same scaling of gradients for each group!, e.g v_i and w_ij
            if self.group_alpha:
                step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single.max(1))[:, np.newaxis] + self.noise)
                step_pair   = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair.max(3).max(2))[:, :, np.newaxis, np.newaxis] + self.noise)
            else:
                step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single) + self.noise)
                step_pair   = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair) + self.noise)


            #compute metrics for logging and stop criteria
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
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

            max_g = np.max(np.abs(g))

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['xnorm_pair'][-self.convergence_prev-1]
                xnorm_diff = np.abs((xnorm_prev - np.sqrt(xnorm_pair))) / xnorm_prev


                wij_deviation_prev = self.progress.optimization_log['sum_deviation_wij'][-self.convergence_prev-1]
                wij_deviation_diff = np.abs(wij_deviation_prev - problems['sum_deviation_wij']) / wij_deviation_prev

            else:
                xnorm_diff = np.nan
                wij_deviation_diff = np.nan


            #start decay at iteration i
            if self.decay and xnorm_diff < self.decay_start and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i



            #update learning rate
            if self.decay and self.it_succesfull_stop_condition > -1:
                    if self.decay_type == "power":
                        alpha *= self.decay_rate
                        #self.beta1 *= 0.9999
                        #beta2 *= 0.9999
                        #beta3 *= 0.9999
                    elif self.decay_type == "lin":
                        alpha = self.alpha0 / (1 + (i - self.it_succesfull_stop_condition) / self.decay_rate)
                    elif self.decay_type == "step":
                        alpha *= self.decay_rate
                        self.decay_start *= 5e-1
                        self.it_succesfull_stop_condition = -1
                        # self.beta1 *= self.alpha_decay
                        # self.beta2 *= self.alpha_decay
                    elif self.decay_type == "sqrt":
                        alpha = self.alpha0  / (1 + (np.sqrt(1 + i - self.it_succesfull_stop_condition)) / self.decay_rate)
                        #beta1 = self.beta1  * np.power(0.99, (i-self.it_succesfull_stop_condition))
                        #beta2 = self.beta2  * np.power(0.99, (i-self.it_succesfull_stop_condition))


            #print out (and possiblly plot) progress
            self.progress.log_progress(i + 1,
                                       xnorm_pair=np.sqrt(xnorm_pair),
                                       gnorm_pair=np.sqrt(gnorm_plot_pair),
                                       xnorm_diff=xnorm_diff,
                                       max_g=max_g, alpha=alpha,
                                       sum_qij_uneq_1=problems['sum_qij_uneq_1'],
                                       neg_qijab=problems['neg_qijab'],
                                       sum_wij_uneq_0=problems['sum_wij_uneq_0'],
                                       sum_deviation_wij=problems['sum_deviation_wij'],
                                       #sqrtv=np.sqrt(np.sum(second_moment_corrected_pair)),
                                       #m=np.sqrt(np.sum(first_moment_corrected_pair*first_moment_corrected_pair)),
                                       #step=np.sqrt(np.sum(step_pair*step_pair)),
                                       norm_g_reg_pair=np.sqrt(gnorm_reg_plot_pair)
                                       )


            #stop condition
            if self.early_stopping:

                if xnorm_diff < self.epsilon:


                    if self.qij_condition:

                        if (problems['sum_qij_uneq_1'] == 0) and (problems['neg_qijab'] == 0):

                            if self.refinement and objfun.gibbs_steps == 1:
                                print("Start Refinement...")
                                self.epsilon *= 1e-3
                                objfun.gibbs_steps = 10

                            else:
                                ret = {
                                    "code": 1,
                                    "message": "Stopping condition (xnorm diff < {0} and #qij violations = 0 ) successfull.".format(
                                        self.epsilon)
                                 }
                                return fx, x, ret
                    else:
                        ret = {
                            "code": 1,
                            "message": "Stopping condition (xnorm diff < {0}) successfull.".format(
                               self.epsilon)
                        }
                        return fx, x, ret


            #update parameters
            if not self.fix_v:
                x_single =x_moment_corrected_single - alpha * step_single#x_single - alpha * step_single#
            x_pair = x_moment_corrected_pair - alpha * step_pair#x_pair - alpha * step_pair#

            x=objfun.structured_to_linear(x_single, x_pair)

        return fx, x, ret

