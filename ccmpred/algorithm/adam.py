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

    def __init__(self, maxit=100, alpha0=1e-3, alpha_decay=1e1, beta1=0.9, beta2=0.999, beta3=0.9, noise=1e-8,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False, decay_type="step",
                 decay=False, start_decay=1e-4, fix_v=False, group_alpha=False, qij_condition=False):
        self.maxit = maxit
        self.alpha0 = alpha0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.noise = noise
        self.decay=decay
        self.alpha_decay = alpha_decay
        self.start_decay = start_decay
        self.decay_type  = decay_type

        self.fix_v = fix_v
        self.group_alpha = group_alpha

        self.early_stopping = early_stopping
        self.it_succesfull_stop_condition=-1
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev
        self.qij_condition = qij_condition

        self.progress = pr.Progress(plotfile=None,
                                    xnorm_diff=[], max_g=[], alpha=[],
                                    sum_qij_uneq_1=[], neg_qijab=[], sum_wij_uneq_0=[])


    def __repr__(self):

        rep_str="Adam stochastic optimization ( beta1={0} beta2={1} beta3={2} alpha0={3} noise={4} fix_v={5}) \n ".format(
            self.beta1, self.beta2, self.beta3, self.alpha0, self.noise, self.fix_v
        )

        if self.decay:
            rep_str+="decay: decay={0} alpha_decay={1} start_decay={2} decay_type={3}\n".format(
                self.decay, self.alpha_decay, self.start_decay, self.decay_type
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

        subtitle = "L={0} N={1} Neff={2} <br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            [ '||x||', '||x_single||', '||x_pair||','||g||', '||g_single||', '||g_pair||',
              'sum_qij_uneq_1', 'neg_qijab', 'sum_wij_uneq_0',
              'max_g', 'alpha'
              ],
            subtitle
        )
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
        alpha=self.alpha0
        beta1=self.beta1
        beta2=self.beta2
        beta3=self.beta3

        for i in range(self.maxit):

            #finish burn-in phase
            if i == 10:
                objfun.average_sample_counts = True

            fx, g = objfun.evaluate(x)

            #update moment vectors
            first_moment    = beta1 * first_moment + (1-beta1) * (g)
            second_moment   = beta2 * second_moment + (1-beta2) * (g*g)
            x_moment        = beta3 * x_moment + (1-beta3) * (x)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(beta1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(beta2, i+1))
            x_moment_corrected = x_moment / (1 - np.power(beta3, i+1))

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
            max_g = np.max(np.abs(g))

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||x||'][-self.convergence_prev-1]
                xnorm_diff = np.abs(xnorm_prev - xnorm) / xnorm_prev
                wij_diff = len(np.unique(
                    self.progress.optimization_log['sum_wij_uneq_0'][-self.convergence_prev - 1:] + [problems['sum_wij_uneq_0']])) - 1
            else:
                xnorm_diff = np.nan
                wij_diff = np.nan

            #update learning rate
            if self.decay and self.it_succesfull_stop_condition > -1:
                    if self.decay_type == "power":
                        alpha *= self.alpha_decay
                        #beta1 *= self.alpha_decay
                        #beta2 *= self.alpha_decay
                        #beta3 *= self.alpha_decay
                    elif self.decay_type == "lin":
                        alpha = self.alpha0  /(i - self.it_succesfull_stop_condition)
                    elif self.decay_type == "step":
                        alpha *= self.alpha_decay
                        self.start_decay *= 5e-1
                        self.it_succesfull_stop_condition = -1
                        # self.beta1 *= self.alpha_decay
                        # self.beta2 *= self.alpha_decay
                    elif self.decay_type == "sqrt":
                        alpha = self.alpha0 / np.sqrt(i - self.it_succesfull_stop_condition)
                        #beta1 = self.beta1  * np.power(0.99, (i-self.it_succesfull_stop_condition))
                        #beta2 = self.beta2  * np.power(0.99, (i-self.it_succesfull_stop_condition))
                    else:
                        alpha = self.alpha0 / (1 + (i - self.it_succesfull_stop_condition) / self.alpha_decay)

            #start decay at iteration i
            if self.decay and xnorm_diff < self.start_decay and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i





            #print out (and possiblly plot) progress
            self.progress.log_progress(i + 1,
                                       xnorm, np.sqrt(xnorm_single), np.sqrt(xnorm_pair),
                                       gnorm, np.sqrt(gnorm_single), np.sqrt(gnorm_pair),
                                       xnorm_diff=xnorm_diff, max_g=max_g, alpha=alpha,
                                       sum_qij_uneq_1=problems['sum_qij_uneq_1'],
                                       neg_qijab=problems['neg_qijab'],
                                       sum_wij_uneq_0=problems['sum_wij_uneq_0']
                                       )


            #stop condition
            if self.early_stopping:

                if self.qij_condition:
                    if (problems['sum_qij_uneq_1'] == 0) and (problems['neg_qijab'] == 0) and (wij_diff == 0):

                        ret = {
                        "code": 1,
                        "message": "Stopping condition (wij_diff == 0 and sum of qij violations = 0 ) successfull.".format(
                            self.epsilon)
                         }
                        return fx, x, ret

                elif xnorm_diff < self.epsilon:
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
