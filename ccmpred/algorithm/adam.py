import numpy as np
import ccmpred.logo
import ccmpred.monitor.progress as pr
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

    def __init__(self, maxit=100, alpha0=1e-3, alpha_decay=1e1, momentum_estimate1=0.9, momentum_estimate2=0.999, noise=1e-7,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False,
                 decay=False, start_decay=1e-4, fix_v=False, group_alpha=True):
        self.maxit = maxit
        self.alpha0 = alpha0
        self.momentum_estimate1 = momentum_estimate1
        self.momentum_estimate2 = momentum_estimate2
        self.noise = noise
        self.decay=decay
        self.alpha_decay = alpha_decay
        self.start_decay = start_decay

        self.fix_v = fix_v
        self.group_alpha = group_alpha

        self.early_stopping = early_stopping
        self.it_succesfull_stop_condition=-1
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

        self.progress = pr.Progress(plotfile=None,
                                    xnorm_diff=[], max_g=[], gnorm_diff=[], alpha=[])


    def __repr__(self):
        return "Adam stochastic optimization (decay={0} learning_rate={1} momentum_estimate1={2} momentum_estimate2={3} noise={4} fix_v={5}) \n" \
               "convergence criteria: maxit={6} early_stopping={7} epsilon={8} prev={9}".format(
            self.decay, self.alpha0, self.momentum_estimate1, self.momentum_estimate2, self.noise, self.fix_v,
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


    def minimize(self, objfun, x, plotfile):

        subtitle = "L={0} N={1} Neff={2} <br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            ['||x||', '||x_single||', '||x_pair||', '||g||', '||g_single||', '||g_pair||','max_g', 'alpha'],
            subtitle
        )
        self.progress.begin_process()


        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1
        #objfun.msa_sampled =  objfun.init_sample_alignment(5 * objfun.ncol)
        for i in range(self.maxit):


            fx, g = objfun.evaluate(x)

            #update moment vectors
            first_moment    = self.momentum_estimate1 * first_moment + (1-self.momentum_estimate1) * (g)
            second_moment   = self.momentum_estimate2 * second_moment + (1-self.momentum_estimate2) * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(self.momentum_estimate1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.momentum_estimate2, i+1))

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

            # # possible stopping criteria
            # if len(self.lastg) != 0:
            #     self.g_sign.append(np.mean(np.sign(self.lastg * g.copy())))
            # self.lastg = g.copy()
            #
            # self.x_hist.append(xnorm)
            # self.g_hist.append(gnorm)
            #
            # xnorm_diff = 1
            # gnorm_diff = 1
            # sign_g_t10, sign_g_t8, sign_g_t7, sign_g_t6 = [0, 0, 0, 0]
            # if len(self.g_sign) > 10:
            #     self.g_sign.popleft()
            #     sign_g_t10 = np.sum(self.g_sign) / 10.0
            #     sign_g_t8 = np.sum(list(self.g_sign)[2:]) / 8.0
            #
            #     self.x_hist.popleft()
            #     xnorm_prev = self.x_hist[-self.convergence_prev-1]
            #     xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev
            #
            #     self.g_hist.popleft()
            #     gnorm_prev = self.g_hist[-self.convergence_prev-1]
            #     gnorm_diff = np.abs((gnorm_prev - gnorm)) / gnorm_prev
            #
            #
            # if sign_g_t8 < 0:
            #     self.neg_g_sign += 1
            # else:
            #     self.neg_g_sign = 0

            # if self.neg_g_sign > 10:
            #     ret = {
            #         "code": 0,
            #         "message": "Stopping condition (change of gradient direction) successfull."
            #     }
            #
            #     return fx, x, ret
            # ====================================================================================


            #start decay at iteration i
            if xnorm_diff < self.start_decay and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i

            #update learning rate
            alpha=self.alpha0
            if self.decay and self.it_succesfull_stop_condition > -1:
                    #alpha /= np.sqrt(i  - self.it_succesfull_stop_condition)
                    alpha /= (1 + (i - self.it_succesfull_stop_condition) / self.alpha_decay)

            #print out progress
            self.progress.log_progress(i + 1,
                                       xnorm_single, xnorm_pair,
                                       gnorm_single, gnorm_pair,
                                       xnorm_diff=xnorm_diff, max_g=max_g, gnorm_diff=gnorm_diff, alpha=alpha)


            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:

                    # if not self.decay:
                    #
                    #     objfun.persistent=True
                    #     objfun.msa_sampled = objfun.init_sample_alignment(np.min([np.max([50 * objfun.ncol, objfun.nrow]), 10000]))
                    #
                    #     #self.momentum_estimate1 = 0.5
                    #     #self.momentum_estimate2 = 0.555
                    #
                    #     self.epsilon *= 1e-4
                    #
                    #     self.decay=True
                    #     self.alpha_decay=5.0
                    #     self.it_succesfull_stop_condition=i
                    #
                    #     print("Turn on decay and persistent CD. Decrease epsilon to {0}. ".format(self.epsilon))
                    # if not self.decay and objfun.gibbs_steps == 1 and not objfun.persistent:
                    #     objfun.gibbs_steps = 5
                    #     self.epsilon *= 1e-1        #decrease convergence criterion
                    #     print("Use 5 Gibss sampling steps. Set learning rate to {0}. Decrease epsilon to {1}".format(self.learning_rate, self.epsilon))
                    # elif not self.decay and objfun.gibbs_steps == 5:
                    #     self.decay = True
                    #     self.epsilon *= 1e-1        #decrease convergence criterion
                    #     self.it_succesfull_stop_condition=i
                    #     print("Turn on decaying learning rate. Use 5 Gibss sampling steps. Decrease epsilon to {0}".format(self.epsilon))
                    # elif self.decay and self.decrease > 1:
                    #     self.decrease -= 1.0
                    #     print("Decrease decay. Set deacay to {0}".format(self.decrease))
                    # elif self.decay and not objfun.persistent:
                    #     objfun.persistent=True
                    #     objfun.gibbs_steps = 1
                    #     #self.epsilon *= 1e-1        #decrease convergence criterion
                    #     self.learning_rate = 1e-3
                    #     self.decrease = 10.0
                    #     self.it_succesfull_stop_condition=i
                    #     print("Turn on persistent CD. Decrease epsilon to {0}. Set learnign rate to {1}. Set decrease tp {2}.".format(self.epsilon, self.learning_rate, self.decrease))
                    # else:
                    # else:
                    ret = {
                        "code": 0,
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                    }
                    return fx, x, ret



            first_moment_corrected_single, first_moment_corrected_pair = objfun.linear_to_structured(first_moment_corrected, objfun.ncol)
            second_moment_corrected_single, second_moment_corrected_pair = objfun.linear_to_structured(second_moment_corrected, objfun.ncol)

            #use same scaing of gradients for each group!, e.g v_i and w_ij
            if self.group_alpha:
                step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single.max(1))[:, np.newaxis] + self.noise)
                step_pair   = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair.max(3).max(2))[:, :, np.newaxis, np.newaxis] + self.noise)
            else:
                step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single) + self.noise)
                step_pair   = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair) + self.noise)


            #update parameters
            if not self.fix_v:
                x_single -= alpha * step_single
            x_pair -= alpha * step_pair

            x=objfun.structured_to_linear(x_single, x_pair)
            #x -= alpha * first_moment_corrected / ( np.sqrt(second_moment_corrected) + self.noise)


        return fx, x, ret
