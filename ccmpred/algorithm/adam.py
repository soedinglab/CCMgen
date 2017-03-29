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

    def __init__(self, maxit=100, alpha0=1e-3, alpha_decay=1e1, beta1=0.9, beta2=0.999, noise=1e-8,
                 epsilon=1e-5, convergence_prev=5, early_stopping=False,
                 decay=False, start_decay=1e-4, fix_v=False, group_alpha=False, qij_condition=False):
        self.maxit = maxit
        self.alpha0 = alpha0
        self.beta1 = beta1
        self.beta2 = beta2
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
        self.qij_condition = qij_condition

        self.progress = pr.Progress(plotfile=None,
                                    xnorm_diff=[], max_g=[], alpha=[], beta1=[], beta2=[])


    def __repr__(self):
        return "Adam stochastic optimization (decay={0} learning_rate={1} beta1={2} beta2={3} noise={4} fix_v={5}) \n" \
               "convergence criteria: maxit={6} early_stopping={7} epsilon={8} prev={9}".format(
            self.decay, self.alpha0, self.beta1, self.beta2, self.noise, self.fix_v,
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


    def minimize(self, objfun, x, plotfile):

        subtitle = "L={0} N={1} Neff={2} <br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            [ '||x||', '||x_single||', '||x_pair||','||g||', '||g_single||', '||g_pair||','beta1','beta2', 'max_g', 'alpha'],
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
        alpha=self.alpha0
        for i in range(self.maxit):


            fx, g = objfun.evaluate(x)

            #update moment vectors
            first_moment    = self.beta1 * first_moment + (1-self.beta1) * (g)
            second_moment   = self.beta2 * second_moment + (1-self.beta2) * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(self.beta1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.beta2, i+1))

            first_moment_corrected_single, first_moment_corrected_pair = objfun.linear_to_structured(first_moment_corrected, objfun.ncol)
            second_moment_corrected_single, second_moment_corrected_pair = objfun.linear_to_structured(second_moment_corrected, objfun.ncol)

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

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||x||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev
            else:
                xnorm_diff = 1


            # step decay: reduce the learning rate by a constant (e.g. 0.5) whenever the xnorm < eps
            if self.decay and xnorm_diff < self.start_decay:
                alpha *= self.alpha_decay
                self.beta1 *= self.alpha_decay
                self.beta2 *= self.alpha_decay
                self.start_decay *= 5e-1

            # #start decay at iteration i
            # if xnorm_diff < self.start_decay and self.it_succesfull_stop_condition < 0:
            #     self.it_succesfull_stop_condition = i
            #
            # #update learning rate
            # alpha=self.alpha0
            # if self.decay and self.it_succesfull_stop_condition > -1:
            #         #alpha /= np.sqrt(i  - self.it_succesfull_stop_condition)
            #         alpha /= (1 + (i - self.it_succesfull_stop_condition) / self.alpha_decay)


            #print out progress
            self.progress.log_progress(i + 1,
                                       xnorm, np.sqrt(xnorm_single), np.sqrt(xnorm_pair),
                                       gnorm, np.sqrt(gnorm_single), np.sqrt(gnorm_pair),
                                       xnorm_diff=xnorm_diff, max_g=max_g,
                                       beta1=self.beta1,beta2=self.beta2, alpha=alpha)

            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:

                    #compute q_ij
                    nr_pairs_qij_error = 0
                    if self.qij_condition:
                        _, x_pair_centered = ccmpred.sanity_check.normalize_potentials(x_single, x_pair)
                        model_prob_flat, nr_pairs_qij_error = ccmpred.model_probabilities.compute_qij(
                            objfun.freqs_pair, x_pair_centered, objfun.regularization.lambda_pair, objfun.Nij, verbose=False
                        )
                    if nr_pairs_qij_error == 0:
                        ret = {
                            "code": 0,
                            "message": "Stopping condition (xnorm diff < {0}) successfull and q_ij ok.".format(self.epsilon)
                        }
                        return fx, x, ret
                    else:
                        print("Stopping condition (xnorm diff < {0}) successfull but {1} pair(s) with q_ijab < 0".format(
                            self.epsilon, nr_pairs_qij_error)
                        )


            #update parameters
            if not self.fix_v:
                x_single -= alpha * step_single
            x_pair -= alpha * step_pair

            x=objfun.structured_to_linear(x_single, x_pair)
            #x -= alpha * first_moment_corrected / ( np.sqrt(second_moment_corrected) + self.noise)


        return fx, x, ret
