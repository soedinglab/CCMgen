import numpy as np
import ccmpred.logo
import ccmpred.monitor.progress as pr
from collections import deque
import ccmpred.model_probabilities
import ccmpred.sanity_check
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

    m = mom1*m + (1-mom1)*dx
    v = mom2*v + (1-mom2)*(dx**2)
    x += - learning_rate * m / (np.sqrt(v) + eps)


    If setting mom1=0, then Adam becomes RMSProp as described by Hinton here:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    v = mom2*v + (1-mom2)*dx**2
    x += - learning_rate * dx / (np.sqrt(v) + eps)

    """

    def __init__(self, ccm, alpha0=1e-3, beta1=0.9, beta2=0.999, beta3=0.9, noise=1e-8,
                 maxit=100, epsilon=1e-5, convergence_prev=5, early_stopping=False, qij_condition=False,
                 decay_type="step", decay_rate=1e1, decay=False, decay_start=1e-4,
                 fix_v=False, plotfile=None):

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


        self.fix_v = fix_v

        self.maxit = maxit
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev
        self.qij_condition = qij_condition


        self.gpair_sign = deque([], maxlen=self.convergence_prev)

        plot_title = "L={0} N={1} Neff={2} Diversity={3}<br>".format(
            ccm.L, ccm.N, np.round(ccm.neff, decimals=3),
            np.round(ccm.diversity, decimals=3)
        )
        self.progress = pr.Progress(plotfile, plot_title)


        if self.alpha0 == 0:
                self.alpha0 = 2e-3 *   np.log(ccm.neff)

    def __repr__(self):

        rep_str="Adam (beta1={0} beta2={1} beta3={2} alpha0={3} noise={4} fix-v={5}) \n".format(
            self.beta1, self.beta2, self.beta3, np.round(self.alpha0, decimals=5), self.noise, self.fix_v
        )

        if self.decay:
            rep_str+="\tdecay: decay={0} decay-rate={1} decay-start={2} decay-type={3}\n".format(
                self.decay, np.round(self.decay_rate, decimals=3), self.decay_start, self.decay_type
            )
        else:
            rep_str+="\tdecay: decay={0}\n".format(
              self.decay
            )

        if self.early_stopping:
            rep_str+="\tconvergence criteria: maxit={0} early-stopping={1} epsilon={2} prev={3}\n".format(
                self.maxit, self.early_stopping, self.epsilon, self.convergence_prev
            )
        else:
            rep_str+="\tconvergence criteria: maxit={0} early-stopping={1}\n".format(
                self.maxit, self.early_stopping
            )

        return rep_str

    def minimize(self, objfun, x):


        subtitle = self.progress.title + self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_title(subtitle)


        #initialize the moment vectors
        first_moment_pair = np.zeros((objfun.ncol,objfun.ncol, 21 , 21))
        second_moment_pair = np.zeros((objfun.ncol, objfun.ncol, 21 , 21))
        x_moment_pair = np.zeros((objfun.ncol, objfun.ncol, 21, 21))
        first_moment_single = np.zeros((objfun.ncol, 20))
        second_moment_single = np.zeros((objfun.ncol, 20))
        x_moment_single = np.zeros((objfun.ncol, 20))

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations",
            "num_iterations": self.maxit
        }

        #test for persistent contrastive divergence!
        if objfun.persistent and self.epsilon > 1e-8:
            objfun.persistent = False

        upper_triangular_indices = np.triu_indices(objfun.ncol, k=1)
        fx = -1
        alpha=self.alpha0
        for i in range(self.maxit):

            fx, gx, greg = objfun.evaluate(x)
            g = gx + greg

            #decompose gradients and parameters
            x_single, x_pair = objfun.linear_to_structured(x)
            gx_single, gx_pair = objfun.linear_to_structured(gx)
            g_reg_single, g_reg_pair = objfun.linear_to_structured(greg)
            g_single, g_pair = objfun.linear_to_structured(g)


            #flattened
            #print g_pair[0,10,3,5], "==", g_pair[10,0,5,3] #yes it is identical :)
            g_pair_flat = g_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            gx_pair_flat = gx_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            g_reg_pair_flat = g_reg_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            x_pair_flat = x_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()


            #compute the percentage of gradient sign changes over the last "convergence_prev" iterations
            gpair_sign_current = np.sign(gx_pair_flat)
            self.gpair_sign.append(gpair_sign_current)
            nr_gradients_non_zero = len(np.where(np.sum(np.array(self.gpair_sign) == 0, axis=0) < len(self.gpair_sign))[0])
            nr_gradients_with_changes = len(np.where(np.sum(np.abs(np.diff(np.array(self.gpair_sign), axis=0)), axis=0) > 0)[0])

            ids_nonzero = set(np.where(np.sum(np.array(self.gpair_sign) == 0, axis=0) < len(self.gpair_sign))[0])
            ids_changes = set(np.where(np.sum(np.abs(np.diff(np.array(self.gpair_sign), axis=0)), axis=0) > 0)[0])
            ids_nonzero_nochanges = list(ids_nonzero - ids_changes)
            #print ids_nonzero_nochanges[:5], gpair_sign_current[ids_nonzero_nochanges[:5]], g_pair.flatten()[ids_nonzero_nochanges[:5]], x_pair.flatten()[ids_nonzero_nochanges[:5]]

            mean_nr_changes = np.sum(np.diff(np.array(self.gpair_sign), axis=0) != 0) / np.float(nr_gradients_non_zero)
            percent_gradient_changes = nr_gradients_with_changes / float(nr_gradients_non_zero)




            #update moment, adaptivity and parameter averages
            first_moment_pair    = self.beta1 * first_moment_pair + (1-self.beta1) * (g_pair)
            second_moment_pair   = self.beta2 * second_moment_pair + (1-self.beta2) * (g_pair**2)
            x_moment_pair        = self.beta3 * x_moment_pair + (1-self.beta3) * (x_pair)

            #compute bias corrected moments
            first_moment_corrected_pair     = first_moment_pair     / (1 - np.power(self.beta1, i+1))
            second_moment_corrected_pair    = second_moment_pair    / (1 - np.power(self.beta2, i+1))
            x_moment_corrected_pair         = x_moment_pair         / (1 - np.power(self.beta3, i+1))

            #compute the update step
            step_pair   = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair) + self.noise)

            if not self.fix_v:
                #update moment, adaptivity and parameter averages
                first_moment_single    = self.beta1 * first_moment_single + (1-self.beta1) * (g_single)
                second_moment_single   = self.beta2 * second_moment_single + (1-self.beta2) * (g_single*g_single)
                x_moment_single        = self.beta3 * x_moment_single + (1-self.beta3) * (x_single)

                #compute bias corrected moments
                first_moment_corrected_single  = first_moment_single / (1 - np.power(self.beta1, i+1))
                second_moment_corrected_single = second_moment_single / (1 - np.power(self.beta2, i+1))
                x_moment_corrected_single = x_moment_single / (1 - np.power(self.beta3, i+1))

                #compute the update step
                step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single) + self.noise)


            #compute norm of coupling parameters
            xnorm_pair = np.sqrt(np.sum(x_pair_flat * x_pair_flat))  # np.sqrt(np.sum(x_pair * x_pair))

            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||w||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm_pair)) / xnorm_prev

                wij_diff_prev = self.progress.optimization_log['sum_w'][-self.convergence_prev]
                wij_diff = np.abs((wij_diff_prev - problems['sum_deviation_wij'])) / wij_diff_prev

            else:
                xnorm_diff = np.nan
                wij_diff = np.nan


            #start decay at iteration i
            if self.decay and xnorm_diff < self.decay_start and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i

            #update learning rate
            if self.decay and self.it_succesfull_stop_condition > -1:
                t = i - self.it_succesfull_stop_condition
                if self.decay_type == "sig":
                    alpha *= 1.0 / (1 + self.decay_rate * t)
                elif self.decay_type == "power":
                    alpha *= self.decay_rate
                elif self.decay_type == "lin":
                    alpha = self.alpha0 / (1 + t * self.decay_rate)
                elif self.decay_type == "step":
                    alpha *= self.decay_rate
                    self.decay_start *= 5e-1
                    self.it_succesfull_stop_condition = -1
                elif self.decay_type == "sqrt":
                    alpha = self.alpha0  / (1 + (np.sqrt(1 + t)) / self.decay_rate)
                elif self.decay_type == "sqrt":
                    alpha *= 1.0  / (1 + (np.sqrt(1 + t)) / self.decay_rate)
                elif self.decay_type == "keras":
                    alpha = self.alpha0 / (1 + self.decay_rate * (t+1))
                    alpha *=  np.sqrt(1. - np.power(self.beta2, t+1)) / (1. - np.power(self.beta1,t+1))



            #print out progress
            log_metrics={}
            log_metrics['||w||'] = xnorm_pair
            log_metrics['mean_gw_std'] = np.mean(gx_pair_flat) / np.std(gx_pair_flat)
            log_metrics['meanchange'] = mean_nr_changes
            log_metrics['%grad'] = percent_gradient_changes
            log_metrics['||g||'] = np.sqrt(np.sum(g_pair_flat * g_pair_flat))
            log_metrics['||g_w||'] = np.sqrt(np.sum(gx_pair_flat * gx_pair_flat))
            log_metrics['||g_w||norm'] = log_metrics['||g_w||'] / len(gx_pair_flat)
            log_metrics['||greg_w||'] = np.sqrt(np.sum(g_reg_pair_flat * g_reg_pair_flat))
            log_metrics['xnorm_diff'] = xnorm_diff
            log_metrics['max_g'] = np.max(np.abs(gx))
            log_metrics['alpha'] = alpha
            log_metrics['#qij_uneq_1'] = problems['sum_qij_uneq_1']
            log_metrics['#neg_qij'] = problems['neg_qijab']
            log_metrics['#wij_uneq_0'] = problems['sum_wij_uneq_0']
            log_metrics['sum_w'] = problems['sum_deviation_wij']


            if not self.fix_v:
                log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
                log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
                log_metrics['||g_v||'] = np.sqrt(np.sum(gx_single * gx_single))
                log_metrics['||g||'] = np.sqrt(np.sum(gx * gx))
                log_metrics['||g_reg_v||'] = np.sqrt(np.sum(g_reg_single * g_reg_single))

            self.progress.log_progress(i + 1, **log_metrics)


            #stop condition
            if self.early_stopping and  xnorm_diff < self.epsilon:

                if self.epsilon > 1e-8:
                    objfun.persistent = True
                    self.epsilon = 1e-8
                else:
                    if self.qij_condition:

                        if (problems['sum_qij_uneq_1'] == 0) and (problems['neg_qijab'] == 0):
                            ret = {
                                "code": 1,
                                "message": "Stopping condition (xnorm diff < {0} and #qij violations = 0 ) successfull.".format(self.epsilon),
                                "num_iterations": i
                             }
                            return fx, x, ret
                    else:
                        ret = {
                            "code": 1,
                            "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon),
                            "num_iterations": i
                        }
                        return fx, x, ret


            #update parameters
            if not self.fix_v:
                x_single =x_moment_corrected_single - alpha * step_single
            x_pair = x_moment_corrected_pair - alpha * step_pair

            x=objfun.structured_to_linear(x_single, x_pair)

            if xnorm_diff < 1e-6:
                objfun.averaging=True

        #write header line
        self.progress.print_header()
        return fx, x, ret

    def get_parameters(self):
        parameters={}

        parameters['convergence']={}
        parameters['convergence']['maxit'] = self.maxit
        parameters['convergence']['early_stopping'] = self.early_stopping
        parameters['convergence']['epsilon'] = self.epsilon
        parameters['convergence']['convergence_prev'] = self.convergence_prev
        parameters['convergence']['qij_condition'] = self.qij_condition

        parameters['decay']={}
        parameters['decay']['alpha0'] = self.alpha0
        parameters['decay']['decay'] = self.decay
        parameters['decay']['decay_rate'] = self.decay_rate
        parameters['decay']['decay_start'] = self.decay_start
        parameters['decay']['decay_type'] = self.decay_type

        parameters['beta1'] = self.beta1
        parameters['beta2'] = self.beta2
        parameters['beta3'] = self.beta3
        parameters['noise'] = self.noise
        parameters['fix_v'] = self.fix_v

        return parameters