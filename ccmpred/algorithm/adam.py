import numpy as np
import ccmpred.logo
import ccmpred.monitor.progress as pr
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

    def __init__(
            self, ccm, alpha0=1e-3, beta1=0.9, beta2=0.999, beta3=0.9, noise=1e-8,
            maxit=100, epsilon=1e-5, convergence_prev=5, early_stopping=False,
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


        plot_title = "L={0} N={1} Neff={2} Diversity={3}<br>".format(
            ccm.L, ccm.N, np.round(ccm.neff, decimals=3),
            np.round(ccm.diversity, decimals=3)
        )
        self.progress = pr.Progress(plotfile, plot_title)


        if self.alpha0 == 0:
                self.alpha0 = 2e-3 * np.log(ccm.neff)

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
            #print g_pair[0,10,3,5], "==", g_pair[10,0,5,3] #yes, it is identical
            g_pair_flat = g_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            gx_pair_flat = gx_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            g_reg_pair_flat = g_reg_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()
            x_pair_flat = x_pair[upper_triangular_indices[0], upper_triangular_indices[1],:20,:20].flatten()



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


            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||w||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm_pair)) / xnorm_prev
            else:
                xnorm_diff = np.nan

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
                elif self.decay_type == "keras":
                    alpha = self.alpha0 / (1 + self.decay_rate * (t+1))
                    alpha *=  np.sqrt(1. - np.power(self.beta2, t+1)) / (1. - np.power(self.beta1,t+1))


            #print out progress
            log_metrics={}
            log_metrics['||w||'] = xnorm_pair
            log_metrics['||g||'] = np.sqrt(np.sum(g_pair_flat * g_pair_flat))
            log_metrics['||g_w||'] = np.sqrt(np.sum(gx_pair_flat * gx_pair_flat))
            log_metrics['||g_w||norm'] = log_metrics['||g_w||'] / len(gx_pair_flat)
            log_metrics['||g_reg_w||'] = np.sqrt(np.sum(g_reg_pair_flat * g_reg_pair_flat))
            log_metrics['xnorm_diff'] = xnorm_diff
            log_metrics['max_g'] = np.max(np.abs(gx))
            log_metrics['alpha'] = alpha


            if not self.fix_v:
                log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
                log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
                log_metrics['||g_v||'] = np.sqrt(np.sum(gx_single * gx_single))
                log_metrics['||g||'] = np.sqrt(np.sum(gx * gx))
                log_metrics['||g_reg_v||'] = np.sqrt(np.sum(g_reg_single * g_reg_single))

            self.progress.log_progress(i + 1, **log_metrics)


            #stop condition
            if self.early_stopping and  xnorm_diff < self.epsilon:
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