import numpy as np
import ccmpred.logo
import ccmpred.monitor.progress as pr


class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, ccm, maxit=100, alpha0=5e-3,
                 decay=True,  decay_start=1e-3, decay_rate=10, decay_type="lin",
                 fix_v=False, epsilon=1e-5, convergence_prev=5, early_stopping=False):


        self.maxit = maxit
        self.alpha0 = alpha0

        #initial learning rate defined wrt to effective number of sequences
        if self.alpha0 == 0:
                self.alpha0 = 5e-2 / np.sqrt(ccm.neff)

        #decay settings
        self.decay=decay
        self.decay_start = decay_start
        self.decay_rate = np.float(decay_rate)
        self.decay_type = decay_type
        self.it_succesfull_stop_condition=-1

        #single potentials will not be optimized if fix_v=True
        self.fix_v=fix_v

        #convergence settings for optimization
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

        #whether optimization is run with constraints (non-contacts are masked)
        self.non_contact_indices = ccm.non_contact_indices

        #optimization progress logger
        self.progress = ccm.progress



    def __repr__(self):
        rep_str="Gradient descent optimization (alpha0={0})\n".format( np.round(self.alpha0, decimals=8))

        rep_str+="convergence criteria: maxit={0} early_stopping={1} epsilon={2} prev={3}\n".format(
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)

        if self.decay:
            rep_str+="\t\t\tdecay: decay={0} decay_rate={1} decay_start={2} \n".format(
               self.decay, np.round(self.decay_rate, decimals=8), self.decay_start
            )
        else:
            rep_str+="\t\t\tdecay: decay={0}\n".format(
              self.decay
            )

        return rep_str

    def minimize(self, objfun, x):

        subtitle = self.progress.title + self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_title(subtitle)

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations",
            "num_iterations": self.maxit
        }

        fx = -1
        alpha = self.alpha0
        persistent=False
        for i in range(self.maxit):

            #in case CD has property persistent=True
            #turn on persistent CD when learning rate is small enough
            if objfun.persistent and alpha < self.alpha0/10:
                persistent=True

            fx, gx, greg = objfun.evaluate(x, persistent)
            g = gx + greg

            #decompose gradients and parameters
            x_single, x_pair = objfun.linear_to_structured(x)
            g_single, g_pair = objfun.linear_to_structured(g)
            gx_single, gx_pair = objfun.linear_to_structured(gx)
            g_reg_single, g_reg_pair = objfun.linear_to_structured(greg)

            #masking: set coupling gradients for all pairs (i,j) with d_ij > contact_thr = 0
            if self.non_contact_indices is not None:
                g_pair[self.non_contact_indices[0], self.non_contact_indices[1], :, :] = 0


            #compute norm of coupling parameters
            xnorm_pair = np.sqrt(np.sum(x_pair * x_pair)/2) #np.sqrt(np.sum(x_pair * x_pair))

            if i > self.convergence_prev:
                xnorm_prev = self.progress.optimization_log['||w||'][-self.convergence_prev]
                xnorm_diff = np.abs((xnorm_prev - xnorm_pair)) / xnorm_prev
            else:
                xnorm_diff = 1.0

            #start decay at iteration i
            if self.decay and xnorm_diff < self.decay_start and self.it_succesfull_stop_condition < 0:
                self.it_succesfull_stop_condition = i

            #new step size
            if self.it_succesfull_stop_condition > 0:
                t = i - self.it_succesfull_stop_condition + 1
                if self.decay_type == "lin":
                    alpha = self.alpha0 / (1 + self.decay_rate * t)
                if self.decay_type == "sig":
                    alpha *= 1.0 / (1 + self.decay_rate * t)
                if self.decay_type == "sqrt":
                    alpha = self.alpha0 / np.sqrt(1 + self.decay_rate * t)
                if self.decay_type == "exp":
                    alpha = self.alpha0  * np.exp(- self.decay_rate * t)


            #print out progress
            log_metrics={}
            log_metrics['||w||'] = xnorm_pair
            log_metrics['||g||'] = np.sqrt(np.sum(g_pair * g_pair)/2)
            log_metrics['||g_w||'] = np.sqrt(np.sum(gx_pair * gx_pair)/2)
            log_metrics['||greg_w||'] = np.sqrt(np.sum(g_reg_pair * g_reg_pair)/2)
            log_metrics['xnorm_diff'] = xnorm_diff
            log_metrics['max_g'] = np.max(np.abs(gx))
            log_metrics['alpha'] = alpha
            log_metrics['PCD'] = persistent

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
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon),
                        "num_iterations": i
                    }
                    return fx, x, ret

            # update parameters
            if not self.fix_v:
                x_single -= alpha * g_single
            x_pair -=  alpha * g_pair

            x = objfun.structured_to_linear(x_single, x_pair)

        return fx, x, ret

    def get_parameters(self):
        parameters={}

        parameters['convergence'] = {}
        parameters['convergence']['maxit'] = self.maxit
        parameters['convergence']['early_stopping'] = self.early_stopping
        parameters['convergence']['epsilon'] = self.epsilon
        parameters['convergence']['convergence_prev'] = self.convergence_prev

        parameters['decay'] = {}
        parameters['decay']['alpha0'] =  self.alpha0
        parameters['decay']['decay'] = self.decay
        parameters['decay']['decay_start'] = self.decay_start
        parameters['decay']['decay_rate'] = self.decay_rate
        parameters['decay']['decay_type'] = self.decay_type

        parameters['fix_v'] = self.fix_v

        return parameters
