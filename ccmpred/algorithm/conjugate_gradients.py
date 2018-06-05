import numpy as np
import ccmpred.monitor.progress as pr


class conjugateGradient(object):
    """Optimize objective function usign conjugate gradients"""

    def __init__(
            self, ccm, maxit=100, ftol=1e-4, max_linesearch=5, alpha_mul=0.5, wolfe=0.2,
            epsilon=1e-3, convergence_prev=5):
        self.maxit = maxit
        self.ftol = ftol
        self.max_linesearch = max_linesearch
        self.alpha_mul = alpha_mul
        self.wolfe = wolfe
        self.epsilon = epsilon
        self.convergence_prev = convergence_prev

        #optimization progress logger
        self.progress = ccm.progress

        self.g_x = None


    def __repr__(self):
        return "conjugate gradient optimization (ftol={0} max_linesearch={1} alpha_mul={2} wolfe={3}) \n" \
               "\tconvergence criteria: maxit={4} epsilon={5} convergence_prev={6} \n".format(
            self.ftol, self.max_linesearch, self.alpha_mul, self.wolfe, self.maxit, self.epsilon, self.convergence_prev)


    def set_epsilon(self, eps):
        self.epsilon = eps

    def set_maxit(self, maxit):
        self.maxit = maxit


    def minimize(self, objfun, x):

        subtitle = self.progress.title + self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_title(subtitle)

        #for initialization of linesearch
        fx, g_x, g_reg = objfun.evaluate(x)
        g = g_x + g_reg
        gnorm = np.sum(g*g)

        # print and plot progress
        x_single, x_pair = objfun.linear_to_structured(x)
        g_single, g_pair = objfun.linear_to_structured(g)

        log_metrics={}
        log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
        log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
        log_metrics['||w||'] = np.sqrt(np.sum(x_pair * x_pair))
        log_metrics['||g||'] = np.sqrt(gnorm)
        log_metrics['||g_w||'] = np.sqrt(np.sum(g_pair * g_pair))
        log_metrics['||g_v||'] = np.sqrt(np.sum(g_single * g_single))
        log_metrics['max_g'] = np.max(g)
        log_metrics['fx'] = fx
        log_metrics['step'] = 0
        log_metrics['#lsearch'] = 0
        log_metrics['diff_fx'] = np.nan
        self.progress.log_progress(0, **log_metrics)


        gprevnorm = None
        alpha_prev = None
        dg_prev = None
        s = None

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations",
            "num_iterations": self.maxit
        }

        alpha = 1 / np.sqrt(gnorm)
        rel_diff_fx=np.nan
        for iteration in range(self.maxit):

            if iteration > 0:
                beta = gnorm / gprevnorm
                s = beta * s - g
                dg = np.sum(s * g)
                alpha = alpha_prev * dg_prev / dg
            else:
                s = -g
                dg = np.sum(s * g)


            n_linesearch, fx, alpha, g_x, g_reg, x = self.linesearch(x, fx, g, objfun, s, alpha)
            g = g_x + g_reg

            g_x_single, g_x_pair = objfun.linear_to_structured(g_x)
            self.g_x = objfun.structured_to_linear(g_x_single, g_x_pair/2)


            if n_linesearch < 0:
                ret['message'] = "Cannot find appropriate line search distance -- this might indicate a numerical problem with the gradient!"
                ret['code'] = -2
                ret['num_iterations'] = iteration
                break


            # convergence check
            if len(self.progress.optimization_log['fx']) >= self.convergence_prev:
                check_fx = self.progress.optimization_log['fx'][-self.convergence_prev]
                rel_diff_fx = (check_fx - fx) / check_fx
                if rel_diff_fx < self.epsilon:
                    ret['message'] = 'Success!'
                    ret['code'] = 0
                    ret['num_iterations'] = iteration
                    break



            #update optimization specific values
            gprevnorm = gnorm
            gnorm = np.sum(g * g)

            alpha_prev = alpha
            dg_prev = dg

            # print and plot progress
            x_single, x_pair = objfun.linear_to_structured(x)
            g_single, g_pair = objfun.linear_to_structured(g)

            log_metrics={}
            log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
            log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
            log_metrics['||w||'] = np.sqrt(np.sum(x_pair * x_pair))
            log_metrics['||g||'] = np.sqrt(gnorm)
            log_metrics['||g_w||'] = np.sqrt(np.sum(g_pair * g_pair))
            log_metrics['||g_v||'] = np.sqrt(np.sum(g_single * g_single))
            log_metrics['max_g'] = np.max(g)
            log_metrics['fx'] = fx
            log_metrics['step'] = alpha
            log_metrics['#lsearch'] = n_linesearch
            log_metrics['diff_fx'] = rel_diff_fx

            self.progress.log_progress(iteration+1, **log_metrics)



        return fx, x, ret


    def linesearch(self, x0, fx, g, objfun, s, alpha):
        dg_init = np.sum(g * s) #!!!!!!!!!!!!!!!!!!!! this was formerly dg_init = np.sum(g * g)
        dg_test = dg_init * self.ftol

        n_linesearch = 0
        fx_init = fx

        x = x0.copy()

        while True:
            if n_linesearch >= self.max_linesearch:
                return -1, fx, alpha, g, x

            n_linesearch += 1

            x = x0 + alpha * s

            fx_step, g_x, g_reg = objfun.evaluate(x)
            g = g_x + g_reg

            # armijo condition
            if fx_step < fx_init + alpha * dg_test:
                #print("fx_step: {0} fx_init + alpha * dg_test: {1} alpha * dg_test: {2} alpha {3}".format(fx_step, fx_init + alpha * dg_test, alpha * dg_test, alpha))

                dg = np.sum(s * g)
                #print("dg: {0} self.wolfe * dg_init: {1} dg_init: {2}".format(dg, self.wolfe * dg_init,  dg_init))
                if dg < self.wolfe * dg_init:
                    fx = fx_step
                    return n_linesearch, fx, alpha, g_x, g_reg, x

            alpha *= self.alpha_mul


    def get_gradient_x(self):


        return(self.g_x)



    def get_parameters(self):
        parameters={}

        parameters['convergence']={}
        parameters['convergence']['maxit'] = self.maxit
        parameters['convergence']['epsilon'] = self.epsilon
        parameters['convergence']['convergence_prev'] = self.convergence_prev

        parameters['ftol'] = self.ftol
        parameters['max_linesearch'] = self.max_linesearch
        parameters['alpha_mul'] = self.alpha_mul
        parameters['wolfe'] = self.wolfe

        return parameters