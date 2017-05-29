import numpy as np
import ccmpred.logo
import sys
import ccmpred.monitor.progress as pr
import ccmpred.model_probabilities


class conjugateGradient():
    """Optimize objective function usign conjugate gradients"""

    def __init__(self, maxit=100, ftol=1e-4, max_linesearch=5, alpha_mul=0.5, wolfe=0.2, epsilon=1e-3, convergence_prev=5):
        self.maxit = maxit
        self.ftol = ftol
        self.max_linesearch = max_linesearch
        self.alpha_mul = alpha_mul
        self.wolfe = wolfe
        self.epsilon = epsilon
        self.convergence_prev = convergence_prev

        metrics=['xnorm', 'xnorm_single','xnorm_pair', 'gnorm', 'gnorm_single', 'gnorm_pair',
                 'fx', 'step','n_linesearch', 'xnorm_pair', 'rel_diff_fx', 'max_g',
                 'sum_deviation_wij']

        self.progress = pr.Progress(metrics=metrics)





    def __repr__(self):
        return "conjugate gradient optimization (ftol={0} max_linesearch={1} alpha_mul={2} wolfe={3}) \n" \
               "convergence criteria: maxit={4} epsilon={5} convergence_prev={6} ".format(
            self.ftol, self.max_linesearch, self.alpha_mul, self.wolfe, self.maxit, self.epsilon, self.convergence_prev)

    def set_epsilon(self, eps):
        self.epsilon = eps

    def set_maxit(self, maxit):
        self.maxit = maxit


    def minimize(self, objfun, x, plotfile):


        diversity = np.sqrt(objfun.nrow)/objfun.ncol
        subtitle = "L={0} N={1} Neff={2} Diversity={3}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3), np.round(diversity,decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_options(plotfile, subtitle)
        self.progress.begin_process()



        #for initialization of linesearch
        fx, g = objfun.evaluate(x)


        x_single, x_pair = objfun.linear_to_structured(x)
        g_single, g_pair = objfun.linear_to_structured(g)

        xnorm_single = np.sum(x_single * x_single)
        xnorm_pair = np.sum(x_pair * x_pair)

        gnorm_single = np.sum(g_single * g_single)
        gnorm_pair = np.sum(g_pair * g_pair)
        gnorm = gnorm_single + gnorm_pair
        max_g = np.max(g)

        # print out progress
        self.progress.log_progress(
            0,
            xnorm= np.sqrt(xnorm_single+xnorm_pair),
            xnorm_single= np.sqrt(xnorm_single),
            xnorm_pair= np.sqrt(xnorm_pair),
            gnorm=np.sqrt(gnorm_single+gnorm_pair),
            gnorm_single=np.sqrt(gnorm_single),
            gnorm_pair=np.sqrt(gnorm_pair),
            fx=fx,
            step=0,
            n_linesearch=0,
            max_g=max_g,
            rel_diff_fx=np.nan,
            sum_deviation_wij=0
        )


        gprevnorm = None
        alpha_prev = None
        dg_prev = None
        s = None

        ret = {
            "message": "Unknown",
            "code": -9999
        }



        alpha = 1 / np.sqrt(gnorm)
        iteration = 0
        rel_diff_fx=np.nan
        while True:
            if iteration >= self.maxit:
                ret['message'] = "Reached maximum number of iterations"
                ret['code'] = 2
                break

            if iteration > 0:
                beta = gnorm / gprevnorm
                s = beta * s - g
                dg = np.sum(s * g)
                alpha = alpha_prev * dg_prev / dg
            else:
                s = -g
                dg = np.sum(s * g)


            n_linesearch, fx, alpha, g, x = self.linesearch(x, fx, g, objfun, s, alpha)

            if n_linesearch < 0:
                ret['message'] = "Cannot find appropriate line search distance -- this might indicate a numerical problem with the gradient!"
                ret['code'] = -2
                break


            # convergence check
            if len(self.progress.optimization_log['fx']) >= self.convergence_prev:
                check_fx = self.progress.optimization_log['fx'][-self.convergence_prev]
                rel_diff_fx = (check_fx - fx) / check_fx
                if rel_diff_fx < self.epsilon:
                    ret['message'] = 'Success!'
                    ret['code'] = 0
                    break

            #for plotting
            x_single, x_pair = objfun.linear_to_structured(x)
            xnorm_single = np.sum(x_single * x_single)
            xnorm_pair   = np.sum(x_pair * x_pair)

            g_single, g_pair = objfun.linear_to_structured(g)
            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)

            max_g = np.max(g)

            #update optimization specific values
            gprevnorm = gnorm
            gnorm = gnorm_single + gnorm_pair

            alpha_prev = alpha
            dg_prev = dg

            iteration += 1


            #compute number of problems with qij
            problems = ccmpred.model_probabilities.get_nr_problematic_qij(
                objfun.freqs_pair, x_pair, objfun.regularization.lambda_pair, objfun.Nij, epsilon=1e-2, verbose=False)


            # print out progress
            self.progress.log_progress(
                iteration,
                xnorm=np.sqrt(xnorm_single + xnorm_pair),
                xnorm_single=np.sqrt(xnorm_single),
                xnorm_pair=np.sqrt(xnorm_pair),
                gnorm=np.sqrt(gnorm_single + gnorm_pair),
                gnorm_single=np.sqrt(gnorm_single),
                gnorm_pair=np.sqrt(gnorm_pair),
                fx=fx,
                step=alpha,
                n_linesearch=n_linesearch,
                max_g=max_g,
                rel_diff_fx=rel_diff_fx,
                sum_deviation_wij=problems['sum_deviation_wij']
            )



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

            fx_step, g = objfun.evaluate(x)

            # armijo condition
            if fx_step < fx_init + alpha * dg_test:
                #print("fx_step: {0} fx_init + alpha * dg_test: {1} alpha * dg_test: {2} alpha {3}".format(fx_step, fx_init + alpha * dg_test, alpha * dg_test, alpha))

                dg = np.sum(s * g)
                #print("dg: {0} self.wolfe * dg_init: {1} dg_init: {2}".format(dg, self.wolfe * dg_init,  dg_init))
                if dg < self.wolfe * dg_init:
                    fx = fx_step
                    return n_linesearch, fx, alpha, g, x

            alpha *= self.alpha_mul

