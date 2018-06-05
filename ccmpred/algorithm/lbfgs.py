import numpy as np
import ccmpred.monitor.progress as pr
from scipy.optimize import minimize as min

class LBFGS(object):
    """Optimize objective function usign lbfgs"""

    def __init__(self, progress, maxit=100, ftol=1e-4, max_linesearch=20, maxcor=5, non_contact_indices=None):

        self.max_linesearch=max_linesearch
        self.ftol = ftol
        self.maxit = maxit
        self.maxcor = maxcor

        # whether optimization is run with constraints (non-contacts are masked)
        self.non_contact_indices = non_contact_indices

        # optimization progress logger
        self.progress = progress

        self.g_x = None
        self.objfun=None
        self.iteration=0


    def __repr__(self):

        repr_str = "LBFGS optimization (ftol={0}, maxcor={1}, max_ls={2})\n".format(
            self.ftol,self.maxcor,self.max_linesearch)
        repr_str += "\tconvergence criteria: maxit={0} \n".format(self.maxit)

        return repr_str

    def lbfgs_f(self, x):

        fx, g_x, g_reg = self.objfun.evaluate(x)

        #gradient is computed x 2 in pll.evaluate because of compatibility with conjugate gradient optimization!!
        g_x_single, g_x_pair = self.objfun.linear_to_structured(g_x)
        g_reg_single, g_reg_pair = self.objfun.linear_to_structured(g_reg)
        g = self.objfun.structured_to_linear(g_x_single+g_reg_single, (g_x_pair+g_reg_pair)/2)

        # masking: set coupling gradients for all pairs (i,j) with d_ij > contact_thr = 0
        if self.non_contact_indices is not None:
            g_single, g_pair = self.objfun.linear_to_structured(g)
            g_pair[self.non_contact_indices[0], self.non_contact_indices[1], :, :] = 0
            g = self.objfun.structured_to_linear(g_single, g_pair)

        return fx, g

    def print_and_plot(self, x):

        self.iteration += 1

        x_single, x_pair = self.objfun.finalize(x)

        log_metrics={}
        log_metrics['||v+w||'] = np.sqrt(np.sum(x_single * x_single) + np.sum(x_pair * x_pair)/2)
        log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
        log_metrics['||w||'] = np.sqrt(np.sum(x_pair * x_pair)/2)
        self.progress.log_progress(self.iteration, **log_metrics)

    def minimize(self, objfun, x):

        self.objfun = objfun

        subtitle = self.progress.title + self.__repr__().replace("\n", "<br>")
        subtitle += objfun.__repr__().replace("\n", "<br>")
        self.progress.set_plot_title(subtitle)

        res = min(self.lbfgs_f,
            x,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxls': self.max_linesearch,
                'gtol': 1e-05,
                'eps': 1e-08,
                'maxiter': self.maxit,
                'ftol': self.ftol,
                'maxfun': 15000,
                'maxcor': self.maxcor,
                'disp': False
            },
            callback=self.print_and_plot
            )


        ret = {
            "code": res.status,
            "message": res.message.decode("utf-8"),
            "num_iterations": res.nit
        }

        return res.fun, res.x, ret

    def get_gradient_x(self):

        return(self.g_x)

    def get_parameters(self):
        parameters={}

        parameters['convergence']={}
        parameters['convergence']['maxit'] = self.maxit
        parameters['convergence']['max_linesearch'] = self.max_linesearch
        parameters['convergence']['ftol'] = self.ftol


        return parameters