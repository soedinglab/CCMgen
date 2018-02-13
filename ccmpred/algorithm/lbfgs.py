import numpy as np
import ccmpred.monitor.progress as pr
from scipy.optimize import minimize as min

class LBFGS(object):
    """Optimize objective function usign lbfgs"""

    def __init__(
            self, ccm,
            maxit=100, ftol=1e-4, max_linesearch=20, maxcor=5,
            plotfile=None):

        self.max_linesearch=max_linesearch
        self.ftol = ftol
        self.maxit = maxit
        self.maxcor = maxcor

        plot_title = "L={0} N={1} Neff={2} Diversity={3}<br>".format(
            ccm.L, ccm.N, np.round(ccm.neff, decimals=3),
            np.round(ccm.diversity, decimals=3)
        )
        self.progress = pr.Progress(plotfile, plot_title)

        self.g_x = None
        self.objfun=None
        self.iteration=0


    def __repr__(self):
        return "LBFGS optimization  \n" \
               "\tconvergence criteria: maxit={0} \n".format(
             self.maxit)

    def lbfgs_f(self, x, *args):

        fx, g_x, g_reg = self.objfun.evaluate(x)
        g = g_x + g_reg

        return fx, g

    def print_and_plot(self, x, ):

        self.iteration += 1

        x_single, x_pair = self.objfun.linear_to_structured(x)

        log_metrics={}
        log_metrics['||v+w||'] = np.sqrt(np.sum(x * x))
        log_metrics['||v||'] = np.sqrt(np.sum(x_single * x_single))
        log_metrics['||w||'] = np.sqrt(np.sum(x_pair * x_pair))
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