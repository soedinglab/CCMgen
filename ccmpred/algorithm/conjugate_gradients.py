import numpy as np
import ccmpred.logo
import sys
import ccmpred.monitor.progress as pr

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

        self.progress = pr.Progress(plotfile=None,
                                    fx=[], max_g=[], step=[], n_linesearch=[], rel_diff_fx=[])

    def __repr__(self):
        return "conjugate gradient optimization (ftol={0} max_linesearch={1} alpha_mul={2} wolfe={3}) \n" \
               "convergence criteria: maxit={4} epsilon={5} convergence_prev={6} ".format(
            self.ftol, self.max_linesearch, self.alpha_mul, self.wolfe, self.maxit, self.epsilon, self.convergence_prev)


    def begin_progress(self):

        header_tokens = [('iter', 8), ('ls', 3), ('fx', 12), ('|x|', 12), ('|g|', 12)]
        header_tokens += [('|x_single|', 12), ('|x_pair|', 12), ('|g_single|', 12), ('|g_pair|', 12)]
        header_tokens += [('step', 12)]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        self.optimization_log['||x||'] = []
        self.optimization_log['||x_single||'] = []
        self.optimization_log['||x_pair||'] = []
        self.optimization_log['||g||'] = []
        self.optimization_log['||g_single||'] = []
        self.optimization_log['||g_pair||'] = []
        self.optimization_log['step'] = []



        if ccmpred.logo.is_tty:
            print("\x1b[1;77m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, xnorm, x_single, x_pair, gnorm, g_single, g_pair, fx, n_iter, n_ls, step, plotfile):

        xnorm_single = np.sum(x_single * x_single)
        xnorm_pair = np.sum(x_pair *x_pair )

        gnorm_single = np.sum(g_single * g_single)
        gnorm_pair = np.sum(g_pair * g_pair)

        data_tokens = [(n_iter, '8d'), (n_ls, '3d'), (fx, '12g'), (xnorm, '12g'), (gnorm, '12g')]
        data_tokens += [(xnorm_single, '12g'), (xnorm_pair, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g')]
        data_tokens += [(step, '12g')]

        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))

        if plotfile is not None:
            self.optimization_log['||x||'].append(xnorm)
            self.optimization_log['||x_single||'].append(xnorm_single)
            self.optimization_log['||x_pair||'].append(xnorm_pair)
            self.optimization_log['||g||'].append(gnorm)
            self.optimization_log['||g_single||'].append(gnorm_single)
            self.optimization_log['||g_pair||'].append(gnorm_pair)
            self.optimization_log['step'].append(step)
            self.plot_progress(plotfile)



        sys.stdout.flush()


    def plot_progress(self, plotfile):


        title="Optimization Log <br>"
        title += self.__repr__().replace("\n", "<br>")


        data=[]
        for k,v in self.optimization_log.iteritems():
            data.append(
                go.Scatter(
                    x=range(1, len(v)+1),
                    y=v,
                    mode='lines',
                    name=k
                )
            )

        plot = {
            "data": data,
            "layout": go.Layout(
                title = title,
                xaxis1 = dict(
                    title="iteration",
                    exponentformat="e",
                    showexponent='All'
                ),
                yaxis1 = dict(
                    title="metric",
                    exponentformat="e",
                    showexponent='All'
                ),
            font = dict(size=18),
            )
        }

        plotly_plot(plot, filename=plotfile, auto_open=False)




    def minimize(self, objfun, x, plotfile):

        subtitle = " L={0} N={1} Neff={2}<br>".format(objfun.ncol, objfun.nrow, np.round(objfun.neff, decimals=3))
        subtitle += self.__repr__().replace("\n", "<br>")
        self.progress.plot_options(
            plotfile,
            ['fx', '||x||', '||x_single||', '||x_pair||', '||g||', '||g_single||', '||g_pair||', 'max_g', 'step', 'rel_diff_fx'],
            subtitle
        )
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
        self.progress.log_progress(0,
                                   xnorm_single, xnorm_pair,
                                   gnorm_single, gnorm_pair,
                                   fx=fx, max_g=max_g, step=0, n_linesearch=0, rel_diff_fx=0)


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

            # print out progress
            self.progress.log_progress(iteration,
                                       xnorm_single, xnorm_pair,
                                       gnorm_single, gnorm_pair,
                                       fx = fx, max_g=max_g,  step=alpha, n_linesearch=n_linesearch, rel_diff_fx=rel_diff_fx)



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

