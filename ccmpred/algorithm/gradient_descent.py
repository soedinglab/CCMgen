import numpy as np
import ccmpred.logo
import sys
from collections import deque
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot

class gradientDescent():
    """Optimize objective function using gradient descent"""

    def __init__(self, maxit=100, alpha0=5e-3, alpha_decay=10, epsilon=1e-5, convergence_prev=5, early_stopping=False):
        self.maxit = maxit
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay

        self.g_sign = deque([])
        self.g_hist = deque([])
        self.x_hist = deque([])
        self.lastg = np.array([])
        self.neg_g_sign = 0

        #for plotting
        self.optimization_log={}

        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev

    def __repr__(self):
        return "Gradient descent optimization (alpha0={0} alpha_decay={1}) \n" \
               "convergence criteria: maxiter={2} early_stopping={3} epsilon={4} prev={5}".format(
            self.alpha0, self.alpha_decay, self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)


    def begin_progress(self):

        header_tokens = [('iter', 8),
                         ('|x|', 12), ('|x_single|', 12), ('|x_pair|', 12),
                         ('|g|', 12), ('|g_single|', 12), ('|g_pair|', 12),
                         ('xnorm_diff', 12), ('gnorm_diff', 12),
                         ('sign_g_t10', 12), ('sign_g_t8', 12),
                         ('step', 12)
                         ]


        self.optimization_log['||x||'] = []
        self.optimization_log['||x_single||'] = []
        self.optimization_log['||x_pair||'] = []
        self.optimization_log['||g||'] = []
        self.optimization_log['||g_single||'] = []
        self.optimization_log['||g_pair||'] = []
        self.optimization_log['step'] = []


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, n_iter, xnorm_single, xnorm_pair, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8, step, plotfile):


        xnorm = xnorm_single+xnorm_pair
        gnorm = gnorm_single+gnorm_pair


        data_tokens = [(n_iter, '8d'),
                       (xnorm, '12g'), (xnorm_single, '12g'), (xnorm_pair, '12g'),
                       (gnorm, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g'),
                       (xnorm_diff, '12g'), (gnorm_diff, '12g'),
                       (sign_g_t10, '12g'), (sign_g_t8, '12g'),
                       (step, '12g')
                       ]


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

        self.begin_progress()

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }

        fx = -1


        for i in range(self.maxit):

            fx, g = objfun.evaluate(x)

            #new step size
            alpha = self.alpha0 / (1 + i / self.alpha_decay)

            # ========================================================================================
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)


            xnorm_single = np.sum(x_single * x_single)
            xnorm_pair = np.sum(x_pair * x_pair)
            xnorm = xnorm_single + xnorm_pair

            gnorm_single = np.sum(g_single * g_single)
            gnorm_pair = np.sum(g_pair * g_pair)
            gnorm = gnorm_single + gnorm_pair

            # possible stopping criteria
            if len(self.lastg) != 0:
                self.g_sign.append(np.mean(np.sign(self.lastg * g.copy())))
            self.lastg = g.copy()

            self.x_hist.append(xnorm)
            self.g_hist.append(gnorm)

            xnorm_diff = 1
            gnorm_diff = 1
            sign_g_t10, sign_g_t8, sign_g_t7, sign_g_t6 = [0, 0, 0, 0]
            if len(self.g_sign) > 10:
                self.g_sign.popleft()
                sign_g_t10 = np.sum(self.g_sign) / 10.0
                sign_g_t8 = np.sum(list(self.g_sign)[2:]) / 8.0


                self.x_hist.popleft()
                xnorm_prev = self.x_hist[-self.convergence_prev-1]
                xnorm_diff = np.abs((xnorm_prev - xnorm)) / xnorm_prev

                self.g_hist.popleft()
                gnorm_prev = self.g_hist[-self.convergence_prev-1]
                gnorm_diff = np.abs((gnorm_prev - gnorm)) / gnorm_prev

            if sign_g_t8 < 0:
                self.neg_g_sign += 1
            else:
                self.neg_g_sign = 0

            # ====================================================================================

            #print out progress
            self.progress(i, xnorm_single, xnorm_pair, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8, alpha, plotfile)

            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:
                    ret = {
                        "code": 0,
                        "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                    }
                    return fx, x, ret


            #update parameters
            x -= alpha * g




        return -1, x, ret
