import numpy as np
import ccmpred.logo
import sys
from collections import deque
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot

class Adam():
    """
    Optimize objective function using Adam

    This is an implementation of the Adam algorithm:
        Kingma, D. P., & Ba, J. L. (2015)
        Adam: a Method for Stochastic Optimization. International Conference on Learning Representations

    Adaptive Moment Estimation (Adam) computes adaptive learning rates for each parameter.
    In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop,
    Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum

    """

    def __init__(self, maxit=100, learning_rate=1e-3, momentum_estimate1=0.9, momentum_estimate2=0.999, noise=1e-7, epsilon=1e-5, convergence_prev=5, early_stopping=False, decay=False):
        self.maxit = maxit
        self.learning_rate = learning_rate
        self.momentum_estimate1 = momentum_estimate1
        self.momentum_estimate2 = momentum_estimate2
        self.noise = noise
        self.decay=decay

        self.g_hist = deque([])
        self.g_sign = deque([])
        self.x_hist = deque([])

        #for plotting
        self.optimization_log={}

        self.lastg = np.array([])
        self.neg_g_sign = 0

        self.early_stopping = early_stopping
        self.it_succesfull_stop_condition=-1
        self.epsilon = epsilon
        self.convergence_prev=convergence_prev
        self.decrease = 5.0

    def __repr__(self):
        return "Adam stochastic optimization (decay={0} learning_rate={1} momentum_estimate1={2} momentum_estimate2={3} noise={4}) \n" \
               "convergence criteria: maxit={5} early_stopping={6} epsilon={7} prev={8}".format(
            self.decay, self.learning_rate, self.momentum_estimate1, self.momentum_estimate2, self.noise,
            self.maxit, self.early_stopping, self.epsilon, self.convergence_prev)

    def begin_process(self):

        header_tokens = [('iter', 8),
                         ('|x|', 12), ('|x_single|', 12), ('|x_pair|', 12),
                         ('|g|', 12), ('|g_single|', 12), ('|g_pair|', 12),
                         #('|first moment|', 12), ('|second moment|', 12),
                         ('xnorm_diff', 12), ('max_g', 12), ('gnorm_diff', 12),
                         ('sign_g_t10', 12), ('sign_g_t8', 12), ('alpha', 12)
                         ]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))


        self.optimization_log['||x||'] = []
        self.optimization_log['||x_single||'] = []
        self.optimization_log['||x_pair||'] = []
        self.optimization_log['||g||'] = []
        self.optimization_log['||g_single||'] = []
        self.optimization_log['||g_pair||'] = []
        self.optimization_log['max_g'] = []
        self.optimization_log['alpha'] = []

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, n_iter, xnorm_single, xnorm_pair, g, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8, alpha, plotfile ):

        xnorm = xnorm_single + xnorm_pair
        gnorm = gnorm_single + gnorm_pair
        max_g = np.max(np.abs(g))

        if plotfile is not None:
            self.optimization_log['||x||'].append(xnorm)
            self.optimization_log['||x_single||'].append(xnorm_single)
            self.optimization_log['||x_pair||'].append(xnorm_pair)
            self.optimization_log['||g||'].append(gnorm)
            self.optimization_log['||g_single||'].append(gnorm_single)
            self.optimization_log['||g_pair||'].append(gnorm_pair)
            self.optimization_log['max_g'].append(max_g)
            self.optimization_log['alpha'].append(alpha)
            self.plot_progress(plotfile)


        data_tokens = [(n_iter, '8d'),
                       (xnorm, '12g'), (xnorm_single, '12g'), (xnorm_pair, '12g'),
                       (gnorm, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g'),
                       (xnorm_diff, '12g'), (max_g, '12g'), (gnorm_diff, '12g'),
                       (sign_g_t10, '12g'), (sign_g_t8, '12g'), (alpha, '12g')
                       ]


        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))


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
                    visible="legendonly",
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

        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)

        self.begin_process()

        ret = {
            "code": 2,
            "message": "Reached maximum number of iterations"
        }
        fx = -1


        for i in range(self.maxit):


            fx, g = objfun.evaluate(x)

            #update moment vectors
            first_moment    = self.momentum_estimate1 * first_moment + (1-self.momentum_estimate1) * (g)
            second_moment   = self.momentum_estimate2 * second_moment + (1-self.momentum_estimate2) * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment / (1 - np.power(self.momentum_estimate1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.momentum_estimate2, i+1))

            first_moment_corrected_single, first_moment_corrected_pair = objfun.linear_to_structured(first_moment_corrected, objfun.ncol)
            second_moment_corrected_single, second_moment_corrected_pair = objfun.linear_to_structured(second_moment_corrected, objfun.ncol)

            # ========================================================================================
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)


            #print x_single[9, :], np.sum(x_single[9, :])

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


            #update learning rate
            alpha  = self.learning_rate
            if(self.decay):
                if (self.early_stopping and self.it_succesfull_stop_condition > -1) or (not self.early_stopping):
                    #alpha /= np.sqrt(i  - self.it_succesfull_stop_condition)
                    alpha /= (1 + (i - self.it_succesfull_stop_condition) / self.decrease)

            #print out progress
            self.progress(i + 1, xnorm_single, xnorm_pair, g, gnorm_single, gnorm_pair, xnorm_diff, gnorm_diff, sign_g_t10, sign_g_t8, alpha, plotfile)


            #stop condition
            if self.early_stopping:
                if xnorm_diff < self.epsilon:

                    if not self.decay and objfun.gibbs_steps == 1 and not objfun.persistent:
                        objfun.gibbs_steps = 5
                        self.epsilon *= 1e-1        #decrease convergence criterion
                        print("Use 5 Gibss sampling steps. Set learning rate to {0}. Decrease epsilon to {1}".format(self.learning_rate, self.epsilon))
                    elif not self.decay and objfun.gibbs_steps == 5:
                        self.decay = True
                        self.epsilon *= 1e-1        #decrease convergence criterion
                        self.it_succesfull_stop_condition=i
                        print("Turn on decaying learning rate. Use 5 Gibss sampling steps. Decrease epsilon to {0}".format(self.epsilon))
                    elif self.decay and self.decrease > 1:
                        self.decrease -= 1.0
                        print("Decrease decay. Set deacay to {0}".format(self.decrease))
                    elif self.decay and not objfun.persistent:
                        objfun.persistent=True
                        objfun.gibbs_steps = 1
                        #self.epsilon *= 1e-1        #decrease convergence criterion
                        self.learning_rate = 1e-3
                        self.decrease = 10.0
                        self.it_succesfull_stop_condition=i
                        print("Turn on persistent CD. Decrease epsilon to {0}. Set learnign rate to {1}. Set decrease tp {2}.".format(self.epsilon, self.learning_rate, self.decrease))
                    else:
                        ret = {
                            "code": 0,
                            "message": "Stopping condition (xnorm diff < {0}) successfull.".format(self.epsilon)
                        }
                        return fx, x, ret


                # if self.neg_g_sign > 10:
                #     ret = {
                #         "code": 0,
                #         "message": "Stopping condition (change of gradient direction) successfull."
                #     }
                #
                #     return fx, x, ret


            #update parameters


            step_single = first_moment_corrected_single / ( np.sqrt(second_moment_corrected_single.max(1))[:, np.newaxis] + self.noise)
            x_single -= alpha * step_single

            step_pair = first_moment_corrected_pair / ( np.sqrt(second_moment_corrected_pair.max(3).max(2))[:, :, np.newaxis, np.newaxis] + self.noise)
            x_pair -= alpha * step_pair
            x=objfun.structured_to_linear(x_single, x_pair)

            #x -= alpha * first_moment_corrected / ( np.sqrt(second_moment_corrected) + self.noise)


        return fx, x, ret
