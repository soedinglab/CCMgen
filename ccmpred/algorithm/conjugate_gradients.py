import numpy as np
import ccmpred.logo
import sys


class conjugateGradient():
    """Optimize objective function usign conjugate gradients"""

    def __init__(self, maxiter=100, ftol=1e-4, max_linesearch=5, alpha_mul=0.5, wolfe=0.2, epsilon=1e-3, convergence_prev=5):
        self.maxiter = maxiter
        self.ftol = ftol
        self.max_linesearch = max_linesearch
        self.alpha_mul = alpha_mul
        self.wolfe = wolfe
        self.epsilon = epsilon
        self.convergence_prev = convergence_prev

    def __repr__(self):
        return "conjugate gradient optimization (epsilon={0} convergence_prev={1} maxiter={2} " \
               "ftol={3} max_linesearch={4} alpha_mul={5} wolfe={6})".format(
            self.epsilon, self.convergence_prev, self.maxiter, self.ftol, self.max_linesearch, self.alpha_mul, self.wolfe)


    def begin_progress(self):

        header_tokens = [('iter', 8), ('ls', 3), ('fx', 12), ('|x|', 12), ('|g|', 12)]
        header_tokens += [('|x_single|', 12), ('|x_pair|', 12), ('|g_single|', 12), ('|g_pair|', 12)]
        header_tokens += [('step', 12)]


        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[1;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, xnorm, x_single, x_pair, gnorm, g_single, g_pair, fx, n_iter, n_ls, step):

        xnorm_single = np.sum(x_single * x_single)
        xnorm_pair = np.sum(x_pair *x_pair )

        gnorm_single = np.sum(g_single * g_single)
        gnorm_pair = np.sum(g_pair * g_pair)

        data_tokens = [(n_iter, '8d'), (n_ls, '3d'), (fx, '12g'), (xnorm, '12g'), (gnorm, '12g')]
        data_tokens += [(xnorm_single, '12g'), (xnorm_pair, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g')]
        data_tokens += [(step, '12g')]

        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))


        sys.stdout.flush()


    def minimize(self, objfun, x):

        #objfun.begin_progress()
        self.begin_progress()

        fx, g = objfun.evaluate(x)
        gnorm = np.sum(g * g)
        xnorm = np.sum(x * x)
        x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
        g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
        #objfun.progress(x, g, fx, iteration, n_linesearch, alpha)
        self.progress(xnorm, x_single, x_pair, gnorm, g_single, g_pair, fx, 0, 0, 0)

        gprevnorm = None
        alpha_prev = None
        dg_prev = None
        s = None

        ret = {
            "message": "Unknown",
            "code": -9999
        }

        if gnorm / xnorm < self.epsilon:
            ret['message'] = "Already minimized!"
            ret['code'] = 1
            return fx, x, ret

        lastfx = []

        alpha = 1 / np.sqrt(gnorm)
        iteration = 0
        while True:
            if iteration >= self.maxiter:
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

            gprevnorm = gnorm
            gnorm = np.sum(g * g)
            xnorm = np.sum(x * x)
            alpha_prev = alpha
            dg_prev = dg

            # convergence check
            if len(lastfx) >= self.convergence_prev:
                check_fx = lastfx[-self.convergence_prev]
                #print("check_fx: {0} (check_fx - fx) / check_fx: {1}".format(check_fx, (check_fx - fx) / check_fx))
                if (check_fx - fx) / check_fx < self.epsilon:
                    ret['message'] = 'Success!'
                    ret['code'] = 0
                    break

            lastfx.append(fx)

            iteration += 1

            #objfun.progress(x, g, fx, iteration, n_linesearch, alpha)
            x_single, x_pair = objfun.linear_to_structured(x, objfun.ncol)
            g_single, g_pair = objfun.linear_to_structured(g, objfun.ncol)
            self.progress(xnorm, x_single, x_pair, gnorm, g_single, g_pair, fx, iteration, n_linesearch, alpha)

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

