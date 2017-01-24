import numpy as np



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


    def minimize(self, objfun, x):
        objfun.begin_progress()

        fx, g = objfun.evaluate(x)
        gnorm = np.sum(g * g)
        xnorm = np.sum(x * x)

        objfun.progress(x, g, fx, 0, 1, 0)

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
                if (check_fx - fx) / check_fx < self.epsilon:
                    ret['message'] = 'Success!'
                    ret['code'] = 0
                    break

            lastfx.append(fx)

            iteration += 1
            objfun.progress(x, g, fx, iteration, n_linesearch, alpha)

        return fx, x, ret

    def linesearch(self, x0, fx, g, objfun, s, alpha):
        dg_init = np.sum(g * g)
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
                dg = np.sum(s * g)
                if dg < self.wolfe * dg_init:
                    fx = fx_step
                    return n_linesearch, fx, alpha, g, x

            alpha *= self.alpha_mul

