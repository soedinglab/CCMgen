import numpy as np


def linesearch(x0, fx, g, objfun, s, alpha, ftol=1e-4, max_linesearch=5, alpha_mul=0.5, wolfe=0.2):
    dg_init = np.sum(g * g)
    dg_test = dg_init * ftol

    n_linesearch = 0
    fx_init = fx

    x = x0.copy()

    while True:
        if n_linesearch >= max_linesearch:
            print("MAX_LINESEARCH")
            return -1, fx, alpha, g, x

        n_linesearch += 1

        x = x0 + alpha * s

        fx_step, g = objfun.evaluate(x)

        # armijo condition
        if fx_step < fx_init + alpha * dg_test:
            dg = np.sum(s * g)
            if dg < wolfe * dg_init:
                fx = fx_step
                return n_linesearch, fx, alpha, g, x

        alpha *= alpha_mul


def minimize(objfun, x, maxiter, linesearch_fn=linesearch, epsilon=1e-3, convergence_prev=5):

    objfun.begin_progress()

    fx, g = objfun.evaluate(x)
    gnorm = np.sum(g * g)
    xnorm = np.sum(x * x)

    gprevnorm = None
    alpha_prev = None
    dg_prev = None
    s = None

    if gnorm / xnorm < epsilon:
        print("ALREADY MINIMIZED")
        return fx, x

    lastfx = []

    alpha = 1 / np.sqrt(gnorm)
    iteration = 0
    while True:
        if iteration >= maxiter:
            print("MAX ITER")
            break

        if iteration > 0:
            beta = gnorm / gprevnorm
            s = beta * s - g
            dg = np.sum(s * g)
            alpha = alpha_prev * dg_prev / dg

        else:
            s = -g
            dg = np.sum(s * g)

        n_linesearch, fx, alpha, g, x = linesearch_fn(x, fx, g, objfun, s, alpha)

        if n_linesearch < 0:
            print("NO LINESEARCH")
            # ret = n_linesearch
            break

        gprevnorm = gnorm
        gnorm = np.sum(g * g)
        xnorm = np.sum(x * x)
        alpha_prev = alpha
        dg_prev = dg

        # convergence check
        if len(lastfx) >= convergence_prev:
            check_fx = lastfx[-convergence_prev]
            if (check_fx - fx) / check_fx < epsilon:
                print("SUCCESS")
                break

        lastfx.append(fx)

        iteration += 1
        objfun.progress(x, g, fx, iteration, n_linesearch, alpha)

    return fx, x
