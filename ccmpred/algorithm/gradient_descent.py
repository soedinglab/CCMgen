def minimize(objfun, x, maxiter, step):

    objfun.begin_progress()

    for i in range(maxiter):
        fx, g = objfun.evaluate(x)
        alpha = step(i)
        x -= alpha * g
        objfun.progress(x, g, fx, i, 1, alpha)

    return fx, x
