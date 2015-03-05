def minimize(objfun, x, maxiter, step):

    objfun.begin_progress()

    for i in range(maxiter):
        fx, g = objfun.evaluate(x)
        x -= step * g
        objfun.progress(x, g, fx, i, 1, step)

    return fx, x
