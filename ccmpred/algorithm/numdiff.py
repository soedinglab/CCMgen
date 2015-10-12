import numpy as np
import random


random.seed(42)


def numdiff(objfun, x, epsilon=1e-8):

    _, g0 = objfun.evaluate(x)

    x0_single, x0_pair = objfun.linear_to_structured(x)
    g0_single, g0_pair = objfun.linear_to_structured(g0)
    ncol = x0_single.shape[0]

    print("Pos                             Symbolic             Numeric               Delta")
    while True:

        if random.random() <= 0.2:
            i = random.randint(0, ncol - 1)
            a = random.randint(0, 19)

            xA = np.copy(x0_single)
            xB = np.copy(x0_single)
            xA[i, a] -= epsilon
            xB[i, a] += epsilon

            fxA, _ = objfun.evaluate(objfun.structured_to_linear(xA, x0_pair))
            fxB, _ = objfun.evaluate(objfun.structured_to_linear(xB, x0_pair))

            symdiff = g0_single[i, a]
            numdiff = (fxB - fxA) / (2 * epsilon)

            posstr = "v[{i:3d}, {a:2d}]".format(i=i, a=a)

        else:
            i = random.randint(0, ncol - 1)
            j = random.randint(0, ncol - 1)
            a = random.randint(0, 20)
            b = random.randint(0, 20)

            xA = np.copy(x0_pair)
            xB = np.copy(x0_pair)
            xA[i, j, a, b] -= epsilon
            xA[j, i, b, a] -= epsilon
            xB[i, j, a, b] += epsilon
            xB[j, i, b, a] += epsilon

            fxA, _ = objfun.evaluate(objfun.structured_to_linear(x0_single, xA))
            fxB, _ = objfun.evaluate(objfun.structured_to_linear(x0_single, xB))

            symdiff = g0_pair[i, j, a, b]
            symdiff2 = g0_pair[j, i, b, a]
            print("                       {0: .10e}".format(symdiff2))
            numdiff = (fxB - fxA) / (2 * epsilon)

            posstr = "w[{i:3d}, {j:3d}, {a:2d}, {b:2d}]".format(i=i, j=j, a=a, b=b)

        print("{posstr:20s}   {symdiff: .10e}   {numdiff: .10e}   {delta: .10e}".format(posstr=posstr, symdiff=symdiff, numdiff=numdiff, delta=symdiff - numdiff))
