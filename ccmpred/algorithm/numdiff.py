import numpy as np
import random
import sys
import copy

random.seed(42)

class numDiff():
    """Debug Gradients with numerical differentiation"""

    def __init__(self, maxit=100, epsilon=1e-5):
        self.epsilon = epsilon
        self.maxit = maxit


    def minimize(self, objfun, x):
        """
        Compute analytical and numerical gradient for objfun

        :param objfun:
        :param x:
        :return:
        """

        x0_single, x0_pair = objfun.linear_to_structured(x)
        ncol = x0_single.shape[0]

        #non-zero couplings to also evaluate regularizer
        x0_pair = np.random.random((ncol, ncol, 21, 21))
        temp = copy.deepcopy(x0_pair)
        temp = np.swapaxes(temp,0,1)
        temp = np.swapaxes(temp,2,3)
        x0_pair = x0_pair + temp    #to obtain symmetry

        x = objfun.structured_to_linear(x0_single, x0_pair)

        _, g0, g_reg = objfun.evaluate(x)
        g = g0 + g_reg

        g_single, g_pair = objfun.linear_to_structured(g)





        print("Comparing analytical gradient to numerical gradient with stepsize 2 * {0}".format(self.epsilon))
        print("Pos                                    x                 g            DeltaG")

        iteration = 0
        while True:
            iteration += 1
            if iteration >= self.maxit:
                break

            if random.random() <= 0.2:
                #check single potentials
                i = random.randint(0, ncol - 1)
                a = random.randint(0, 19)

                xA = np.copy(x0_single)
                xB = np.copy(x0_single)
                xA[i, a] -= self.epsilon
                xB[i, a] += self.epsilon

                fxA, _, _ = objfun.evaluate(objfun.structured_to_linear(xA, x0_pair))
                fxB, _, _ = objfun.evaluate(objfun.structured_to_linear(xB, x0_pair))

                symdiff = g_single[i, a]
                symdiff2 = None
                numdiff = (fxB - fxA) / (2 * self.epsilon)

                xval = x0_single[i, a]
                symmval = None
                posstr = "v[{i:3d}, {a:2d}]".format(i=i, a=a)
                posstr2 = None

            else:
                #check pair potentials
                i = random.randint(0, ncol - 1)
                j = random.randint(0, ncol - 1)
                a = random.randint(0, 20)
                b = random.randint(0, 20)

                xA = np.copy(x0_pair)
                xB = np.copy(x0_pair)
                xA[i, j, a, b] -= self.epsilon
                xA[j, i, b, a] -= self.epsilon
                xB[i, j, a, b] += self.epsilon
                xB[j, i, b, a] += self.epsilon

                #numerical differentiation for value at x+eps and x-eps
                fxA, _, _ = objfun.evaluate(objfun.structured_to_linear(x0_single, xA))
                fxB, _, _ = objfun.evaluate(objfun.structured_to_linear(x0_single, xB))
                numdiff = (fxB - fxA) / (2 * self.epsilon)

                #actual value (and its symmetric counterpart)
                xval = x0_pair[i, j, a, b]
                symmval = x0_pair[j, i, b, a]
                #analytical gradient
                symdiff = g_pair[i, j, a, b]
                symdiff2 = g_pair[j, i, b, a]

                posstr = "w[{i:3d}, {j:3d}, {a:2d}, {b:2d}]".format(i=i, j=j, a=a, b=b)
                posstr2 = "w[{j:3d}, {i:3d}, {b:2d}, {a:2d}]".format(i=i, j=j, a=a, b=b)

            print("{posstr:20s}   {xval: .10e} {symdiff: .10e}".format(posstr=posstr, xval=xval, symdiff=symdiff,))

            #print symmetrical value and gradient for pair emissions
            if symdiff2 is not None and symmval is not None and posstr2 is not None:
                print("{posstr:20s}   {xval: .10e} {symdiff: .10e}".format(posstr=posstr2, xval=symmval, symdiff=symdiff2))

            print("gNumeric                                 {numdiff: .10e} {delta: .10e}".format(posstr=posstr, xval=xval, numdiff=numdiff, delta=symdiff - numdiff))

            print("")

        sys.exit(0)
