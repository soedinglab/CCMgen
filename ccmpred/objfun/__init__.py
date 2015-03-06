import numpy as np


class ObjectiveFunction(object):
    def __init__(self):
        pass

    @classmethod
    def init_from_default(cls, msa):
        raise NotImplemented()

    @classmethod
    def init_from_parfile(cls, msa, f):
        raise NotImplemented()

    def write_parfile(self, f, x):
        raise NotImplemented()

    def evaluate(self, x):
        raise NotImplemented()

    def begin_progress(self):
        print("    iter  ls           fx          |x|          |g|        step")

    def progress(self, x, g, fx, n_iter, n_ls, step):
        xnorm = np.sum(x * x)
        gnorm = np.sum(g * g)
        print("{n_iter:8d} {n_ls:3d} {fx:12g} {xnorm:12g} {gnorm:12g} {step:8g}".format(n_iter=n_iter, n_ls=n_ls, fx=fx, xnorm=xnorm, gnorm=gnorm, step=step))
