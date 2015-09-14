import numpy as np
import sys


class ObjectiveFunction(object):
    def __init__(self):
        self.compare_raw = None
        self.linear_to_structured = None
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
        if self.compare_raw:
            print("    iter  ls           fx          |x|          |g|         step  dist_single    dist_pair")
        else:
            print("    iter  ls           fx          |x|          |g|         step")

    def progress(self, x, g, fx, n_iter, n_ls, step):
        xnorm = np.sum(x * x)
        gnorm = np.sum(g * g)

        if self.compare_raw:
            ox_single, ox_pair = self.linear_to_structured(x)
            dx_single = ox_single - self.compare_raw.x_single
            dx_pair = ox_pair - self.compare_raw.x_pair

            dist_single = np.sqrt(np.sum(dx_single ** 2))
            dist_pair = np.sqrt(np.sum(dx_pair ** 2))

            print("{n_iter:8d} {n_ls:3d} {fx:12g} {xnorm:12g} {gnorm:12g} {step: 12g} {dist_single: 12g} {dist_pair: 12g}".format(n_iter=n_iter, n_ls=n_ls, fx=fx, xnorm=xnorm, gnorm=gnorm, step=step, dist_single=dist_single, dist_pair=dist_pair))

        else:
            print("{n_iter:8d} {n_ls:3d} {fx:12g} {xnorm:12g} {gnorm:12g} {step: 8g}".format(n_iter=n_iter, n_ls=n_ls, fx=fx, xnorm=xnorm, gnorm=gnorm, step=step))

        sys.stdout.flush()
