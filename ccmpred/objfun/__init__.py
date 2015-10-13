import numpy as np
import sys
import ccmpred.raw
import ccmpred.logo


class ObjectiveFunction(object):
    def __init__(self):
        self.compare_raw = None
        self.trajectory_file = None
        self.linear_to_structured = None

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

        header_tokens = [('iter', 8), ('ls', 3), ('fx', 12), ('|x|', 12), ('|g|', 12)]

        if self.linear_to_structured:
            header_tokens += [('|x_single|', 12), ('|x_pair|', 12), ('|g_single|', 12), ('|g_pair|', 12)]

        header_tokens += [('step', 12)]

        if self.compare_raw:
            header_tokens += [('dist_single', 12), ('dist_pair', 12)]

        headerline = (" ".join("{0:>{1}s}".format(ht, hw) for ht, hw in header_tokens))

        if ccmpred.logo.is_tty:
            print("\x1b[1;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def progress(self, x, g, fx, n_iter, n_ls, step):
        xnorm = np.sum(x * x)
        gnorm = np.sum(g * g)

        data_tokens = [(n_iter, '8d'), (n_ls, '3d'), (fx, '12g'), (xnorm, '12g'), (gnorm, '12g')]

        if self.linear_to_structured:
            ox_single, ox_pair = self.linear_to_structured(x)
            xnorm_single = np.sum(ox_single ** 2)
            xnorm_pair = np.sum(ox_pair ** 2)

            og_single, og_pair = self.linear_to_structured(g)
            gnorm_single = np.sum(og_single ** 2)
            gnorm_pair = np.sum(og_pair ** 2)

            data_tokens += [(xnorm_single, '12g'), (xnorm_pair, '12g'), (gnorm_single, '12g'), (gnorm_pair, '12g')]

        data_tokens += [(step, '12g')]

        if self.compare_raw and self.linear_to_structured:
            ox_single, ox_pair = self.linear_to_structured(x)
            dx_single = ox_single - self.compare_raw.x_single
            dx_pair = ox_pair - self.compare_raw.x_pair

            dist_single = np.sqrt(np.sum(dx_single ** 2))
            dist_pair = np.sqrt(np.sum(dx_pair ** 2))

            data_tokens += [(dist_single, '12g'), (dist_pair, '12g')]

        print(" ".join("{0:{1}}".format(dt, df) for dt, df in data_tokens))

        if self.trajectory_file and self.linear_to_structured:
            x_single, x_pair = self.linear_to_structured(x)
            raw = ccmpred.raw.CCMRaw(x_single.shape[0], x_single, x_pair, None)
            ccmpred.raw.write_msgpack(self.trajectory_file.format(n_iter), raw)

        sys.stdout.flush()
