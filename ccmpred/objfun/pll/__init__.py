import numpy as np

import ccmpred.raw
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.pll.cext


class PseudoLikelihood(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, weights, lambda_single, lambda_pair):

        if hasattr(lambda_single, '__call__'):
            lambda_single = lambda_single(msa)

        if hasattr(lambda_pair, '__call__'):
            lambda_pair = lambda_pair(msa)

        self.msa = msa
        self.lambda_single = lambda_single
        self.lambda_pair = lambda_pair

        self.nrow, self.ncol = msa.shape
        self.nsingle = self.ncol * 20
        self.nsingle_padded = self.nsingle + 32 - (self.nsingle % 32)
        self.nvar = self.nsingle_padded + self.ncol * self.ncol * 21 * 32

        self.weights = weights
        self.v_centering = calculate_centering(msa, self.weights)

        # memory allocation for intermediate variables
        self.g = np.empty((self.nsingle_padded + self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))
        self.g2 = np.empty((self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))

    @classmethod
    def init_from_default(cls, msa, weights, lambda_single=10, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2):

        res = cls(msa, weights, lambda_single, lambda_pair)

        x = np.zeros((res.nvar, ), dtype=np.dtype('float64'))
        x[:res.nsingle] = res.v_centering

        return x, res

    @classmethod
    def init_from_raw(cls, msa, weights, raw, lambda_single=10, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2):
        res = cls(msa, weights, lambda_single, lambda_pair)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        x_single = raw.x_single

        x_pair = np.zeros((21, res.ncol, 32, res.ncol))
        x_pair[:, :, :21, :] = np.transpose(raw.x_pair, (3, 1, 2, 0))

        x = np.zeros((res.nvar, ), dtype=np.dtype('float64'))

        x[:res.nsingle] = x_single.reshape(-1)
        x[res.nsingle_padded:] = x_pair.reshape(-1)

        res.v_centering[:] = x_single.reshape(-1)

        return x, res

    def finalize(self, x):
        x_single = x[:self.nsingle].reshape((self.ncol, 20))
        x_pair = np.transpose(x[self.nsingle_padded:].reshape((21, self.ncol, 32, self.ncol))[:, :, :21, :], (3, 1, 2, 0))

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, {})

    def evaluate(self, x):
        return ccmpred.objfun.pll.cext.evaluate(x, self.g, self.g2, self.v_centering, self.weights, self.msa, self.lambda_single, self.lambda_pair)


def calculate_centering(msa, weights, tau=0.1):
    nrow, ncol = msa.shape
    wsum = np.sum(weights)

    single_counts = ccmpred.counts.single_counts(msa, weights)

    aa_global_frac = np.sum(single_counts, axis=0) / (ncol * wsum)

    aafrac = single_counts / (wsum - single_counts[:, 20])[:, np.newaxis]
    aafrac[:, 20] = 0

    aafrac_pseudo = (1 - tau) * aafrac[:, :20] + tau * aa_global_frac[np.newaxis, :20]
    aafrac_logsum = np.sum(np.log(aafrac_pseudo), axis=1)

    v_center = np.log(aafrac_pseudo) - aafrac_logsum[:, np.newaxis] / 20
    return v_center.T.reshape((ncol * 20,))
