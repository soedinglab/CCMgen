import numpy as np

import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.pll.cext


class PseudoLikelihood(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, lambda_single, lambda_pair, clustering_threshold):

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

        self.weights = calculate_weights(msa, clustering_threshold)
        self.v_centering = calculate_centering(msa, self.weights)

        # memory allocation for intermediate variables
        self.g = np.empty((self.nsingle_padded + self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float32'))
        self.g2 = np.empty((self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float32'))

    @classmethod
    def init_from_default(cls, msa, lambda_single=1, lambda_pair=lambda msa: msa.shape[1] * 0.2, clustering_threshold=0.8):
        res = cls(msa, lambda_single, lambda_pair, clustering_threshold)

        x = np.zeros((res.nvar, ), dtype=np.dtype('float32'))
        x[:res.nsingle] = res.v_centering

        return x, res

    def evaluate(self, x):
        return ccmpred.objfun.pll.cext.evaluate(x, self.g, self.g2, self.v_centering, self.weights, self.msa, self.lambda_single, self.lambda_pair)


def calculate_weights(msa, cutoff=0.8):
    if cutoff >= 1:
        return np.ones((msa.shape[0],), dtype="float32")

    ncol = msa.shape[1]

    # calculate pairwise sequence identity between all alignments
    ids = np.sum(msa[:, np.newaxis, :] == msa[np.newaxis, :, :], axis=2)

    # calculate number of cluster members at identity cutoff
    n_cluster = np.sum(ids > cutoff * ncol, axis=0)

    return (1 / n_cluster.astype("float32"))


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
    return v_center.reshape((ncol * 20,))
