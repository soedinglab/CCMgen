import numpy as np

import ccmpred.raw
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd.cext


class ContrastiveDivergence(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, weights, lambda_single, lambda_pair, n_samples):

        if hasattr(lambda_single, '__call__'):
            lambda_single = lambda_single(msa)

        if hasattr(lambda_pair, '__call__'):
            lambda_pair = lambda_pair(msa)

        self.msa = msa
        self.weights = weights
        self.lambda_single = lambda_single
        self.lambda_pair = lambda_pair

        self.nrow, self.ncol = msa.shape
        self.nsingle = self.ncol * 20
        self.nvar = self.nsingle + self.ncol * self.ncol * 21 * 21
        self.n_samples = n_samples

        # get constant alignment counts
        self.msa_counts_single = ccmpred.counts.single_counts(msa)
        self.msa_counts_pair = ccmpred.counts.pair_counts(msa)

        # compute number of ungapped rows per column, pair of columns
        self.nrow_nogaps_single = msa.shape[0] - self.msa_counts_single[:, 20]
        self.nrow_nogaps_pair = msa.shape[0] - (
            np.sum(self.msa_counts_pair[:, :, 20, :], axis=2) +
            np.sum(self.msa_counts_pair[:, :, :, 20], axis=2) -
            self.msa_counts_pair[:, :, 20, 20]
        )

        # reset gap counts
        self.msa_counts_single[:, 20] = 0
        self.msa_counts_pair[:, :, :, 20] = 0
        self.msa_counts_pair[:, :, 20, :] = 0

        # pick an initial sample alignment
        self.msa_sampled = np.empty((n_samples, msa.shape[1]), dtype="uint8")
        self.msa_sampled[:] = msa[np.random.choice(range(msa.shape[0]), size=self.n_samples, replace=True), :]

        # remove gaps from sample alignment
        self.msa_sampled = ccmpred.objfun.cd.cext.remove_gaps(self.msa_sampled, self.msa_counts_single[:, :20].reshape(-1))

        # TODO weight sequences?
        # TODO centered regularization?

    @classmethod
    def init_from_default(cls, msa, weights, lambda_single=10, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2, n_samples=1000):
        res = cls(msa, weights, lambda_single, lambda_pair, n_samples)
        x = np.zeros((res.nvar, ), dtype=np.dtype('float64'))

        return x, res

    @classmethod
    def init_from_raw(cls, msa, weights, raw, lambda_single=10, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2, n_samples=1000):
        res = cls(msa, weights, lambda_single, lambda_pair, n_samples)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        x_single = raw.x_single
        x_pair = np.transpose(raw.x_pair, (0, 2, 1, 3))

        x = np.hstack((x_single.reshape((-1,)), x_pair.reshape((-1),)))

        return x, res

    def finalize(self, x):
        x_single = x[:self.nsingle].reshape((self.ncol, 20))
        x_pair = np.transpose(x[self.nsingle:].reshape((self.ncol, 21, self.ncol, 21)), (0, 2, 1, 3))

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, {})

    def evaluate(self, x):

        self.msa_sampled = ccmpred.objfun.cd.cext.sample_sequences(self.msa_sampled, x)

        sample_counts_single = ccmpred.counts.single_counts(self.msa_sampled)
        sample_counts_pair = ccmpred.counts.pair_counts(self.msa_sampled)

        # renormalize to the number of non-gapped rows in the original sequence alignment
        sample_counts_single *= self.nrow_nogaps_single[:, np.newaxis] / self.n_samples
        sample_counts_pair *= self.nrow_nogaps_pair[:, :, np.newaxis, np.newaxis] / self.n_samples

        g_single = -self.msa_counts_single + sample_counts_single
        g_pair = -self.msa_counts_pair + sample_counts_pair

        # regularization
        x_single = x[:self.nsingle].reshape((self.ncol, 20))
        x_pair = x[self.nsingle:].reshape((self.ncol, 21, self.ncol, 21))
        x_pair = np.transpose(x_pair, (0, 2, 1, 3))

        g_single[:, :20] += 2 * self.lambda_single * x_single
        g_pair += 2 * self.lambda_pair * x_pair

        # set gradients for gap states to 0
        g_single[:, 20] = 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        # reorder dimensions for gradient
        g_pair = np.transpose(g_pair, (0, 2, 1, 3))

        return -1, np.hstack((g_single[:, :20].reshape(-1), g_pair.reshape(-1)))
