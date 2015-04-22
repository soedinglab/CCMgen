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

        # init sample alignment
        self.msa_sampled = self.init_sample_alignment()

        # allocate centering - should be filled with init_* functions
        self.centering_x_single = np.zeros((self.ncol, 20), dtype=np.dtype('float64'))

        # TODO weight sequences?

    def init_sample_alignment(self):

        msa_sampled = np.empty((self.n_samples, self.msa.shape[1]), dtype="uint8")

        # make initial sample alignment from real alignment
        msa_sampled[:] = self.msa[np.random.choice(range(self.msa.shape[0]), size=self.n_samples, replace=True), :]

        # remove gaps from sample alignment
        colfreqs = self.msa_counts_single[:, :20]
        colfreqs /= np.sum(colfreqs, axis=1)[:, np.newaxis]

        return ccmpred.gaps.remove_gaps_probs(msa_sampled, colfreqs)

    @classmethod
    def init_from_raw(cls, msa, weights, raw, lambda_single=1e4, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2, n_samples=1000):
        res = cls(msa, weights, lambda_single, lambda_pair, n_samples)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        x_single = raw.x_single
        x_pair = np.transpose(raw.x_pair, (0, 2, 1, 3))
        x = np.hstack((x_single.reshape((-1,)), x_pair.reshape((-1),)))

        res.centering_x_single[:] = x_single

        return x, res

    def finalize(self, x):
        x_single = x[:self.nsingle].reshape((self.ncol, 20))
        x_pair = np.transpose(x[self.nsingle:].reshape((self.ncol, 21, self.ncol, 21)), (0, 2, 1, 3))

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, {})

    def sample_sequences(self, x):
        return ccmpred.objfun.cd.cext.sample_sequences(self.msa_sampled, x)

    def evaluate(self, x):

        self.msa_sampled = self.sample_sequences(x)

        sample_counts_single = ccmpred.counts.single_counts(self.msa_sampled)
        sample_counts_pair = ccmpred.counts.pair_counts(self.msa_sampled)

        # renormalize to the number of non-gapped rows in the original sequence alignment
        sample_counts_single *= self.nrow_nogaps_single[:, np.newaxis] / self.n_samples
        sample_counts_pair *= self.nrow_nogaps_pair[:, :, np.newaxis, np.newaxis] / self.n_samples

        g_single = sample_counts_single - self.msa_counts_single
        g_pair = sample_counts_pair - self.msa_counts_pair

        # regularization
        x_single = x[:self.nsingle].reshape((self.ncol, 20)) - self.centering_x_single
        x_pair = x[self.nsingle:].reshape((self.ncol, 21, self.ncol, 21))
        x_pair = np.transpose(x_pair, (0, 2, 1, 3))

        g_single[:, :20] += 2 * self.lambda_single * x_single
        g_pair += 2 * self.lambda_pair * x_pair

        # set gradients for gap states to 0
        g_single[:, 20] = 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        for i in range(self.ncol):
            g_pair[i, i, :, :] = 0

        # reorder dimensions for gradient
        g_pair = np.transpose(g_pair, (0, 2, 1, 3))

        return -1, np.hstack((g_single[:, :20].reshape(-1), g_pair.reshape(-1)))
