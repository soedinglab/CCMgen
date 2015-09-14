import numpy as np

import ccmpred.raw
import ccmpred.gaps
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd.cext


class ContrastiveDivergence(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, weights, regularization, n_samples):
        super(ContrastiveDivergence, self).__init__()

        self.msa = msa
        self.weights = weights
        self.regularization = regularization

        self.nrow, self.ncol = msa.shape
        self.nsingle = self.ncol * 20
        self.nvar = self.nsingle + self.ncol * self.ncol * 21 * 21
        self.n_samples = n_samples

        # get constant alignment counts
        self.msa_counts_single = ccmpred.counts.single_counts(msa)
        self.msa_counts_pair = ccmpred.counts.pair_counts(msa)

        # reset gap counts
        self.msa_counts_single[:, 20] = 0
        self.msa_counts_pair[:, :, :, 20] = 0
        self.msa_counts_pair[:, :, 20, :] = 0

        # init sample alignment
        self.msa_sampled = self.init_sample_alignment()

        self.linear_to_structured = lambda x: linear_to_structured(x, self.ncol)

    def init_sample_alignment(self):
        return self.msa.copy()

    @classmethod
    def init_from_raw(cls, msa, weights, raw, regularization):
        n_samples = msa.shape[0]

        res = cls(msa, weights, regularization, n_samples)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        x = structured_to_linear(raw.x_single, raw.x_pair)
        return x, res

    def finalize(self, x):
        x_single, x_pair = linear_to_structured(x, self.ncol, clip=True)

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, {})

    def sample_sequences(self, x):
        return ccmpred.objfun.cd.cext.sample_sequences(self.msa_sampled, x)

    def evaluate(self, x):

        self.msa_sampled = self.sample_sequences(x)

        sample_counts_single = ccmpred.counts.single_counts(self.msa_sampled)
        sample_counts_pair = ccmpred.counts.pair_counts(self.msa_sampled)

        g_single = sample_counts_single - self.msa_counts_single
        g_pair = sample_counts_pair - self.msa_counts_pair

        x_single, x_pair = linear_to_structured(x, self.ncol)
        _, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair)

        # set gradients for gap states to 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        for i in range(self.ncol):
            g_pair[i, i, :, :] = 0

        g = structured_to_linear(g_single[:, :20], g_pair)
        return -1, g


def linear_to_structured(x, ncol, clip=False):
    """Convert linear vector of variables into multidimensional arrays.

    in linear memory, memory order is v[j, a] and w[i, a, j, b] (dimensions Lx20 and Lx21xLx21)
    output will have  memory order of v[j, a] and w[i, j, a, b] (dimensions Lx20 and LxLx21x21)
    """

    nsingle = ncol * 20

    x_single = x[:nsingle].reshape((ncol, 20))
    x_pair = np.transpose(x[nsingle:].reshape((ncol, 21, ncol, 21)), (0, 2, 1, 3))

    if clip:
        x_pair = x_pair[:, :, :21, :21]

    return x_single, x_pair


def structured_to_linear(x_single, x_pair):
    """Convert structured variables into linear array

    with input arrays of memory order v[j, a] and w[i, j, a, b] (dimensions Lx20 and LxLx21x21)
    output will have  memory order of v[j, a] and w[i, a, j, b] (dimensions Lx20 and Lx21xLx21)
    """

    ncol = x_single.shape[0]
    nsingle = ncol * 20
    nvar = nsingle + ncol * ncol * 21 * 21

    out_x_pair = np.zeros((ncol, 21, ncol, 21), dtype='float64')
    out_x_pair[:, :21, :, :21] = np.transpose(x_pair[:, :, :21, :21], (0, 2, 1, 3))

    x = np.zeros((nvar, ), dtype='float64')
    x[:nsingle] = x_single.reshape(-1)
    x[nsingle:] = out_x_pair.reshape(-1)

    return x
