import numpy as np

import ccmpred.raw
import ccmpred.regularization
import ccmpred.objfun
import ccmpred.objfun.pll.cext
import ccmpred.counts

class PseudoLikelihood(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, freqs, weights, regularization):
        super(PseudoLikelihood, self).__init__()

        self.msa = msa
        self.nrow, self.ncol = msa.shape

        #use msa counts with pseudo counts - numerically more stable?? but gradient does not fit ll fct!!
        #neff = np.sum(weights)
        #freqs_single, freqs_pair = freqs
        #msa_counts_single, msa_counts_pair = neff * freqs_single, neff * freqs_pair
        #use msa counts without pseudo counts
        msa_counts_single, msa_counts_pair = ccmpred.counts.both_counts(msa, weights)

        msa_counts_single[:, 20] = 0
        msa_counts_pair[:, :, 20, :] = 0
        msa_counts_pair[:, :, :, 20] = 0

        for i in range(self.ncol):
            msa_counts_pair[i, i, :, :] = 0

        self.g_init = structured_to_linear(msa_counts_single, 2 * msa_counts_pair)

        self.nsingle = self.ncol * 21
        self.nsingle_padded = self.nsingle + 32 - (self.nsingle % 32)
        self.nvar = self.nsingle_padded + self.ncol * self.ncol * 21 * 32

        self.weights = weights

        self.regularization = regularization

        # memory allocation for intermediate variables
        self.g = np.empty((self.nsingle_padded + self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))
        self.g2 = np.empty((self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))

        self.linear_to_structured = lambda x: linear_to_structured(x, self.ncol, clip=True)
        self.structured_to_linear = structured_to_linear

    @classmethod
    def init_from_default(cls, msa, freqs, weights, regularization):
        res = cls(msa, freqs, weights, regularization)

        if hasattr(regularization, "center_x_single"):
            ncol = msa.shape[1]
            x_pair = np.zeros((ncol, ncol, 21, 21), dtype="float64")
            x = structured_to_linear(regularization.center_x_single, x_pair)

        else:
            x = np.zeros((res.nvar, ), dtype=np.dtype('float64'))

        return x, res

    @classmethod
    def init_from_raw(cls, msa, freqs, weights, raw, regularization):
        res = cls(msa, freqs, weights, regularization)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        if raw.x_single.shape[1] == 20:
            temp = np.zeros((raw.ncol, 21))
            temp[:,:20] = raw.x_single
            raw.x_single = temp

        x = structured_to_linear(raw.x_single, raw.x_pair)

        return x, res

    def finalize(self, x, meta):
        x_single, x_pair = linear_to_structured(x, self.ncol, clip=True)
        return ccmpred.raw.CCMRaw(self.ncol, x_single[:, :20], x_pair, meta)

    def evaluate(self, x):

        #pointer to g == self.g
        fx, g = ccmpred.objfun.pll.cext.evaluate(x, self.g, self.g2, self.weights, self.msa)
        self.g -= self.g_init

        x_single, x_pair = linear_to_structured(x, self.ncol)
        g_single, g_pair = linear_to_structured(g, self.ncol)

        fx_reg, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair)

        g_reg = structured_to_linear(g_single_reg, g_pair_reg)
        fx += fx_reg
        g += g_reg

        return fx, g

    def __repr__(self):
        return "PLL "


def linear_to_structured(x, ncol, clip=False):
    """Convert linear vector of variables into multidimensional arrays.

    in linear memory, memory order is v[a, j] and w[b, j, a, i] (dimensions 21xL + padding + 21xLx32xL)
    output will have  memory order of v[j, a] and w[i, j, a, b] (dimensions Lx21     and     LxLx32x21)
    """

    nsingle = ncol * 21
    nsingle_padded = nsingle + 32 - (nsingle % 32)

    x_single = x[:nsingle].reshape((21, ncol)).T
    x_pair = np.transpose(x[nsingle_padded:].reshape((21, ncol, 32, ncol)), (3, 1, 2, 0))

    if clip:
        x_pair = x_pair[:, :, :21, :21]

    return x_single, x_pair


def structured_to_linear(x_single, x_pair):
    """Convert structured variables into linear array

    with input arrays of memory order v[j, a] and w[i, j, a, b] (dimensions Lx21     and     LxLx32x21)
    output will have  memory order of v[a, j] and w[b, j, a, i] (dimensions 21xL + padding + 21xLx32xL)
    """

    ncol = x_single.shape[0]
    nsingle = ncol * 21
    nsingle_padded = nsingle + 32 - (nsingle % 32)
    nvar = nsingle_padded + ncol * ncol * 21 * 32

    out_x_pair = np.zeros((21, ncol, 32, ncol), dtype='float64')
    out_x_pair[:21, :, :21, :] = np.transpose(x_pair[:, :, :21, :21], (3, 1, 2, 0))

    x = np.zeros((nvar, ), dtype='float64')
    x[:nsingle] = x_single.T.reshape(-1)
    x[nsingle_padded:] = out_x_pair.reshape(-1)

    return x
