import numpy as np

import ccmpred.raw
import ccmpred.regularization
import ccmpred.objfun
import ccmpred.objfun.pll.cext
import ccmpred.counts
import ccmpred.parameter_handling

class PseudoLikelihood():
    def __init__(self, msa, weights, regularization, pseudocounts, x_single, x_pair):

        self.msa = msa
        self.nrow, self.ncol = msa.shape
        self.weights = weights
        self.neff = np.sum(weights)
        self.regularization = regularization

        self.structured_to_linear = lambda x_single, x_pair: \
            ccmpred.parameter_handling.structured_to_linear(
                x_single, x_pair, nogapstate=False, padding=True)
        self.linear_to_structured = lambda x: \
            ccmpred.parameter_handling.linear_to_structured(
                x, self.ncol, nogapstate=False, add_gap_state=False, padding=True)

        self.x_single = x_single
        self.x_pair = x_pair
        self.x = self.structured_to_linear(self.x_single, self.x_pair)

        #use msa counts with pseudo counts - numerically more stable?? but gradient does not fit ll fct!!
        #self.freqs_single, self.freqs_pair = ccm.pseudocounts.freqs
        #msa_counts_single, msa_counts_pair = neff * freqs_single, neff * freqs_pair
        #use msa counts without pseudo counts
        msa_counts_single, msa_counts_pair = pseudocounts.counts

        msa_counts_single[:, 20] = 0
        msa_counts_pair[:, :, 20, :] = 0
        msa_counts_pair[:, :, :, 20] = 0

        for i in range(self.ncol):
            msa_counts_pair[i, i, :, :] = 0

        #non_gapped counts
        # self.Ni = msa_counts_single.sum(1)
        # self.Nij = msa_counts_pair.sum(3).sum(2)

        #no pseudo counts in gradient calculation
        #pairwise gradient is two-fold
        self.g_init = ccmpred.parameter_handling.structured_to_linear(
            msa_counts_single, 2 * msa_counts_pair)

        self.nsingle = self.ncol * 21
        self.nsingle_padded = self.nsingle + 32 - (self.nsingle % 32)
        self.nvar = self.nsingle_padded + self.ncol * self.ncol * 21 * 32

        # memory allocation for intermediate variables
        #gradient for single and pair potentials
        self.g = np.empty((self.nsingle_padded + self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))
        #gradient for only pair potentials
        self.g2 = np.empty((self.ncol * self.ncol * 21 * 32,), dtype=np.dtype('float64'))


    def finalize(self, x):
        return ccmpred.parameter_handling.linear_to_structured(
            x, self.ncol, clip=True, nogapstate=False, add_gap_state=False, padding=True)

    def evaluate(self, x):

        #pointer to g == self.g
        #pairwise gradient is two-fold  because auf symmetrization
        fx, g = ccmpred.objfun.pll.cext.evaluate(x, self.g, self.g2, self.weights, self.msa)
        g -= self.g_init

        x_single, x_pair = self.linear_to_structured(x)

        #pairwise gradient is two-fold !
        fx_reg, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair)
        g_pair_reg *= 2
        g_reg = self.structured_to_linear(g_single_reg, g_pair_reg)
        fx += fx_reg

        return fx, g, g_reg

    def get_parameters(self):
        return {'padding' : True,
                'pseudocounts': False}

    def __repr__(self):
        return "PLL "

