import numpy as np

import ccmpred.counts
import ccmpred.substitution_matrices


class PseudoCounts(object):
    """Add pseudocounts to prevent vanishing amino acid frequencies"""

    def __init__(self, msa, weights):

        self.msa = msa
        self.N,  self.L = self.msa.shape
        self.weights=weights
        self.neff = np.sum(weights) if self.weights is not None else self.N

        #with weights
        self.counts = ccmpred.counts.both_counts(self.msa, self.weights)

        self.pseudocount_n_single   = None
        self.pseudocount_n_pair     = None
        self.pseudocount_type       = None
        self.remove_gaps            = None
        self.pseudocount_ratio_single = None
        self.pseudocount_ratio_pair = None


    def calculate_global_aa_freq(self):

        single_counts, _ = self.counts

        #normalized with gaps
        single_freq = single_counts / self.neff

        #single freq counts normalized without gaps
        single_freq = self.degap(single_freq, True)


        return np.mean(single_freq[:, :20], axis=0)[np.newaxis, :][0]

    def calculate_frequencies_vanilla(self):

        print("Calculating AA Frequencies as in C++ CCMpred vanilla: 1 pseudocount is added to single_counts")

        self.pseudocount_type   = "ccmpred-vanilla"

        #without sequence weights
        single_counts, pair_counts = ccmpred.counts.both_counts(self.msa, None)

        #add one pseudocunt to every single amino acid count
        single_counts += 1

        #normalized with gaps
        single_freq = single_counts / (self.N + 21)
        pair_freq = pair_counts / self.N

        return single_freq, pair_freq

    def calculate_frequencies_dev_center_v(self):

        self.pseudocount_type   = "constant_pseudocounts"

        #with weights
        single_counts, pair_counts = self.counts

        self.pseudocount_ratio_single = 0.1
        self.pseudocount_ratio_pair = 0.1

        print("Calculating AA Frequencies as in dev-center-v: " +
              str(np.round(pseudocount_ratio, decimals=5)) +
              " percent pseudocounts for single freq (constant pseudocounts from global frequencies with gaps)")

        #normalized with gaps
        single_freq = single_counts / self.N
        pair_freq = pair_counts / self.N

        #pseudocounts from global aa frequencies with gaps
        pcounts = self.constant_pseudocounts(single_freq)

        #single freq counts normalized without gaps
        single_freq = self.degap(single_freq, True)
        pair_freq = self.degap(pair_freq, True)

        single_freq_pc = (1 - self.pseudocount_ratio_single) * single_freq + self.pseudocount_ratio_single * pcounts
        pair_freq_pc = ((1 - self.pseudocount_ratio_pair ) ** 2) * \
                       (pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]) + \
                       (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

        return single_freq_pc, pair_freq_pc

    def calculate_frequencies(self, pseudocount_type, pseudocount_n_single=1, pseudocount_n_pair=None, remove_gaps=False):

        self.pseudocount_n_single   = pseudocount_n_single
        self.pseudocount_n_pair     = pseudocount_n_pair
        self.pseudocount_type       = pseudocount_type
        self.remove_gaps = remove_gaps

        single_counts, pair_counts = self.counts

        if pseudocount_n_pair is None:
            pseudocount_n_pair = pseudocount_n_single

        self.pseudocount_ratio_single = pseudocount_n_single / (self.neff + pseudocount_n_single)
        self.pseudocount_ratio_pair = pseudocount_n_pair / (self.neff + pseudocount_n_pair)

        #frequencies are normalized WITH gaps
        single_freq = single_counts / self.neff
        pair_freq = pair_counts / self.neff

        if (remove_gaps):
            single_freq = self.degap(single_freq,True)
            pair_freq = self.degap(pair_freq, True)

        pcounts = getattr(self, pseudocount_type)(single_freq)

        single_freq_pc = (1 - self.pseudocount_ratio_single) * single_freq + self.pseudocount_ratio_single * pcounts
        pair_freq_pc = ((1 - self.pseudocount_ratio_pair) ** 2) * \
                       (pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]) + \
                       (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

        return single_freq_pc, pair_freq_pc

    @staticmethod
    def degap(freq, keep_dims=False):
        if len(freq.shape) == 2 :
            out = freq[:, :20] / (1 - freq[:, 20])[:, np.newaxis]
        else:
            freq_sum = freq[:,:,:20, :20].sum(3).sum(2)[:, :,  np.newaxis, np.newaxis]
            out = freq[:, :, :20, :20] / (freq_sum + 1e-10)

        if keep_dims:
            if len(freq.shape) == 2 :
                out2 = np.zeros((freq.shape[0], 21))
                out2[:, :20] = out
            else:
                out2 = np.zeros((freq.shape[0], freq.shape[1], 21, 21))
                out2[:, :, :20, :20] = out
            out = out2

        return out

    def uniform_pseudocounts(self, single_freq):
        uniform_pc = np.zeros_like(single_freq)
        uniform_pc.fill(1./21)
        return uniform_pc

    def constant_pseudocounts(self, single_freq):
        return np.mean(single_freq, axis=0)[np.newaxis, :]

    def substitution_matrix_pseudocounts(self, single_freq, substitution_matrix=ccmpred.substitution_matrices.BLOSUM62):
        """
        Substitution matrix pseudocounts

        $\tilde{q}(x_i = a) = \sum_{b=1}^{20} p(a | b) q_0(x_i = b)$
        """
        single_freq_degap = self.degap(single_freq)

        # $p(b) = \sum{a=1}^{20} p(a, b)$
        pb = np.sum(substitution_matrix, axis=0)

        # p(a | b) = p(a, b) / p(b)
        cond_prob = substitution_matrix / pb[np.newaxis, :]

        freqs_pc = np.zeros_like(single_freq)
        freqs_pc[:, :20] = np.sum(cond_prob[np.newaxis, :, :] * single_freq_degap[:, np.newaxis, :], axis=2)

        return freqs_pc

    def no_pseudocounts(self, single_freq):
        return single_freq
