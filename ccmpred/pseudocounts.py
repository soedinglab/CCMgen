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
        self.freqs = None

        self.pseudocount_n_single       = None
        self.pseudocount_n_pair         = None
        self.pseudocount_type           = None
        self.remove_gaps                = None
        self.pseudocount_ratio_single    = None
        self.pseudocount_ratio_pair     = None

        #will be computed from Freq with pseudo-counts and Neff
        self.Ni = None
        self.Nij = None


    def calculate_Ni(self, freqs_single=None):

        if freqs_single is not None:
            #counts may include pseudo-counts
            single_counts = freqs_single * self.neff
        else:
            single_counts, pair_counts = self.counts

        # reset gap counts
        single_counts[:, 20] = 0

        Ni = single_counts.sum(1)

        self.Ni = Ni

    def calculate_Nij(self, freqs_pair=None):

        if freqs_pair is not None:
            #counts may include pseudo-counts
            pair_counts = freqs_pair * self.neff
        else:
            single_counts, pair_counts = self.counts

        # reset gap counts
        pair_counts[:, :, :, 20] = 0
        pair_counts[:, :, 20, :] = 0

        # non_gapped counts
        Nij = pair_counts.sum(3).sum(2)

        self.Nij = Nij

    def calculate_global_aa_freq(self):

        single_counts, _ = self.counts

        #normalized with gaps
        single_freq = single_counts / self.neff

        #single freq counts normalized without gaps
        single_freq = self.degap(single_freq, True)


        return np.mean(single_freq[:, :20], axis=0)[np.newaxis, :][0]

    def calculate_frequencies(self, pseudocount_type, pseudocount_n_single=1, pseudocount_n_pair=None, remove_gaps=False):


        self.pseudocount_n_single   = pseudocount_n_single
        self.pseudocount_n_pair     = pseudocount_n_pair
        self.pseudocount_type       = pseudocount_type
        self.remove_gaps            = remove_gaps

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

        self.freqs = single_freq_pc, pair_freq_pc

        #compute weighted non-gapped sequence counts
        self.calculate_Ni(single_freq_pc)
        self.calculate_Nij(pair_freq_pc)

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
        uniform_pc.fill(1. / single_freq.shape[1])
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
