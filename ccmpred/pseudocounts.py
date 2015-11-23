import numpy as np

import ccmpred.counts
import ccmpred.substitution_matrices


def calculate_frequencies(msa, weights, pseudocount_function, pseudocount_n_single=1, pseudocount_n_pair=None):

    if pseudocount_n_pair is None:
        pseudocount_n_pair = pseudocount_n_single

    single_counts, pair_counts = ccmpred.counts.both_counts(msa, weights)
    nrow = np.sum(weights) if weights is not None else msa.shape[0]

    pseudocount_ratio_single = pseudocount_n_single / (nrow + pseudocount_n_single)
    pseudocount_ratio_pair = pseudocount_n_pair / (nrow + pseudocount_n_pair)

    single_freq = single_counts / nrow
    pair_freq = pair_counts / nrow

    pcounts = pseudocount_function(single_freq)

    single_freq_pc = (1 - pseudocount_ratio_single) * single_freq + pseudocount_ratio_single * pcounts
    pair_freq_pc = ((1 - pseudocount_ratio_pair) ** 2) * (
        pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]
    ) + (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

    return single_freq_pc, pair_freq_pc


def degap(single_freq, keep_dims=False):
    out = single_freq[:, :20] / (1 - single_freq[:, 20])[:, np.newaxis]

    if keep_dims:
        out2 = np.zeros((single_freq.shape[0], 21))
        out2[:, :20] = out
        out = out2

    return out


def constant_pseudocounts(single_freq):
    return np.mean(single_freq, axis=0)[np.newaxis, :]


def substitution_matrix_pseudocounts(single_freq, substitution_matrix=ccmpred.substitution_matrices.BLOSUM62):
    """
    Substitution matrix pseudocounts

    $\tilde{q}(x_i = a) = \sum_{b=1}^{20} p(a | b) q_0(x_i = b)$
    """
    single_freq_degap = degap(single_freq)

    # $p(b) = \sum{a=1}^{20} p(a, b)$
    pb = np.sum(substitution_matrix, axis=0)

    # p(a | b) = p(a, b) / p(b)
    cond_prob = substitution_matrix / pb[np.newaxis, :]

    freqs_pc = np.zeros_like(single_freq)
    freqs_pc[:, :20] = np.sum(cond_prob[np.newaxis, :, :] * single_freq_degap[:, np.newaxis, :], axis=2)

    return freqs_pc


def no_pseudocounts(single_freq):
    return single_freq
