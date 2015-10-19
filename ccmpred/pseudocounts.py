import numpy as np

import ccmpred.counts


def calculate_frequencies(msa, weights, pseudocount_function, pseudocount_n=1):
    single_counts, pair_counts = ccmpred.counts.both_counts(msa, weights)
    nrow = np.sum(weights) if weights is not None else msa.shape[0]

    pseudocount_ratio = pseudocount_n / (nrow + pseudocount_n)

    single_freq = single_counts / nrow
    pair_freq = pair_counts / nrow

    pcounts = pseudocount_function(single_freq)

    single_freq_pc = (1 - pseudocount_ratio) * single_freq + pseudocount_ratio * pcounts
    pair_freq_pc = ((1 - pseudocount_ratio) ** 2) * (
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


def substitution_matrix_pseudocounts(single_freq):
    raise Exception("Implement me!")


def no_pseudocounts(single_freq):
    return single_freq
