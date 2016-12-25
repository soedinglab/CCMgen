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

    print("Calculating AA Frequencies with " +
          str(np.round(pseudocount_ratio_single, decimals=5)) +
          " percent pseudocounts (" + pseudocount_function.__name__+ " " + str(pseudocount_n_single)+")")

    #normalized with gaps
    single_freq = single_counts / nrow
    pair_freq = pair_counts / nrow

    #normalized with gaps
    pcounts = pseudocount_function(single_freq)

    #remove gaps and renormalize
    single_freq = degap(single_freq, True)
    pair_freq   = degap(pair_freq, True)

    single_freq_pc = (1 - pseudocount_ratio_single) * single_freq + pseudocount_ratio_single * pcounts
    pair_freq_pc = ((1 - pseudocount_ratio_pair) * (1 - pseudocount_ratio_pair)) * (
        pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]
    ) + (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

    return single_freq_pc, pair_freq_pc


def degap(freq, keep_dims=False):
    if len(freq.shape) == 2 :
        out = freq[:, :20] / (1 - freq[:, 20])[:, np.newaxis]
    else:
        out = freq[:, :, :20, :20] / freq[:,:,:20, :20].sum(3).sum(2)[:, :,  np.newaxis, np.newaxis]

    if keep_dims:
        if len(freq.shape) == 2 :
            out2 = np.zeros((freq.shape[0], 21))
            out2[:, :20] = out
        else:
            out2 = np.zeros((freq.shape[0], freq.shape[1], 21, 21))
            out2[:, :, :20, :20] = out
        out = out2

    return out


def uniform_pseudocounts(single_freq):
    uniform_pc = np.zeros_like(single_freq)
    uniform_pc.fill(1./20)
    return uniform_pc

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
