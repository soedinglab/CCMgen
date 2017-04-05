import numpy as np

import ccmpred.counts
import ccmpred.substitution_matrices

def get_neff(msa):
    single_counts = ccmpred.counts.single_counts(msa)
    single_freqs = (single_counts + 1e-3) / np.sum(single_counts, axis=1)[:, np.newaxis]

    single_freqs = single_freqs[:20]

    entropies = - np.sum(single_freqs * np.log2(single_freqs), axis=1)

    neff = 2 ** np.mean(entropies)

    return neff


def calculate_global_aa_freq(msa, weights):

    single_counts = ccmpred.counts.single_counts(msa, weights)
    neff = np.sum(weights) if weights is not None else msa.shape[0]

    #normalized with gaps
    single_freq = single_counts / neff

    #single freq counts normalized without gaps
    single_freq = degap(single_freq, True)


    return np.mean(single_freq[:, :20], axis=0)[np.newaxis, :][0]


def calculate_frequencies_vanilla(msa):

    print("Calculating AA Frequencies as in C++ CCMpred vanilla: 1 pseudocount is added to single_counts")

    single_counts, pair_counts = ccmpred.counts.both_counts(msa, None)
    nrow =  msa.shape[0]

    #add one pseudocunt to every single amino acid count
    single_counts += 1


    #normalized with gaps
    single_freq = single_counts / (nrow + 21)
    pair_freq = pair_counts / nrow

    return single_freq, pair_freq

def calculate_frequencies_dev_center_v(msa, weights):


    single_counts, pair_counts = ccmpred.counts.both_counts(msa, weights)
    nrow = np.sum(weights) if weights is not None else msa.shape[0]

    pseudocount_ratio = 0.1

    print("Calculating AA Frequencies as in dev-center-v: " +
          str(np.round(pseudocount_ratio, decimals=5)) +
          " percent pseudocounts for single freq (constant pseudocounts from global frequencies with gaps)")

    #normalized with gaps
    single_freq = single_counts / nrow
    pair_freq = pair_counts / nrow

    #pseudocounts from global aa frequencies with gaps
    pcounts = constant_pseudocounts(single_freq)

    #single freq counts normalized without gaps
    single_freq = degap(single_freq, True)
    pair_freq = degap(pair_freq, True)

    single_freq_pc = (1 - pseudocount_ratio) * single_freq + pseudocount_ratio * pcounts
    pair_freq_pc = ((1 - pseudocount_ratio) ** 2) * \
                   (pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]) + \
                   (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

    return single_freq_pc, pair_freq_pc


def calculate_frequencies(msa, weights, pseudocount_function, pseudocount_n_single=1, pseudocount_n_pair=None, remove_gaps=False):

    if pseudocount_n_pair is None:
        pseudocount_n_pair = pseudocount_n_single

    single_counts, pair_counts = ccmpred.counts.both_counts(msa, weights)
    neff = np.sum(weights) if weights is not None else msa.shape[0]

    pseudocount_ratio_single = pseudocount_n_single / (neff + pseudocount_n_single)
    pseudocount_ratio_pair = pseudocount_n_pair / (neff + pseudocount_n_pair)

    print("Calculating AA Frequencies with {0} percent pseudocounts ({1} {2} {3})".format(np.round(pseudocount_ratio_single, decimals=5),
                                                                                                                     pseudocount_function.__name__,
                                                                                                                     pseudocount_n_single,
                                                                                                                     pseudocount_n_pair))

    #frequencies are normalized WITH gaps
    single_freq = single_counts / neff
    pair_freq = pair_counts / neff

    pcounts = pseudocount_function(single_freq)

    if (remove_gaps):
        single_freq = degap(single_freq,True)
        pair_freq = degap(pair_freq, True)

    single_freq_pc = (1 - pseudocount_ratio_single) * single_freq + pseudocount_ratio_single * pcounts
    pair_freq_pc = ((1 - pseudocount_ratio_pair) ** 2) * \
                   (pair_freq - single_freq[:, np.newaxis, :, np.newaxis] * single_freq[np.newaxis, :, np.newaxis, :]) + \
                   (single_freq_pc[:, np.newaxis, :, np.newaxis] * single_freq_pc[np.newaxis, :, np.newaxis, :])

    return single_freq_pc, pair_freq_pc



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


def uniform_pseudocounts(single_freq):
    uniform_pc = np.zeros_like(single_freq)
    uniform_pc.fill(1./21)
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
