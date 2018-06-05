import numpy as np
from ccmpred.weighting.cext import count_ids, calculate_weights_simple
import ccmpred.counts
from ccmpred.pseudocounts import PseudoCounts

def get_HHsuite_neff(msa):
    """
    Adapted from the HHsuite manual:

    The number of effective sequences is exp of the average sequence entropy over all columns of the alignment.
    Hence, Neff is bounded by 0 from below and 20 from above.
    In practice, it is bounded by the entropy of a column with background amino acid distribution f_a:
    Neff < sum_a=1^20 f_a log f_a approx 16

    Parameters
    ----------
    msa

    Returns
    -------

    """

    # frequencies including gaps
    single_counts = ccmpred.counts.single_counts(msa)
    single_freqs = (single_counts + 1e-3) / np.sum(single_counts, axis=1)[:, np.newaxis]


    single_freqs = single_freqs[:, :20]
    entropies = - np.sum(single_freqs * np.log2(single_freqs), axis=1)

    neff = 2 ** np.mean(entropies)

    return neff

def weights_uniform(msa):
    """Uniform weights"""
    return np.ones((msa.shape[0],), dtype="float64")


def weights_simple(msa, cutoff=0.8):
    """Simple sequence reweighting from the Morcos et al. 2011 DCA paper"""

    if cutoff >= 1:
        return weights_uniform(msa)

    return calculate_weights_simple(msa, cutoff)



WEIGHTING_TYPE = {
    'simple': lambda msa, cutoff: weights_simple(msa, cutoff),
    'uniform': lambda msa, cutoff: weights_uniform(msa)
}
