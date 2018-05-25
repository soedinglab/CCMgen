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

    # weighted frequencies excluding gaps
    # weights = calculate_weights_simple(msa, cutoff=0.8, ignore_gaps=False)
    # pseudocounts = PseudoCounts(msa, weights)
    # pseudocounts.calculate_frequencies("uniform_pseudocounts", 1, 1, remove_gaps=False)
    # single_freqs = pseudocounts.degap(pseudocounts.freqs[0])

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


def weights_simple(msa, cutoff=0.8, ignore_gaps=False):
    """Simple sequence reweighting from the Morcos et al. 2011 DCA paper"""

    if cutoff >= 1:
        return weights_uniform(msa)

    return calculate_weights_simple(msa, cutoff, ignore_gaps)


def weights_henikoff(msa, ignore_gaps=False ):
        """
        Henikoff weighting according to Henikoff, S and Henikoff, JG. Position-based sequence weights. 1994
        Henikoff weights always sum up to ncol
        """

        single_counts   = ccmpred.counts.single_counts(msa, None)
        if ignore_gaps:
            single_counts[:,0] = 0

        unique_aa       = (single_counts != 0).sum(1)
        nrow, ncol = msa.shape

        cNi = np.array([single_counts[range(ncol), msa[n]] * unique_aa for n in range(nrow)])
        henikoff =  np.array([sum(1/np.delete(cNi[n], np.where(cNi[n] == 0))) for n in range(nrow)])


        #example from henikoff paper
        # msa=np.array([[0,1,3,0,6],[0,2,4,0,2],[0,1,4,0,2],[0,1,5,0,0]])
        # single_counts = np.zeros((5, 7))
        # single_counts[0,0]=4
        # single_counts[1,1]=3
        # single_counts[1,2]=1
        # single_counts[2,3]=1
        # single_counts[2,4]=2
        # single_counts[2,5]=1
        # single_counts[3,0]=4
        # single_counts[4,0]=1
        # single_counts[4,2]=2
        # single_counts[4,6]=1
        # unique_aa       = (single_counts != 0).sum(1)
        # henikoff = np.array([sum(1.0/(single_counts[range(ncol), msa[n]] * unique_aa)) for n in range(nrow)])
        # array([ 1.33333333,  1.33333333,  1.        ,  1.33333333])

        return henikoff


WEIGHTING_TYPE = {
    'simple': lambda msa, cutoff, ignore_gaps: weights_simple(msa, cutoff, ignore_gaps),
    'henikoff': lambda msa, cutoff, ignore_gaps: weights_henikoff(msa, ignore_gaps),
    'uniform': lambda msa, cutoff, ignore_gaps: weights_uniform(msa)
}
