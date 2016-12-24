import numpy as np
from ccmpred.weighting.cext import count_ids, calculate_weights_simple
import ccmpred.counts

def weights_uniform(msa):
    """Uniform weights"""
    return np.ones((msa.shape[0],), dtype="float64")


def weights_simple(msa, ignore_gaps=False, cutoff=0.8):
    """Simple sequence reweighting from the Morcos et al. 2011 DCA paper"""

    if cutoff >= 1:
        return weights_uniform(msa)

    return calculate_weights_simple(msa, cutoff, ignore_gaps)


def weights_henikoff(msa, ignore_gaps=False):
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


def weights_henikoff_pair(msa, ignore_gaps=False):
    """
    Henikoff pair weighting

    Henikoff pair weights will always sum up to ncol*(ncol-1)/2
    """

    pair_counts   = ccmpred.counts.pair_counts(msa, None)
    if ignore_gaps:
        pair_counts[:, :, 0, :] = 0
        pair_counts[:, :, :, 0] = 0

    unique_aa     = (pair_counts != 0).sum(3).sum(2)
    nrow, ncol = msa.shape

    henikoff = np.zeros(nrow)
    for n in range(nrow):
        for k in range(ncol-1):
            for l in range(k+1,ncol):
                if pair_counts[k,l, msa[n][k],msa[n][l]] != 0:
                    henikoff[n] += 1/(pair_counts[k,l, msa[n][k],msa[n][l]]  * unique_aa[k,l])


    return henikoff




