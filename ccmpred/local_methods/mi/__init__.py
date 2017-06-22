import numpy as np
import scipy.stats

def compute_mi(counts, normalized=False):
    """

    :param counts: single and pairwise amino acid counts
    :param remove_gaps: do not count gaps
    :param normalized: According to Martin et al 2005
        (Using information theory to search for co-evolving residues in proteins)
        MI is normalized by joint entropy
    :return:
    """

    single_counts, pair_counts = counts


    L = pair_counts.shape[0]
    indices_i_less_j = np.triu_indices(L, k=1) #excluding diagonal

    #compute shannon and joint shannon entropy
    shannon_entropy = scipy.stats.entropy(single_counts.transpose(),base=2)

    joint_shannon_entropy = np.zeros((L, L))
    pair_counts_flat = pair_counts.reshape(L, L, pair_counts.shape[2]*pair_counts.shape[3])
    joint_shannon_entropy[indices_i_less_j] = scipy.stats.entropy(pair_counts_flat[indices_i_less_j].transpose(), base=2)

    #compute mutual information
    mi = np.zeros((L, L))
    mi[indices_i_less_j] =  [shannon_entropy[i] + shannon_entropy[j] - joint_shannon_entropy[i,j] for i,j in zip(*indices_i_less_j)]

    #According to Martin et al 2005
    if normalized:
        mi[indices_i_less_j] /= joint_shannon_entropy[indices_i_less_j]

    #symmetrize
    mi += mi.transpose()


    return mi

def compute_mi_pseudocounts(freqs):

    single_freqs, pair_freqs = freqs

    L = pair_freqs.shape[0]
    indices_i_less_j = np.triu_indices(L, k=1) #excluding diagonal
    mi = np.zeros((L, L))

    #works as it should
    # outer = single_freqs[indices_i_less_j[0]][10, :20, np.newaxis] * single_freqs[indices_i_less_j[1]][10, np.newaxis, :20]
    # print outer[4,7]
    # print outer[7,4]
    # print single_freqs[indices_i_less_j[0]][10,4] * single_freqs[indices_i_less_j[1]][10,7]
    # print single_freqs[indices_i_less_j[0]][10,7] * single_freqs[indices_i_less_j[1]][10,4]

    mi_raw = pair_freqs[indices_i_less_j][:, :20, :20] * np.log2(pair_freqs[indices_i_less_j][:, :20, :20] / (single_freqs[indices_i_less_j[0]][:, :20, np.newaxis] * single_freqs[indices_i_less_j[1]][:, np.newaxis, :20]) )


    mi[indices_i_less_j] = mi_raw.sum(2).sum(1)

    #symmetrize
    mi += mi.transpose()

    return mi
