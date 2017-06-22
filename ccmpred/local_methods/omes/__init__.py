import numpy as np


def compute_omes(counts, fodoraldrich=False):
    """

    Chi squared statistic:
    X^2 = sum_{i=1}^N  [(O_i - E_i)^2 / E_i   ]       # comparing counts
        = N sum_{i=1}^N [(O_i/N - p_i)^2 / p_i ]      # comparing frequencies

    O_i = number of observations of type i => pairwise amino acid counts
    E_i = Np_i = the expected (theoretical) occurence of type i,
                 asserted by the null hypothesis that the fraction of type i in the population is p_{i}



    According to Kass & Horovitz, 2002:
    Mapping Pathways of Allosteric Communication in GroEL by Analysis of Correlated Mutations

    omes(i,j) =                     [ count_ij(a,b) - (count_i(a) * count_j(b))/N_ij ] ^2
                    sum_(a,b=1)^20   -----------------------------------------------------
                                           (count_i(a) * count_j(b))/N_ij


    According to Fodor & Aldrich, 2004:
    Influence of conservation on calculations of amino acid covariance in multiple sequence alignments.
    omes(i,j) =                     [ count_ij(a,b) - (count_i(a) * count_j(b))/N_ij ] ^2
                    sum_(a,b=1)^20   -----------------------------------------------------
                                                     N_ij


    Here we implement Kass & Horovitz! (see line 43)

    :return:
    """

    single_counts, pair_counts = counts
    Nij = pair_counts.sum(3).sum(2) #== Neff
    L = single_counts.shape[0]

    # gaps do not add
    # if gap_treatment:
    #     Nij = pair_counts[:, :, :20, :20].sum(3).sum(2)

    # compute chi square statistic
    Nexp = np.outer(single_counts[:, :20], single_counts[:, :20]).reshape((L, L, 20, 20))

    #works as it should
    # print Nexp[0, 11, 2, 4]
    # print single_counts[0, 2] * single_counts[11, 4]


    Nexp /= Nij[:, :, np.newaxis, np.newaxis]
    diff = (pair_counts[:, :, :20, :20] - Nexp)

    if fodoraldrich:
        omes = (diff * diff) / Nij[:, :, np.newaxis, np.newaxis]  # Fodor & Aldrich: we divide by Nij(neff)
    else:
        omes = (diff * diff) / Nexp  # Kass & Horovitz: we divide by Nexp

    omes = omes.sum(3).sum(2)


    return omes



def compute_omes_freq(counts, freqs, fodoraldrich=False, ignore_zero_counts=True):


    single_freqs, pair_freqs = freqs
    single_counts, pair_counts = counts
    Nij = pair_counts.sum(3).sum(2) #== Neff
    L = single_freqs.shape[0]

    # gaps do not add
    # if gap_treatment:
    #     Nij = pair_counts[:, :, :20, :20].sum(3).sum(2)

    # compute chi square statistic
    Nexp = single_freqs[:, np.newaxis, :20, np.newaxis] * single_freqs[np.newaxis, :, np.newaxis, :20]

    #works as it should
    # print Nexp[0, 11, 2, 4]
    # print single_counts[0, 2] * single_counts[11, 4]


    Nexp *= Nij[:, :, np.newaxis, np.newaxis]
    diff = (pair_counts[:, :, :20, :20] - Nexp)


    if fodoraldrich:
        omes_full = (diff * diff) / Nij[:, :, np.newaxis, np.newaxis]  # Fodor & Aldrich: we divide by Nij(neff)
    else:
        omes_full = (diff * diff) / Nexp  # Kass & Horovitz: we divide by Nexp



    #compute statistics only for non-zero  pair counts
    if ignore_zero_counts:
        ind_nonzero_ab  = np.nonzero(pair_counts[:, :, :20, :20])
        omes = np.zeros((L, L, 20, 20))
        omes[ind_nonzero_ab] = omes_full[ind_nonzero_ab]
    else:
        omes = omes_full

    omes = omes.sum(3).sum(2)

    return omes
