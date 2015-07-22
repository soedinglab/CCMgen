import numpy as np
import ccmpred.counts


def calculate(msa, weights, tau=0.1):
    nrow, ncol = msa.shape
    wsum = np.sum(weights)

    single_counts = ccmpred.counts.single_counts(msa, weights)

    aa_global_frac = np.sum(single_counts, axis=0) / (ncol * wsum)

    aafrac = single_counts / (wsum - single_counts[:, 20])[:, np.newaxis]
    aafrac[:, 20] = 0

    aafrac_pseudo = (1 - tau) * aafrac[:, :20] + tau * aa_global_frac[np.newaxis, :20]
    aafrac_logsum = np.sum(np.log(aafrac_pseudo), axis=1)

    v_center = np.log(aafrac_pseudo) - aafrac_logsum[:, np.newaxis] / 20
    return v_center
