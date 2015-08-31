import numpy as np
from ccmpred.weighting.cext import count_ids, calculate_weights


def weights_uniform(msa):
    """Uniform weights"""
    return np.ones((msa.shape[0],), dtype="float64")


def weights_simple(msa, cutoff=0.8):
    """Simple sequence reweighting from the Morcos et al. 2011 DCA paper"""

    if cutoff >= 1:
        return weights_uniform(msa)

    return calculate_weights(msa, cutoff)
