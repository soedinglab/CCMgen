import numpy as np


def frobenius_score(x):

    return np.sqrt(np.sum(x * x, axis=(2, 3)))


def apc(cmat):
    mean = np.mean(cmat, axis=0)
    apc_term = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
    return cmat - apc_term
