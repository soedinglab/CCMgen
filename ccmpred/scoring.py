import numpy as np


def frobenius_score(x):
    means = np.mean(np.mean(x, axis=2), axis=2)

    x_centered = x - means[:, :, np.newaxis, np.newaxis]
    x_centered = x_centered[:, :, :20, :20]

    return np.sqrt(np.sum(x_centered * x_centered, axis=(2, 3)))


def apc(cmat):
    mean = np.mean(cmat, axis=0)
    apc_term = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
    return cmat - apc_term
