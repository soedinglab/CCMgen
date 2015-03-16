import numpy as np


def frobenius_score(x):
    ncol = x.shape[0]
    means = np.mean(np.mean(x, axis=2), axis=2)

    x_centered = x - means[:, :, np.newaxis, np.newaxis]

    return np.sqrt(np.sum(x_centered * x_centered, axis=(2, 3)))
