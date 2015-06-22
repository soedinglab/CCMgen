import numpy as np


def frobenius_score(x):
    means = np.mean(np.mean(x, axis=2), axis=2)

    x -= means[:, :, np.newaxis, np.newaxis]
    x = x[:, :, :20, :20]

    return np.sqrt(np.sum(x * x, axis=(2, 3)))
