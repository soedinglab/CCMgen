import numpy as np


def calculate(msa, freqs):
    single_freqs, _ = freqs
    lsingle_freqs = np.log(single_freqs)

    v_center = lsingle_freqs - np.mean(lsingle_freqs[:, :20], axis=1)[:, np.newaxis]
    v_center[:, 20] = 0

    return v_center
