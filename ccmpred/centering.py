import numpy as np
import ccmpred.pseudocounts

def calculate(freqs):
    single_freqs, _ = freqs

    #single_freqs either normalized with or without gaps --> same result due to subtraction of mean
    lsingle_freqs = np.log(single_freqs)

    v_center = lsingle_freqs - np.mean(lsingle_freqs[:, :20], axis=1)[:, np.newaxis]
    v_center[:, 20] = 0

    return v_center
