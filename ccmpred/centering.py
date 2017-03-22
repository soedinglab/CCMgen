import numpy as np
import ccmpred.pseudocounts



def center_v(freqs):
    single_freqs, _ = freqs

    #single_freqs either normalized with or without gaps --> same result due to subtraction of mean


    #hack when usign no pseudo counts to be able to take log of zero counts
    eps = 1e-10
    single_freqs[single_freqs<eps]=eps
    lsingle_freqs = np.log(single_freqs)

    #subtract mean of non_gapped frequencies
    v_center = lsingle_freqs - np.mean(lsingle_freqs[:, :20], axis=1)[:, np.newaxis]
    v_center[:, 20] = 0

    return v_center


def center_zero(freqs):
    single_freqs, _ = freqs

    return np.zeros_like(single_freqs)


def center_vanilla(freqs):
    single_freqs, _ = freqs

    #single_freqs will never be zero as there is at least 1 pseudo count
    lsingle_freqs = np.log(single_freqs)


    v_center = lsingle_freqs - lsingle_freqs[:, 20][:, np.newaxis]
    v_center[:, 20] = 0



    return v_center

