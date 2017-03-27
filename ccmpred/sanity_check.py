import numpy as np


def check_single_potentials(x_single, verbose=0):

    if any(np.abs(x_single.sum(1)) > 1e-10):
        if verbose: print("Warning: Some single potentials do not sum to 0:")
        for i in range(x_single.shape[0]):
            sum_vi = np.abs(np.sum(x_single[i]))
            if sum_vi > 1e-10:
                if verbose: print "i={0} has sum_a(v_ia)={1}".format(i+1, np.sum(x_single[i]))


        return 0

    return 1

def check_pair_potentials(x_pair, verbose=0):

    for i in range(x_pair.shape[0]-1):
        for j in range(i+1, x_pair.shape[0]):
            sum_wij = np.abs(np.sum(x_pair[i, j]))
            if sum_wij > 1e-10:
                if verbose: print "i={0} j={1} has sum_ab(w_ijab)={2}".format(i+1, j+1, np.sum(x_pair[i,j]))
                return 0

    return 1


def normalize_potentials(x_single, x_pair):
    """

    Enforce gauge choice that

    :param x_single:
    :param x_pair:
    :return:
    """

    means = np.mean(np.mean(x_pair, axis=2), axis=2)
    x_pair_centered = x_pair - means[:, :, np.newaxis, np.newaxis]

    means = np.mean(x_single, axis=1)
    x_single_centered = x_single - means[:, np.newaxis]

    return x_single_centered, x_pair_centered
