import numpy as np


def check_single_potentials(x_single, verbose=0):

    if any(np.abs(x_single.sum(1)) > 1e-10):
        print("Warning: Some single potentials do not sum to 0")

        if verbose:
            indices = np.where(np.abs(x_single.sum(1)) > 1e-10)[0]
            for ind in indices[:10]:
                print "eg: i={0:<2} has sum_a(v_ia)={1}".format(ind+1, np.sum(x_single[ind]))

        return 0

    return 1

def check_pair_potentials(x_pair, verbose=0):

    if any(np.abs(x_pair.sum(2).sum(2)[np.triu_indices(x_pair.shape[0], k=1)]) > 1e-10):
        print("Warning: Some pair potentials do not sum to 0")

        if verbose:
            indices_triu = np.triu_indices(x_pair.shape[0], 1)
            indices = np.where(np.abs(x_pair.sum(2).sum(2)[indices_triu]) > 1e-10)[0]
            for ind in indices[:10]:
                i = indices_triu[0][ind]
                j = indices_triu[1][ind]
                print "eg: i={0:<2} j={1:<2} has sum_ab(w_ijab)={2}".format(i+1, j+1, np.sum(x_pair[i,j]))

        return 0

    return 1


def centering_potentials( x_single, x_pair):
    """

    Enforce gauge choice that

    :param x_single:
    :param x_pair:
    :return:
    """

    means = np.mean(np.mean(x_pair[:, :, :20, :20], axis=2), axis=2)
    x_pair[:, :, :20, :20] -=  means[:, :, np.newaxis, np.newaxis]

    means = np.mean(x_single[: , :20], axis=1)
    x_single[: , :20] -= means[:, np.newaxis]


    return x_single, x_pair
