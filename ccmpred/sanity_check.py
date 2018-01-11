import numpy as np


def check_single_potentials(x_single, verbose=0, epsilon=1e-5):

    nr_pot_sum_not_zero = np.where(np.abs(x_single.sum(1)) > epsilon)[0]
    if len(nr_pot_sum_not_zero) > 0:
        print("Warning: {0} single potentials do not sum to 0 (eps={1}).".format(len(nr_pot_sum_not_zero), epsilon))

        if verbose:
            for ind in nr_pot_sum_not_zero[:10]:
                print "eg: i={0:<2} has sum_a(v_ia)={1}".format(ind+1, np.sum(x_single[ind]))

        return 0

    return 1

def check_pair_potentials(x_pair, verbose=0, epsilon=1e-5):

    indices_triu = np.triu_indices(x_pair.shape[0], 1)
    nr_pot_sum_not_zero = np.where(np.abs(x_pair.sum(2).sum(2)[indices_triu]) > epsilon)[0]
    if len(nr_pot_sum_not_zero):
        print("Warning: {0}/{1} pair potentials do not sum to 0 (eps={2}).".format(len(nr_pot_sum_not_zero), len(indices_triu[0]), epsilon))

        if verbose:
            for ind in nr_pot_sum_not_zero[:10]:
                i = indices_triu[0][ind]
                j = indices_triu[1][ind]
                print "eg: i={0:<2} j={1:<2} has sum_ab(w_ijab)={2}".format(i+1, j+1, np.sum(x_pair[i,j]))

        return 0

    return 1


def centering_potentials( x_single, x_pair):
    """

    Enforce gauge choice

    :param x_single:
    :param x_pair:
    :return:
    """

    means = np.mean(np.mean(x_pair[:, :, :20, :20], axis=2), axis=2)
    x_pair[:, :, :20, :20] -=  means[:, :, np.newaxis, np.newaxis]

    means = np.mean(x_single[: , :20], axis=1)
    x_single[: , :20] -= means[:, np.newaxis]


    return x_single, x_pair
