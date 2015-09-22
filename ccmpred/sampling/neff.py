import sys
import numpy as np
import scipy.optimize
import functools
import operator

import ccmpred.weighting
import ccmpred.objfun.treecd as treecd

"""
Neff is dependant on the evolutionary distance D by the following
relationship:

Neff = 1 + (N - 1) / (1 + exp((t - D) / a))

where N is the number of sampled leaves and a and t are
parameters fitted from a regression.
"""


# starting parameters from a global fit of a few protein families
RFIT_PARAMETERS = {
    'a': 0.0358,
    't': 0.3553
}


def model(x, a, t, n_leaves):
    return 1 + (n_leaves - 1) / (1 + np.exp((t - x) / a))


def fit_neff_model(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0, n_reps=10):
    """Fit model parameters by computing Neff for some sample points"""
    ns = neff_sampler(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0)

    mrs = [0]
    nfs = [1]

    def find_min_mr_for_max_neff(threshold=0.999):
        # sampling strategy 1: grow mutation rate until we get independent sequences
        mrmax = 0.1
        neffmax = ns(mrmax)
        mrs.append(mrmax)
        nfs.append(neffmax)
        while neffmax < n_leaves * threshold:
            mrmax *= 2
            neffmax = ns(mrmax)
            mrs.append(mrmax)
            nfs.append(neffmax)

    def subdivide(n_divisions=20):

        # sampling strategy 2: sample midpoints for intervals with highest delta-x
        for _ in range(n_divisions):

            # find section with highest delta-x
            deltas = [b - a for a, b in zip(mrs, mrs[1:])]
            splitpos = max(((delta, i) for i, delta in enumerate(deltas)), key=operator.itemgetter(0))[1]

            # calculate midpoint and sample corresponding Neff
            xm = (mrs[splitpos] + mrs[splitpos + 1]) / 2
            ym = ns(xm)

            mrs.insert(splitpos + 1, xm)
            nfs.insert(splitpos + 1, ym)

    find_min_mr_for_max_neff()
    subdivide()

    f = functools.partial(model, n_leaves=n_leaves)

    popt, pcov = scipy.optimize.curve_fit(f, mrs, nfs, p0=(1.0, 1.0))

    mdl = dict(zip('at', popt))

    print("Fit model a={a}, t={t}".format(**mdl))

    return mdl


def evoldist_for_neff(target_neff, n_leaves, model_parameters=RFIT_PARAMETERS):
    """Compute the correct evolutionary distance for a target Neff

    From our model, we solve for x and obtain:
    x = t - a*log((N-1) / (Neff - 1) - 1)

    """

    neffratio = np.log((n_leaves - 1) / (target_neff - 1) - 1)
    mutation_rate = model_parameters['t'] - model_parameters['a'] * neffratio

    if mutation_rate < 0:
        raise Exception("Got negative mutation rate {0} for target Neff {1}! (a={a}, t={t})".format(mutation_rate, target_neff, **model_parameters))

    return mutation_rate


def sample_neff(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0):
    ns = neff_sampler(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0)

    print("x y")
    for _ in range(3):
        for mr in np.arange(0, 4.81, 0.4):
            print(mr, ns(mr))
            sys.stdout.flush()


def neff_sampler(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0):
    msa_sampled = np.empty((n_leaves, ncol), dtype="uint8")

    def inner(mutation_rate):

        treecd.cext.mutate_along_tree(msa_sampled, n_children, branch_lengths, x, n_vertices, seq0, mutation_rate)
        neff = np.sum(ccmpred.weighting.weights_simple(msa_sampled))

        return neff

    return inner
