import sys
import numpy as np
import scipy.optimize
import functools

import ccmpred.weighting
import ccmpred.objfun.treecd as treecd

"""
Neff is dependant on the evolutionary distance D by the following
relationship:

Neff = 1 + (N - 1) / (1 + exp(-a*(D-t)^3 + b*(D-t)))

where N is the number of sampled leaves and a, b and t are
parameters fitted from a regression.
"""


# starting parameters from a global fit of a few protein families
RFIT_PARAMETERS = {
    'a': 0.7513,
    'b': -1.1308,
    't': 1.6381
}


def model(x, a, b, t, n_leaves):
    z = x - t
    return 1 + (n_leaves - 1) / (1 + np.exp(-a * z ** 3 + b * z))


def fit_neff_model(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0, n_reps=1, mr_samples=np.arange(0, 0.6, 0.01), start_parameters=RFIT_PARAMETERS):
    """Fit model parameters by computing Neff for some sample points"""
    ns = neff_sampler(branch_lengths, n_children, n_vertices, n_leaves, ncol, x, seq0)

    mrs = np.array([])
    for r in range(n_reps):
        mrs = np.concatenate((mrs, np.array(mr_samples)))

    nfs = np.array([ns(mr) for mr in mrs])

    f = functools.partial(model, n_leaves=n_leaves)

    popt, pcov = scipy.optimize.curve_fit(f, mrs, nfs, p0=(RFIT_PARAMETERS['a'], RFIT_PARAMETERS['b'], RFIT_PARAMETERS['t']))

    for mr, nf in zip(mrs, nfs):
        print("{0}\t{1}\t{2}".format(mr, nf, f(mr, *popt)))

    mdl = {
        'a': popt[0],
        'b': popt[1],
        't': popt[2]
    }

    print("Fit model a={a}, b={b}, t={t}".format(**mdl))

    return mdl


def evoldist_for_neff(target_neff, n_leaves, model_parameters=RFIT_PARAMETERS):
    """Compute the correct evolutionary distance for a target Neff

    From our model, we substitute z = D - t and obtain the polynomial:
    a*z^3 - b*z + log((N - 1) / (Neff - 1) - 1) = 0

    that can be solved using numpy.roots
    """

    neffratio = np.log((n_leaves - 1) / (target_neff - 1) - 1)
    roots = np.roots([model_parameters['a'], 0, -model_parameters['b'], neffratio])

    dists = [np.real(r) + model_parameters['t'] for r in roots if np.abs(np.imag(r)) < 1e-5]

    # only keep real solution and resubstitute D = z + t
    mutation_rate = min(dists)

    if mutation_rate < 0:
        raise Exception("Got negative mutation rate {0} for target Neff {1}! (a={a}, b={b}, t={t})".format(mutation_rate, target_neff, **model_parameters))

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
