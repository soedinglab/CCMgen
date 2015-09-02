import numpy as np


RFIT_PARAMETERS = {
    'a': -1.238,
    'b': 1.033,
    't': 1.547
}


def evoldist_for_neff(target_neff, n_leaves, model_parameters=RFIT_PARAMETERS):
    """Compute the correct evolutionary distance for a target Neff

    Neff is dependant on the evolutionary distance D by the following
    relationship:

    Neff = 1 + (N - 1) / (1 + exp(-a*(D-t)^3 + b*(D-t)))

    where N is the number of sampled leaves and a, b and t are
    parameters fitted from a regression.

    we substitute z = D - t and obtain the polynomial:
    a*z^3 + b*z + log((N - 1) / (Neff - 1) - 1) = 0
    solve using numpy.roots
    """

    neffratio = np.log((n_leaves - 1) / (target_neff - 1) - 1)
    roots = np.roots([model_parameters['a'], 0, model_parameters['b'], neffratio])

    # only keep real solution and resubstitute D = z + t
    mutation_rate = np.real(roots[np.abs(np.imag(roots)) < 1e-5])[0] + model_parameters['t']

    return mutation_rate
