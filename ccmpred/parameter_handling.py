import numpy as np


def linear_to_structured(x, ncol, clip=False, nogapstate=False, add_gap_state=False, padding=True):
    """Convert linear vector of variables into multidimensional arrays.

    in linear memory, memory order is v[a, j] and w[b, j, a, i] (dimensions 21xL + padding + 21xLx32xL)
    output will have  memory order of v[j, a] and w[i, j, a, b] (dimensions Lx21     and     LxLx32x21)
    """


    single_states = 21
    if nogapstate:
        single_states = 20

    nsingle = ncol * single_states

    if padding:

        nsingle_padded = nsingle + 32 - (nsingle % 32)

        x_single = x[:nsingle].reshape((single_states, ncol)).T
        x_pair = np.transpose(x[nsingle_padded:].reshape((21, ncol, 32, ncol)), (3, 1, 2, 0))

    else:
        x_single = x[:nsingle].reshape((ncol, single_states))
        x_pair = np.transpose(x[nsingle:].reshape((ncol, 21, ncol, 21)), (0, 2, 1, 3))

        if add_gap_state:
            temp = np.zeros((ncol, 21))
            temp[:, :20] = x_single
            x_single = temp

    if clip:
        x_pair = x_pair[:, :, :21, :21]

    return x_single, x_pair


def structured_to_linear(x_single, x_pair, nogapstate=False, padding=True):
    """Convert structured variables into linear array

    with input arrays of memory order v[j, a] and w[i, j, a, b] (dimensions Lx21     and     LxLx32x21)
    output will have  memory order of v[a, j] and w[b, j, a, i] (dimensions 21xL + padding + 21xLx32xL)
    """

    single_states = 21
    if nogapstate:
        single_states = 20


    ncol = x_single.shape[0]
    nsingle = ncol * single_states

    if padding:
        nsingle_padded = nsingle + 32 - (nsingle % 32)
        nvar = nsingle_padded + ncol * ncol * 21 * 32

        out_x_pair = np.zeros((21, ncol, 32, ncol), dtype='float64')
        out_x_pair[:21, :, :21, :] = np.transpose(x_pair[:, :, :21, :21], (3, 1, 2, 0))

        x = np.zeros((nvar, ), dtype='float64')
        x[:nsingle] = x_single[:, :single_states].T.reshape(-1)
        x[nsingle_padded:] = out_x_pair.reshape(-1)

    else:
        nvar = nsingle + ncol * ncol * 21 * 21

        out_x_pair = np.zeros((ncol, 21, ncol, 21), dtype='float64')
        out_x_pair[:, :21, :, :21] = np.transpose(x_pair[:, :, :21, :21], (0, 2, 1, 3))

        x = np.zeros((nvar, ), dtype='float64')

        x[:nsingle] = x_single[:, :single_states].reshape(-1)
        x[nsingle:] = out_x_pair.reshape(-1)

    return x
