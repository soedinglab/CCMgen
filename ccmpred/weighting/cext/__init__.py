import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os.path

array_1d_double = npct.ndpointer(dtype=np.dtype('double'), ndim=1, flags='CONTIGUOUS')
array_2d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=2, flags='CONTIGUOUS')
array_2d_uint64 = npct.ndpointer(dtype=np.dtype('uint64'), ndim=2, flags='CONTIGUOUS')

libweighting = npct.load_library('libweighting', os.path.join(os.path.dirname(__file__), '_build'))

libweighting.count_ids.restype = None
libweighting.count_ids.argtypes = [
    array_2d_char,     # *msa
    array_2d_uint64,    # *n_ids
    ctypes.c_uint64,    # nrow
    ctypes.c_uint64,    # ncol
]

libweighting.calculate_weights_simple.restype = None
libweighting.calculate_weights_simple.argtypes = [
    array_2d_char,      # *msa
    array_1d_double,    # *weights
    ctypes.c_double,    # cutoff
    ctypes.c_uint64,    # nrow
    ctypes.c_uint64,    # ncol
]


def count_ids(msa):
    nrow = msa.shape[0]
    ids = np.zeros((nrow, nrow), dtype="uint64")
    libweighting.count_ids(msa, ids, *msa.shape)

    return ids + ids.T - np.diag(ids.diagonal())


def calculate_weights_simple(msa, cutoff, count_gaps=True):
    nrow = msa.shape[0]
    weights = np.zeros((nrow,), dtype='double')
    libweighting.calculate_weights_simple(msa, weights, cutoff, count_gaps, *msa.shape)

    return weights


if __name__ == '__main__':
    msa = np.array(
        [
            [0, 1, 2],
            [0, 3, 4],
            [0, 3, 2],
            [5, 6, 7]
        ],
        dtype=np.uint8
    )

    print(msa)
    print(count_ids(msa))
