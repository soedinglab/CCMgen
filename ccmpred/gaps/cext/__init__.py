import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os.path

import ccmpred.counts

array_2d_float = npct.ndpointer(dtype=np.dtype('float64'), ndim=2, flags='CONTIGUOUS')
array_2d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=2, flags='CONTIGUOUS')
array_1d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=1, flags='CONTIGUOUS')

libgaps = npct.load_library('libgaps', os.path.join(os.path.dirname(__file__), '_build'))

libgaps.remove_gaps_probs.restype = None
libgaps.remove_gaps_probs.argtypes = [
    array_2d_float,    # *x
    array_2d_char,     # *msa
    ctypes.c_uint32,    # nrow
    ctypes.c_uint32,    # ncol
]


libgaps.remove_gaps_consensus.restype = None
libgaps.remove_gaps_consensus.argtypes = [
    array_2d_char,     # *msa
    array_1d_char,     # *consensus
    ctypes.c_uint32,    # nrow
    ctypes.c_uint32,    # ncol
]


def compute_consensus(msa, ignore_gaps=True):
    counts = ccmpred.counts.single_counts(msa)
    if ignore_gaps:
        counts = counts[:, :20]

    return np.argmax(counts, axis=1).astype('uint8')


def remove_gaps_probs(msa, probs):
    assert(probs.shape[0] == msa.shape[1])
    libgaps.remove_gaps_probs(np.ascontiguousarray(probs), msa, *msa.shape)
    return msa


def remove_gaps_consensus(msa, consensus=None):
    if not consensus:
        consensus = compute_consensus(msa)

    assert(consensus.shape[0] == msa.shape[1])
    libgaps.remove_gaps_consensus(msa, consensus, *msa.shape)

    return msa
