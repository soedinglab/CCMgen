import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os.path

array_1d_float = npct.ndpointer(dtype=np.dtype('float64'), ndim=1, flags='CONTIGUOUS')
array_1d_uint8 = npct.ndpointer(dtype=np.dtype('uint8'), ndim=1, flags='CONTIGUOUS')
array_1d_uint32 = npct.ndpointer(dtype=np.dtype('uint32'), ndim=1, flags='CONTIGUOUS')
array_1d_uint64 = npct.ndpointer(dtype=np.dtype('uint64'), ndim=1, flags='CONTIGUOUS')
array_2d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=2, flags='CONTIGUOUS')

libtreecd = npct.load_library('libtreecd', os.path.join(os.path.dirname(__file__), '_build'))

libtreecd.mutate_along_tree.restype = None
libtreecd.mutate_along_tree.argtypes = [
    array_1d_uint64,  # int32_t *n_children,
    array_1d_float,   # flt *branch_lengths,
    array_1d_float,   # flt *x,
    ctypes.c_uint64,  # uint32_t nvert,
    array_2d_char,    # uint8_t *seqs,
    ctypes.c_uint32,  # uint32_t ncol,
    ctypes.c_double   # flt mutation_rate
]

libtreecd.mutate_sequence.restype = None
libtreecd.mutate_sequence.argtypes = [
    array_1d_uint8,    # int32_t  seq,
    array_1d_float,   # flt *x,
    ctypes.c_uint16,    # uint32_t nmut,
    ctypes.c_uint32     # uint32_t ncol,
]

def mutate_sequence(parent_seq, x, nmut, ncol):
    seq = parent_seq.copy()
    libtreecd.mutate_sequence(seq, x, nmut, ncol)

    return seq

def mutate_along_tree(msa_sampled, n_children, branch_lengths, x, nvert, seq0, mutation_rate):
    msa_sampled[:, :] = 0
    msa_sampled[:seq0.shape[0], :] = seq0
    libtreecd.mutate_along_tree(n_children, branch_lengths, x, nvert, msa_sampled, seq0.shape[1], mutation_rate)

    return msa_sampled
