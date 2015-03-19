import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os.path

array_1d_float = npct.ndpointer(dtype=np.dtype('float64'), ndim=1, flags='CONTIGUOUS')
array_1d_uint32 = npct.ndpointer(dtype=np.dtype('uint32'), ndim=1, flags='CONTIGUOUS')
array_2d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=2, flags='CONTIGUOUS')

libtreecd = npct.load_library('libtreecd', os.path.join(os.path.dirname(__file__), '_build'))

libtreecd.mutate_along_tree.restype = None
libtreecd.mutate_along.argtypes = [
    array_1d_uint32,  # int32_t *n_children,
    array_1d_float,   # flt *branch_lengths,
    array_1d_float,   # flt *x,
    ctypes.c_uint32,  # uint32_t nvert,
    array_2d_char,    # uint8_t *seqs,
    ctypes.c_uint32,  # uint32_t ncol,
    ctypes.c_double   # flt mutation_rate
]


def mutate_along_tree():
    raise NotImplemented()
