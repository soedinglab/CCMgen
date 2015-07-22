import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os.path

array_1d_float = npct.ndpointer(dtype=np.dtype('float64'), ndim=1, flags='CONTIGUOUS')
array_2d_char = npct.ndpointer(dtype=np.dtype('uint8'), ndim=2, flags='CONTIGUOUS')

libpll = npct.load_library('libpll', os.path.join(os.path.dirname(__file__), '_build'))

libpll.evaluate_pll.restype = ctypes.c_double
libpll.evaluate_pll.argtypes = [
    array_1d_float,    # *x
    array_1d_float,    # *g
    array_1d_float,    # *g2
    array_1d_float,    # *weights
    array_2d_char,      # *msa
    ctypes.c_uint32,    # ncol
    ctypes.c_uint32,    # nrow
]


def evaluate(x, g, g2, weights, msa):
    nrow, ncol = msa.shape
    fx = libpll.evaluate_pll(x, g, g2, weights, msa, ncol, nrow)
    return fx, g
