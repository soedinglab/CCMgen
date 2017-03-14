import numpy as np
import ccmpred.raw


def init(ncol, centering=None):

    x_single = np.zeros((ncol, 21))
    x_pair = np.zeros((ncol, ncol, 21, 21))

    if(centering is not None):
        x_single = centering

    return ccmpred.raw.CCMRaw(ncol, x_single, x_pair, {})

