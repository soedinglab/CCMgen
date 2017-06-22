import numpy as np
import json
import gzip

def frobenius_score(x):
    print("\nCompute contact map with frobenius norm.")
    return np.sqrt(np.sum(x * x, axis=(2, 3)))


def apc(cmat):
    print("\nApply Average Product Correction(APC).")
    mean = np.mean(cmat, axis=0)
    apc_term = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
    return cmat - apc_term


def compute_matrix(x_pair, disable_apc=False):

    mat = frobenius_score(x_pair)
    if not disable_apc:
        mat = apc(mat)
    return mat


def write_matrix(matfile, mat, meta):

    print("\nWriting contact map to {0}".format(matfile))

    if matfile.endswith(".gz"):
        with gzip.open(matfile, 'wb') as f:
            np.savetxt(f, mat)
            f.write("#>META> ".encode("utf-8") + json.dumps(meta).encode("utf-8") + b"\n")
        f.close()
    else:
        np.savetxt(matfile, mat)
        with open(matfile,'a') as f:
            f.write("#>META> ".encode("utf-8") + json.dumps(meta).encode("utf-8") + b"\n")
        f.close()


