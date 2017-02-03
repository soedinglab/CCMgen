import ccmpred.scoring
import numpy as np
import json
import gzip



def write_matrix(matfile, res, meta, disable_apc=False):

    print("Writing summed score matrix (with APC={0}) to {1}".format(not disable_apc, matfile))

    mat = ccmpred.scoring.frobenius_score(res.x_pair)
    if not disable_apc:
        mat = ccmpred.scoring.apc(mat)

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


