import msgpack
import numpy as np
import ccmpred.counts
import functools
from six import string_types, StringIO
import gzip


def stream_or_file(mode='r'):
    """Decorator for making a function accept either a filename or file-like object as a first argument"""

    def inner(fn):
        @functools.wraps(fn)
        def streamify(f, *args, **kwargs):
            if isinstance(f, string_types):

                open_fn = gzip.open if f.endswith(".gz") else open

                try:
                    fh = open_fn(f, mode)
                    res = fn(fh, *args, **kwargs)
                finally:
                    fh.close()

                return res
            else:
                return fn(f, *args, **kwargs)

        return streamify

    return inner

def calculate_Ni(msa, weights):
    single_counts = ccmpred.counts.single_counts(msa, weights)
    single_counts = single_counts[:,:20]

    return(single_counts.sum(1))

def calculate_Nij(msa, weights):
    pair_counts = ccmpred.counts.pair_counts(msa, weights)
    pair_counts = pair_counts[:,:,:20,:20]

    return(pair_counts.sum(3).sum(2))


@stream_or_file('wb')
def write_msgpack(outmsgpackfile, res, msa, weights, pair_freq, lambda_pair):

    Nij = calculate_Nij(msa, weights)

    out={
        # write lower triangular matrix row-wise
        # read in as upper triangular matrix column-wise in c++
        'N_ij': Nij[np.tril_indices(res.ncol, k=-1)].tolist(), #rowwise
    }

    model_prob = np.zeros((res.ncol *(res.ncol-1), 400))
    for i in range(res.ncol-1):
        for j in range(i + 1, res.ncol):

            #row-wise flattening
            pair_freq_ij = pair_freq[i, j, :20, :20].flatten()
            x_pair_ij    = res.x_pair[i, j, :20, :20].flatten()

            #row-wise ij
            model_prob[i * res.ncol + j] = pair_freq_ij - (x_pair_ij * lambda_pair / Nij[i,j])


    print( pair_freq[0, 1,0,0] - (res.x_pair[0, 1, 0,0] * lambda_pair / Nij[0,1]))
    print( pair_freq[0, 1,0,0])
    print(res.x_pair[0, 1, 0,0])
    print(lambda_pair)
    print(Nij[0,1])

    model_prob_flat = model_prob.flatten()

    if any(qijab < 0 for qijab in model_prob_flat):
        print("Warning: there are "+str(sum(model_prob_flat < 0))+" negative model probabilites")

        #hack: zet all negative model probabilities to zero
        model_prob_flat[model_prob_flat < 0] = 0

    out['q_ij'] = model_prob_flat.tolist()



    outmsgpackfile.write(msgpack.packb(out))