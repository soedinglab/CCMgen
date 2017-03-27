import msgpack
import numpy as np
import ccmpred.counts
import ccmpred.pseudocounts
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


@stream_or_file('wb')
def write_msgpack(outmsgpackfile, res, weights, msa, freqs, lambda_pair):

    out={}

    neff = np.sum(weights)

    freqs_single, freqs_pair = freqs

    # ENFORCE NO PSEUDO COUNTS
    # freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, ccmpred.pseudocounts.no_pseudocounts, pseudocount_n_single=0, pseudocount_n_pair=0, remove_gaps=False)
    # freqs_single, freqs_pair = freqs


    msa_counts_pair = freqs_pair * neff
    # reset gap counts
    msa_counts_pair[:, :, :, 20] = 0
    msa_counts_pair[:, :, 20, :] = 0

    #non_gapped counts
    Nij = msa_counts_pair.sum(3).sum(2)



    # write lower triangular matrix row-wise
    # read in as upper triangular matrix column-wise in c++
    out['N_ij'] = Nij[np.tril_indices(res.ncol, k=-1)].tolist() #rowwise


    #renormalize pair frequencies without gaps + reshape row-wise
    pair_freq = ccmpred.pseudocounts.degap(freqs_pair).reshape(res.ncol,res.ncol,400)

    #couplings without gaps + reshape row-wise
    x_pair_nogaps = res.x_pair[:,:,:20,:20].reshape(res.ncol,res.ncol, 400)

    #indices for i<j row-wise
    indices_triu = np.triu_indices(res.ncol, 1)

    #compute model probabilties for i<j
    model_prob = np.zeros((res.ncol, res.ncol, 400))
    model_prob[indices_triu] = pair_freq[indices_triu] - (x_pair_nogaps[indices_triu] * lambda_pair / Nij[:,:, np.newaxis][indices_triu])

    if any(model_prob[indices_triu].sum(1) < 0.99):
        print("Warning: there are "+str(sum(model_prob[indices_triu].sum(1) < 0.99))+" pairs with sum(model probabilites) < 0.99. Minimal sum(q_ij): " +str(np.min(model_prob[indices_triu].sum(1))))
        indices = np.where(model_prob[indices_triu].sum(1) < 0.99)[0]
        for ind in indices:
            i = indices_triu[0][ind]
            j = indices_triu[1][ind]
            print('e.g: ', i, j, sum(model_prob[i,j]), sum(pair_freq[i,j].flatten()), sum(x_pair_nogaps[i,j].flatten()), Nij[i,j])

    if any(model_prob[indices_triu].sum(1) > 1.01):
        print("Warning: there are "+str(sum(model_prob[indices_triu].sum(1) > 1.01))+" pairs with sum(model probabilites) > 1.01. Max sum(q_ij): " +str(np.max(model_prob[indices_triu].sum(1))))
        indices = np.where(model_prob[indices_triu].sum(1) > 1.01)[0]
        for ind in indices:
            i = indices_triu[0][ind]
            j = indices_triu[1][ind]
            print('e.g: ', i, j, sum(model_prob[i,j]), sum(pair_freq[i,j].flatten()), sum(x_pair_nogaps[i,j].flatten()), Nij[i,j])

    model_prob_flat = model_prob[indices_triu].flatten() #row-wise upper triangular indices

    if any(qijab < 0 for qijab in model_prob_flat):
        print("Warning: there are "+str(sum(model_prob_flat < 0))+" negative model probabilites. Min qijab: " + str(np.min(model_prob_flat)))

        #hack: set all negative model probabilities to zero
        #model_prob_flat[model_prob_flat < 0] = 0

    if any(np.isnan(qijab) for qijab in model_prob_flat):
        print("Warning: there are "+str(sum(np.isnan(model_prob_flat)))+" nan model probabilites")

        #hack: set nan (due to Nij=0) model probabilities to zero
        #model_prob_flat[np.isnan(model_prob_flat)] = 0

    out['q_ij'] = model_prob_flat.tolist()


    outmsgpackfile.write(msgpack.packb(out))