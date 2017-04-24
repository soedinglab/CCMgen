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


def compute_qij(freqs_pair, x_pair, lambda_pair, Nij):


    L = Nij.shape[0]

    #renormalize pair frequencies without gaps + reshape row-wise
    pair_freq = ccmpred.pseudocounts.degap(freqs_pair).reshape(L,L,400)

    #couplings without gaps + reshape row-wise
    x_pair_nogaps = x_pair[:,:,:20,:20].reshape(L,L, 400)

    #indices for i<j row-wise
    indices_triu = np.triu_indices(L, 1)

    #compute model probabilties for i<j
    model_prob = np.zeros((L, L, 400))
    model_prob[indices_triu] = pair_freq[indices_triu] - (x_pair_nogaps[indices_triu] * lambda_pair / Nij[:,:, np.newaxis][indices_triu])


    model_prob_flat = model_prob[indices_triu].flatten()  # row-wise upper triangular indices

    if any(np.isnan(qijab) for qijab in model_prob_flat):
        print("Warning: there are " + str(sum(np.isnan(model_prob_flat))) + " nan model probabilites")
        # hack: set nan (due to Nij=0) model probabilities to zero
        # model_prob_flat[np.isnan(model_prob_flat)] = 0

    return model_prob_flat

def get_nr_problematic_qij(freqs_pair, x_pair, lambda_pair, Nij, epsilon=0.01, verbose=0):

    L = Nij.shape[0]

    #renormalize pair frequencies without gaps + reshape row-wise
    pair_freq = ccmpred.pseudocounts.degap(freqs_pair).reshape(L,L,400)

    #couplings without gaps + reshape row-wise
    x_pair_nogaps = x_pair[:,:,:20,:20].reshape(L,L, 400)

    #indices for i<j row-wise
    indices_triu = np.triu_indices(L, 1)

    #compute model probabilties for i<j
    model_prob = np.zeros((L, L, 400))
    model_prob[indices_triu] = pair_freq[indices_triu] - (x_pair_nogaps[indices_triu] * lambda_pair / Nij[:,:, np.newaxis][indices_triu])




    problems = {}
    problems['qij_less_1'] = 0
    problems['qij_greater_1'] = 0
    problems['neg_qijab'] = 0


    threshold=1-epsilon
    if any(model_prob[indices_triu].sum(1) < threshold):
        nr_pairs_error = sum(model_prob[indices_triu].sum(1) < threshold)
        problems['qij_less_1'] += nr_pairs_error
        if verbose:
            print("Warning:  {0}/{1} pairs have sum(qij) < {2}. Minimal sum(q_ij): {3}".format(nr_pairs_error, len(indices_triu[0]), threshold, np.min(model_prob[indices_triu].sum(1))))
            indices = np.where(model_prob[indices_triu].sum(1) < threshold)[0]
            for ind in indices[:10]:
                i = indices_triu[0][ind]
                j = indices_triu[1][ind]
                print("e.g: i={0:<2} j={1:<2}: min(qij)={2:<20} sum(qij)={3:<20} sum(pair_freq)={4:<20} sum(x_pair)={5:<20} N_ij={6}".format(
                    i, j, min(model_prob[i,j]), sum(model_prob[i,j]), sum(pair_freq[i,j].flatten()), sum(x_pair_nogaps[i,j].flatten()), Nij[i,j])
                )

    threshold=1+epsilon
    if any(model_prob[indices_triu].sum(1) > threshold):
        nr_pairs_error = sum(model_prob[indices_triu].sum(1) > threshold)
        problems['qij_greater_1'] += nr_pairs_error
        if verbose:
            print("Warning:  {0}/{1} pairs have sum(qij) > {2}. Max sum(q_ij): {3}".format(nr_pairs_error, len(indices_triu[0]), threshold, np.max(model_prob[indices_triu].sum(1))))
            indices = np.where(model_prob[indices_triu].sum(1) > threshold)[0]
            for ind in indices[:10]:
                i = indices_triu[0][ind]
                j = indices_triu[1][ind]
                print("e.g: i={0:<2} j={1:<2}: min(qij)={2:<20} sum(qij)={3:<20} sum(pair_freq)={4:<20} sum(x_pair)={5:<20} N_ij={6}".format(
                    i, j, min(model_prob[i,j]), sum(model_prob[i,j]), sum(pair_freq[i,j].flatten()), sum(x_pair_nogaps[i,j].flatten()), Nij[i,j])
                )

    if any(model_prob[indices_triu].min(1) < 0):
        nr_pairs_error = sum(model_prob[indices_triu].min(1) < 0)
        problems['neg_qijab'] += nr_pairs_error
        if verbose:
            print("Warning:  {0}/{1} pairs have min(q_ij) < 0. Minimal min(q_ij): {2}".format(nr_pairs_error, len(indices_triu[0]), np.min(model_prob[indices_triu].min(1))))
            indices = np.where(model_prob[indices_triu].min(1) < 0)[0]
            for ind in indices[:10]:
                i = indices_triu[0][ind]
                j = indices_triu[1][ind]
                print("e.g: i={0:<2} j={1:<2}: min(qij)={2:<20} sum(qij)={3:<20} sum(pair_freq)={4:<20} sum(x_pair)={5:<20} N_ij={6}".format(
                    i, j, min(model_prob[i,j]), sum(model_prob[i,j]), sum(pair_freq[i,j].flatten()), sum(x_pair_nogaps[i,j].flatten()), Nij[i,j])
                )


    return problems


@stream_or_file('wb')
def write_msgpack(outmsgpackfile, res, weights, freqs, lambda_pair):

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

    #compute q_ij
    model_prob_flat = compute_qij(freqs_pair, res.x_pair, lambda_pair, Nij)

    out['q_ij'] = model_prob_flat.tolist()
    outmsgpackfile.write(msgpack.packb(out))