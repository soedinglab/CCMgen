import numpy as np
import json
import gzip


def frobenius_score(x, squared=False):
    print("\nCompute contact map with frobenius norm (squared={0}).".format(squared))

    if squared:
        return np.sum(x * x, axis=(2, 3))
    else:
        return np.sqrt(np.sum(x * x, axis=(2, 3)))

def apc(cmat):
    print("\nApply Average Product Correction(APC).")
    mean = np.mean(cmat, axis=0)
    apc_term = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
    return cmat - apc_term
    #return apc_term

def compute_scaling_factor_eta(x_pair, ui, uij, nr_states, squared=True):
    scaling_factor_eta=1

    if squared:
        # compute scaling factor in matrix form
        # L=single_freq.shape[0]
        # upper_indices = np.triu_indices(L, k=1)

        x_pair_sq = x_pair * x_pair
        prod = x_pair_sq[:,:,:nr_states,:nr_states] * uij #(L,L,21,21) (L,L,21,21)
        scaling_factor_eta = np.sum(prod)
        #scaling_factor_eta = np.sum(prod[upper_indices[0], upper_indices[1], :, :])
        # print "scaling_factor_eta: ", scaling_factor_eta

        # ui_sq = ui * ui
        # uij_sq = np.transpose(np.multiply.outer(ui_sq, ui_sq), (0,2,1,3))
        # denominator = np.sum(uij_sq[upper_indices[0], upper_indices[1], :, :])
        sum_ui_sq = np.sum(ui * ui)
        denominator = sum_ui_sq * sum_ui_sq
        # print "denominator: ", denominator
        scaling_factor_eta /= denominator
        # print "scaling_factor_eta: ", scaling_factor_eta


        # compute scaling factor element-wise for testing
        # L = single_freq.shape[0]
        # scaling_factor_eta = 0
        # denominator=0
        # ui_sq = ui * ui
        # print "L: {0}".format(L)
        # for i in range(L):
        #     for j in range(L):
        #         for a in range(20):
        #             for b in range(20):
        #                 scaling_factor_eta += (x_pair[i,j,a,b]*x_pair[i,j,a,b]) * ui[i,a] * ui[j,b]
        #                 denominator += ui_sq[i,a] * ui_sq[j,b]
        # print "scaling_factor_eta_elementwise: ", scaling_factor_eta
        # print "denominator: ", denominator
        # print "scaling_factor_eta_elementwise: ", scaling_factor_eta / denominator

    else:

        #stefans entropy correction
        c_ij =  np.sqrt(np.sum(x_pair * x_pair, axis=(3,2)))
        e_ij =  np.sqrt(np.sum(uij, axis=(3,2)))

        scaling_factor_eta = np.sum(c_ij  * e_ij)
        denominator = np.sum(uij)
        scaling_factor_eta /= denominator

    return scaling_factor_eta


def compute_local_correction_couplings(single_freq, x_pair, Neff, lambda_w, entropy=False):
    print("\nApply local correction  with entropy={0}.".format(entropy))

    nr_states = 20

    N_factor = np.sqrt(Neff) * (1.0 / lambda_w)

    if entropy:
        ui = N_factor * single_freq[:, :nr_states] * np.log2(single_freq[:, :nr_states])
    else:
        ui = N_factor * single_freq[:, :nr_states] * (1 - single_freq[:, :nr_states])
    uij = np.transpose(np.multiply.outer(ui, ui), (0,2,1,3))

    ### compute scaling factor eta
    scaling_factor_eta = compute_scaling_factor_eta(x_pair, ui, uij, nr_states)

    xpair_sq = x_pair[:, :, :20, :20] * x_pair[:, :, :20, :20]

    couplings_corrected = xpair_sq - scaling_factor_eta * uij[:, :, :20, :20]

    return couplings_corrected

def compute_local_correction(single_freq, x_pair, Neff, lambda_w, mat, squared=True, entropy=False):

    print("\nApply local correction  with squared={0} and entropy={1}.".format(squared, entropy))

    scaling_factor_eta = 1.0
    nr_states = 20

    #debugging
    #N_factor = Neff / np.sqrt(Neff-1)
    N_factor = np.sqrt(Neff) * (1.0 / lambda_w)

    if entropy:
        ui = N_factor * single_freq[:, :nr_states] * np.log2(single_freq[:, :nr_states])
    else:
        ui = N_factor * single_freq[:, :nr_states] * (1 - single_freq[:, :nr_states])
    uij = np.transpose(np.multiply.outer(ui, ui), (0,2,1,3))


    ### compute scaling factor eta
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! squared !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    scaling_factor_eta = compute_scaling_factor_eta(x_pair, ui, uij, nr_states, squared=squared)

    if not squared:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #correction = np.sqrt(scaling_factor_eta * np.sum(uij, axis=(3,2)))
        correction = scaling_factor_eta * np.sqrt(np.sum(uij, axis=(3, 2)))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    else:
        correction =  scaling_factor_eta * np.sum(uij, axis=(3, 2))


    return scaling_factor_eta, mat - correction
    #return scaling_factor_eta, correction

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


