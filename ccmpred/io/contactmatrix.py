import numpy as np
import json
import gzip
import os

def frobenius_score(x):
    """
    Compute frobenius norm of couplig matrix

    :param x:   pair potentials of dimension [ L x L x 20 x 20 ]
    :param squared:
    :return:
    """
    print("\nCompute contact map using frobenius norm of couplings.")

    return np.sqrt(np.sum(x * x, axis=(2, 3)))

def apc(cmat):
    """
    Compute average product correction (APC) according to Dunn et al 2004

    :param cmat: contact matrix
    :return:    corrected contact matrix
    """
    print("\nApply Average Product Correction (APC).")

    mean = np.mean(cmat, axis=0)
    apc_term = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)

    return cmat - apc_term

def compute_scaling_factor_eta(x_pair, uij, nr_states, squared=True):
    """
    Set the strength of the entropy correction by optimization eta with least squares

    Minimize sum_i,j sum_a,b (w_ijab^2 -  eta * u_ia * u_jb)^2

    :param x_pair:
    :param ui:
    :param uij:
    :param nr_states:
    :param squared:
    :return:
    """

    if squared:

        x_pair_sq = x_pair * x_pair
        prod = x_pair_sq[:,:,:nr_states,:nr_states] * uij #(L,L,21,21) (L,L,21,21)
        scaling_factor_eta = np.sum(prod)

        denominator = np.sum(uij * uij)
        scaling_factor_eta /= denominator

    else:

        #According to Stefan's CCMgen paper
        c_ij =  np.sqrt(np.sum(x_pair * x_pair, axis=(3,2)))
        e_ij =  np.sqrt(np.sum(uij, axis=(3,2)))

        scaling_factor_eta = np.sum(c_ij  * e_ij)
        denominator = np.sum(uij)
        scaling_factor_eta /= denominator

    return scaling_factor_eta

def compute_local_correction(single_freq, x_pair, Neff, lambda_w, mat, squared=True, entropy=False, nr_states=20):

    print("\nApply entropy correction (using {0} states).".format(nr_states))


    #correct for fractional counts
    N_factor = np.sqrt(Neff) * (1.0 / lambda_w)

    if entropy:
        ui = N_factor * single_freq[:, :nr_states] * np.log2(single_freq[:, :nr_states])
    else:
        ui = N_factor * single_freq[:, :nr_states] * (1 - single_freq[:, :nr_states])
    uij = np.transpose(np.multiply.outer(ui, ui), (0,2,1,3))

    ### compute scaling factor eta
    scaling_factor_eta = compute_scaling_factor_eta(x_pair, uij, nr_states, squared=squared)

    if not squared:
        correction = scaling_factor_eta * np.sqrt(np.sum(uij, axis=(3, 2)))
    else:
        correction = scaling_factor_eta * np.sum(uij, axis=(3, 2))

    return scaling_factor_eta, mat - correction

def compute_joint_entropy_correction(pair_freq, neff, lambda_w, braw_x_pair, nr_states = 21):

    print("\nApply joint entropy correction (using {0} states).".format(nr_states))

    N_factor = neff / (lambda_w * lambda_w)

    joint_entropy = - np.sum(
        pair_freq[:, :, :nr_states, :nr_states] * np.log2(pair_freq[:, :, :nr_states, :nr_states]),
        axis=(3, 2)
    )
    uij = N_factor * joint_entropy
    c_ij = frobenius_score(braw_x_pair)

    ### compute scaling factor eta
    scaling_factor = np.sum(c_ij * uij) / np.sum(uij * uij)

    corrected_mat = c_ij - scaling_factor * uij

    return scaling_factor, corrected_mat

def compute_corrected_mat_sergey_style(pair_freq, braw_x_pair, nr_states = 21):

    print("\nApply sergeys joint entropy correction (using {0} states).".format(nr_states))

    joint_entropy = - np.sum(
        pair_freq[:, :, :nr_states, :nr_states] * np.log2(pair_freq[:, :, :nr_states, :nr_states]),
        axis=(3, 2)
    )
    correction  = joint_entropy + np.exp(-joint_entropy)

    corrected_braw = braw_x_pair[:, :, :nr_states, :nr_states] / correction[:,:, np.newaxis, np.newaxis]

    mat = frobenius_score(corrected_braw)

    return(mat)

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

def read_matrix(matfile):
    """
    Read matrix file
    :param mat_file: path to matrix file
    :return: matrix
    """

    if not os.path.exists(matfile):
        raise IOError("Matrix File " + str(matfile) + "cannot be found. ")


    ### Read contact map (matfile can also be compressed file)
    mat = np.genfromtxt(matfile, comments="#")

    ### Read meta data from mat file
    meta = {}
    with open(matfile) as f:
        for line in f:
            if '#>META>' in line:
                meta = json.loads(line.split("> ")[1])

    if len(meta) == 0:
        print(str(matfile) + " does not contain META info. (Line must start with #META!)")

    return mat, meta