import ccmpred
import numpy as np
import json
import datetime


def create(opt, regularization, msa, weights, f, fx, algret, alg):
    meta={}

    meta['version'] = ccmpred.__version__
    meta['method']  = 'ccmpred-py'

    meta['workflow'] = []
    meta['workflow'].append({})
    meta['workflow'][0]['timestamp'] =  str(datetime.datetime.now())


    meta['workflow'][0]['parameters'] = {}
    meta['workflow'][0]['parameters']['regularization'] = {}
    meta['workflow'][0]['parameters']['regularization']['regularization_type'] = 'l2_centered_v'
    meta['workflow'][0]['parameters']['regularization']['regularization_scaling'] = opt.scaling
    meta['workflow'][0]['parameters']['regularization']['lambda_single'] = regularization.lambda_single
    meta['workflow'][0]['parameters']['regularization']['lambda_pair'] = regularization.lambda_pair
    meta['workflow'][0]['parameters']['regularization']['lambda_pair_factor'] = regularization.lambda_pair / (msa.shape[1] - 1)


    meta['workflow'][0]['parameters']['msafile'] = {}
    meta['workflow'][0]['parameters']['msafile']['neff'] = np.sum(weights)
    meta['workflow'][0]['parameters']['msafile']['nrow'] = msa.shape[0]
    meta['workflow'][0]['parameters']['msafile']['ncol'] = msa.shape[1]
    meta['workflow'][0]['parameters']['msafile']['file'] = opt.alnfile

    meta['workflow'][0]['parameters']['gaps'] = {}
    meta['workflow'][0]['parameters']['gaps']["msa-max_gap_ratio"] = opt.max_gap_ratio
    meta['workflow'][0]['parameters']['gaps']["wt-ignore_gaps"] = opt.ignore_gaps


    meta['workflow'][0]['parameters']['pseudocounts'] = {}
    meta['workflow'][0]['parameters']['pseudocounts']['pseudocount_type'] = opt.pseudocounts[0].__name__
    meta['workflow'][0]['parameters']['pseudocounts']['pseudocount_n_single'] = opt.pseudocounts[1]
    if opt.pseudocount_pair_count:
        meta['workflow'][0]['parameters']['pseudocounts']['pseudocount_n_pair'] = opt.pseudocount_pair_count
    else:
        meta['workflow'][0]['parameters']['pseudocounts']['pseudocount_n_pair'] = opt.pseudocounts[1]

    meta['workflow'][0]['parameters']['optimization']={}
    meta['workflow'][0]['parameters']['optimization']['method'] = opt.algorithm
    meta['workflow'][0]['parameters']['optimization']['objfun'] = opt.objfun.__name__

    if (opt.algorithm) == 'conjugate_gradients':
        meta['workflow'][0]['parameters']['optimization']['wolfe'] = alg.wolfe
        meta['workflow'][0]['parameters']['optimization']['alpha_mul'] = alg.alpha_mul
        meta['workflow'][0]['parameters']['optimization']['max_linesearch'] = alg.max_linesearch
        meta['workflow'][0]['parameters']['optimization']['ftol'] = alg.ftol
        meta['workflow'][0]['parameters']['optimization']['epsilon'] = alg.epsilon
        meta['workflow'][0]['parameters']['optimization']['convergence_prev'] = alg.convergence_prev


    meta['workflow'][0]['parameters']['apc']  = not opt.disable_apc
    meta['workflow'][0]['parameters']['weighting'] =  opt.weight.__name__
    if opt.initrawfile:
        meta['workflow'][0]['initrawfile'] = opt.initrawfile


    meta['workflow'][0]['results'] = {}
    #meta['workflow'][0]['results']['num_iterations']
    meta['workflow'][0]['results']['matfile'] = opt.matfile
    meta['workflow'][0]['results']['opt_code'] = algret['code']
    meta['workflow'][0]['results']['opt_message'] = algret['message']
    meta['workflow'][0]['results']['fx_final'] = fx
    if opt.cd_alnfile and hasattr(f, 'msa_sampled'):
        meta['workflow'][0]['results']['cd_alignmentfile'] =   opt.cd_alnfile
    if opt.outrawfile:
        meta['workflow'][0]['results']['rawfile'] = opt.outrawfile
    if opt.outmsgpackfile:
        meta['workflow'][0]['results']['msgpackfile'] = opt.outmsgpackfile
    if opt.outmodelprobmsgpackfile:
        meta['workflow'][0]['results']['modelprobmsgpackfile'] = opt.outmodelprobmsgpackfile

    return meta

