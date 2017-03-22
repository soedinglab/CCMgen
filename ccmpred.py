#!/usr/bin/env python
import argparse
import numpy as np
import sys
import json
import os

import ccmpred.metadata
import ccmpred.weighting
import ccmpred.scoring
import ccmpred.pseudocounts
import ccmpred.initialise_potentials
import ccmpred.raw
import ccmpred.logo
import ccmpred.io
import ccmpred.centering
import ccmpred.regularization
import ccmpred.model_probabilities
import ccmpred.gaps
import ccmpred.sanity_check

import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.objfun.treecd as treecd

import ccmpred.algorithm.gradient_descent as gd
import ccmpred.algorithm.conjugate_gradients as cg
import ccmpred.algorithm.numdiff as nd
import ccmpred.algorithm.adam as ad

EPILOG = """
CCMpred is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods. From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model with 21 states for amino acids and gaps. The L2 norms of the pairwise coupling potentials will be written to the output matfile.
"""

REG_L2_SCALING= {
    "L" : lambda msa : msa.shape[1] - 1,
    "diversity" : lambda msa: msa.shape[1] / np.sqrt(msa.shape[0]),
    "1": lambda msa: 1
}

ALGORITHMS = {
    "conjugate_gradients": lambda opt: cg.conjugateGradient(maxit=opt.maxit, epsilon=opt.epsilon, convergence_prev=opt.convergence_prev),
    "gradient_descent": lambda opt: gd.gradientDescent(maxit=opt.maxit, alpha0=opt.alpha0, alpha_decay=opt.alpha_decay, epsilon=opt.epsilon, convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping),
    "adam": lambda opt: ad.Adam(maxit=opt.maxit, learning_rate=opt.learning_rate, momentum_estimate1=opt.mom1, momentum_estimate2=opt.mom2, noise=1e-7, epsilon=opt.epsilon, convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping, decay=opt.adam_decay),
    "numerical_differentiation": lambda opt: nd.numDiff(maxit=opt.maxit, epsilon=opt.epsilon)
}


class CDAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        gibbs_steps, n_sequences = values

        if n_sequences < 1:
            n_sequences = 1

        namespace.objfun_kwargs = {'gibbs_steps':gibbs_steps, 'persistent': False, 'n_sequences': n_sequences}
        namespace.objfun = cd.ContrastiveDivergence


class CDPLLAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        gibbs_steps, n_sequences = values

        if n_sequences < 1:
            n_sequences = 1


        namespace.objfun_kwargs = {'gibbs_steps':gibbs_steps, 'persistent': False, 'n_sequences': n_sequences, 'pll': True}
        namespace.objfun = cd.ContrastiveDivergence

class PCDAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        gibbs_steps, n_sequences = values

        if n_sequences < 1:
            n_sequences = 1


        namespace.objfun_kwargs = {'gibbs_steps':gibbs_steps, 'persistent': True, 'n_sequences': n_sequences}
        namespace.objfun = cd.ContrastiveDivergence

class TreeCDAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import Bio.Phylo
        treefile, seq0file = values

        tree = Bio.Phylo.read(treefile, "newick")
        seq0, id0 = ccmpred.io.alignment.read_msa(seq0file, parser.values.aln_format, return_identifiers=True)


        namespace.objfun_args = [tree, seq0, id0]
        namespace.objfun = treecd.TreeContrastiveDivergence


class RegL2Action(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        lambda_single, lambda_pair = values

        namespace.regularization = lambda msa, centering, scaling: ccmpred.regularization.L2(lambda_single, lambda_pair * scaling, centering)

class StoreConstParametersAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, arg_default=None, default=None, **kwargs):
        self.arg_default = arg_default
        default = (default, arg_default)
        super(StoreConstParametersAction, self).__init__(option_strings, dest, nargs=nargs, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None or values == self.const:
            values = self.arg_default
        setattr(namespace, self.dest, (self.const, values))


def parse_args():
    parser = argparse.ArgumentParser(description="Recover direct couplings from a multiple sequence alignment", epilog=EPILOG)

    parser.add_argument("-i", "--init-from-raw", dest="initrawfile", default=None, help="Init potentials from raw file")
    parser.add_argument("-r", "--write-raw", dest="outrawfile", default=None, help="Write potentials to raw file")
    parser.add_argument("-b", "--write-msgpack", dest="outmsgpackfile", default=None, help="Write potentials to MessagePack file")
    parser.add_argument("-m", "--write-modelprob-msgpack", dest="outmodelprobmsgpackfile", default=None, help="Write model probabilities as MessagePack file")
    parser.add_argument("-A", "--disable_apc",  dest="disable_apc", action="store_true", default=False, help="Disable average product correction (APC)")
    parser.add_argument("--aln-format", dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--no-logo", dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo")
    parser.add_argument("-p", "--plot_opt_progress", dest="plot_opt_progress", default=False, action="store_true", help="Plot optimization progress")


    parser.add_argument("alnfile", help="Input alignment file to use")
    parser.add_argument("matfile", help="Output matrix file to write")

    grp_of = parser.add_argument_group("Objective Functions")
    grp_of.add_argument("--ofn-pll", dest="objfun", action="store_const", const=pll.PseudoLikelihood, default=pll.PseudoLikelihood, help="Use pseudo-log-likelihood (default)")
    grp_of.add_argument("--ofn-cd",  dest="objfun", action=CDAction, metavar=("GIBBS_STEPS", "N_SEQUENCES"), nargs=2, type=int, help="Use Contrastive Divergence with GIBBS_STEPS of Gibbs sampling steps for sequences and sample N_SEQUENCES sequences")
    grp_of.add_argument("--ofn-cdpll",  dest="objfun", action=CDPLLAction, metavar=("GIBBS_STEPS", "N_SEQUENCES"), nargs=2, type=int, help="Use Contrastive Divergence with GIBBS_STEPS of Gibbs sampling steps for sequences and sample an alignment with N_SEQUENCES sequences")
    grp_of.add_argument("--ofn-pcd", dest="objfun", action=PCDAction, metavar=("GIBBS_STEPS", "N_SEQUENCES"), nargs=2, type=int, help="Use PERSISTENT Contrastive Divergence with GIBBS_STEPS of Gibbs sampling steps for sequences and sample an alignment with N_SEQUENCES sequences")
    grp_of.add_argument("--ofn-tree-cd", action=TreeCDAction, metavar=("TREEFILE", "ANCESTORFILE"), nargs=2, type=str, help="Use Tree-controlled Contrastive Divergence, loading tree data from TREEFILE and ancestral sequence data from ANCESTORFILE")

    grp_al = parser.add_argument_group("Algorithms")
    grp_al.add_argument("--alg-cg", dest="algorithm", action="store_const", const='conjugate_gradients', default='conjugate_gradients', help='Use conjugate gradients (default)')
    grp_al.add_argument("--alg-gd", dest="algorithm", action="store_const", const='gradient_descent', help='Use gradient descent')
    grp_al.add_argument("--alg-nd", dest="algorithm", action="store_const", const='numerical_differentiation', help='Debug gradients with numerical differentiation')
    grp_al.add_argument("--alg-ad", dest="algorithm", action="store_const", const='adam', help='Use Adam')
    grp_al.add_argument("--gd-alpha0",              dest="alpha0",              default=5e-3,   type=float, help="alpha0 parameter for gradient descent")
    grp_al.add_argument("--gd-alpha_decay",         dest="alpha_decay",         default=1e1,    type=float, help="alpha_decay for gradient descent")
    grp_al.add_argument("--ad-learning_rate",       dest="learning_rate",       default=1e-3,   type=float, help="learning rate for adam")
    grp_al.add_argument("--ad-mom1",                dest="mom1",                default=0.9,    type=float, help="momentum 1 for adam")
    grp_al.add_argument("--ad-mom2",                dest="mom2",                default=0.999,  type=float, help="momentum 2 for adam")
    grp_al.add_argument("--ad-decay",               dest="adam_decay",          action="store_true", default=False,  help="decay adam learning rate with 1/sqrt(it)")

    grp_con = parser.add_argument_group("Convergence Criteria")
    grp_con.add_argument("--epsilon",                dest="epsilon",             default=1e-5,   type=float, help="Set convergence criterion: converged when relative change in f (or xnorm) in last CONVERGENCE_PREV iterations < EPSILON [default: 0.01]")
    grp_con.add_argument("--convergence_prev",       dest="convergence_prev",    default=5,      type=int,   help="Set convergence_prev parameter for convergence criterion [default: 5]")
    grp_con.add_argument("--early_stopping",         dest="early_stopping",      default=False,  action="store_true",  help="Apply convergence criteria instead of only maxit")
    grp_con.add_argument("--maxit",                   dest="maxit",              default=500,    type=int, help="Specify the maximum number of iterations [default: %(default)s]")


    grp_wt = parser.add_argument_group("Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const=ccmpred.weighting.weights_simple, default=ccmpred.weighting.weights_simple, help='Use simple weighting (default)')
    grp_wt.add_argument("--wt-henikoff",        dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff, help='Use simple Henikoff weighting')
    grp_wt.add_argument("--wt-henikoff_pair",   dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff_pair, help='Use Henikoff pair weighting ')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const=ccmpred.weighting.weights_uniform, help='Use uniform weighting')

    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-l2", dest="regularization", action=RegL2Action, type=float, nargs=2, metavar=("LAMBDA_SINGLE", "LAMBDA_PAIR"), default=lambda msa, centering, scaling: ccmpred.regularization.L2(10, 0.2 * scaling, centering), help='Use L2 regularization with coefficients LAMBDA_SINGLE, LAMBDA_PAIR * SCALING;  (default: 10 0.2)')
    grp_rg.add_argument("--reg-l2-scale_by_L",      dest="scaling", action="store_const", const="L", default="L", help=" LAMBDA_PAIR * (L-1) (default)")
    grp_rg.add_argument("--reg-l2-scale_by_div",    dest="scaling", action="store_const", const="diversity", help="LAMBDA_PAIR * (L/sqrt(N))")
    grp_rg.add_argument("--reg-l2-noscaling",       dest="scaling", action="store_const", const="1", help="LAMBDA_PAIR")
    grp_rg.add_argument("--center-v",               dest="center_v", action="store_true", default=False, help="Gaussian prior for single emissions centered at v*.")

    grp_gp = parser.add_argument_group("Gap Treatment")
    grp_gp.add_argument("--max_gap_ratio",  dest="max_gap_ratio", default=100, type=int, help="Remove alignment positions with >x% gaps [default: %(default)s  = no removal]")
    grp_gp.add_argument("--wt-ignore-gaps", dest="ignore_gaps", action="store_true", default=False, help="Do not count gaps as identical amino acids during reweighting of sequences.")

    grp_pc = parser.add_argument_group("Pseudocounts")
    grp_pc.add_argument("--pc-submat",      dest="pseudocounts", action=StoreConstParametersAction, default=ccmpred.pseudocounts.substitution_matrix_pseudocounts, const=ccmpred.pseudocounts.substitution_matrix_pseudocounts, nargs="?", metavar="N", type=int, arg_default=1, help="Use N substitution matrix pseudocounts (default) (by default, N=1)")
    grp_pc.add_argument("--pc-constant",    dest="pseudocounts", action=StoreConstParametersAction, const=ccmpred.pseudocounts.constant_pseudocounts,   metavar="N", nargs="?", type=int, arg_default=1, help="Use N constant pseudocounts (by default, N=1)")
    grp_pc.add_argument("--pc-uniform",     dest="pseudocounts", action=StoreConstParametersAction, const=ccmpred.pseudocounts.uniform_pseudocounts,    metavar="N", nargs="?", type=int, arg_default=1, help="Use N uniform pseudocounts, e.g 1/21 (by default, N=1)")
    grp_pc.add_argument("--pc-none",        dest="pseudocounts", action="store_const", const=[ccmpred.pseudocounts.no_pseudocounts, 0], help="Use no pseudocounts")
    grp_pc.add_argument("--pc-pair-count",  dest="pseudocount_pair_count", default=None, type=int, help="Specify a separate number of pseudocounts for pairwise frequencies (default: use same as single counts)")

    grp_db = parser.add_argument_group("Debug Options")
    grp_db.add_argument("--write-trajectory", dest="trajectoryfile", default=None, help="Write trajectory to files with format expression")
    grp_db.add_argument("--write-cd-alignment", dest="cd_alnfile", default=None, metavar="ALNFILE", help="Write PSICOV-formatted sampled alignment to ALNFILE")
    grp_db.add_argument("-c", "--compare-to-raw", dest="comparerawfile", default=None, help="Compare potentials to raw file")
    grp_db.add_argument("--dev-center-v", dest="dev_center_v", action="store_true", default=False, help="Use same settings as in c++ dev-center-v version")
    grp_db.add_argument("--ccmpred-vanilla", dest="vanilla", action="store_true", default=False, help="Use same settings as in default c++ CCMpred")
    grp_db.add_argument("--only_model_prob", dest="only_model_prob", action="store_true", default=False, help="Only compute model probabilties and do not optimize (-i must be specified!).")

    args = parser.parse_args()

    if args.cd_alnfile and args.objfun not in (cd.ContrastiveDivergence, treecd.TreeContrastiveDivergence):
        parser.error("--write-cd-alignment is only supported for (tree) contrastive divergence!")


    return args


def main():

    opt = parse_args()

    if opt.logo:
        ccmpred.logo.logo()

    msa = ccmpred.io.alignment.read_msa(opt.alnfile, opt.aln_format)
    msa, gapped_positions = ccmpred.gaps.remove_gapped_positions(msa, opt.max_gap_ratio)

    weights = opt.weight(msa, opt.ignore_gaps)

    protein=os.path.basename(opt.alnfile).split(".")[0]
    print("Alignment for protein {0} (L={1}) has {2} sequences and Neff(HHsuite-like)={3}".format(protein, msa.shape[1], msa.shape[0], ccmpred.pseudocounts.get_neff(msa)))
    print("Reweighted sequences to Sum(weights)={0:g} using {1} and ignore_gaps={2})".format(np.sum(weights), opt.weight.__name__, opt.ignore_gaps))


    if not hasattr(opt, "objfun_args"):
        opt.objfun_args = []

    if not hasattr(opt, "objfun_kwargs"):
        opt.objfun_kwargs = {}

    if opt.dev_center_v:
        freqs = ccmpred.pseudocounts.calculate_frequencies_dev_center_v(msa, weights)
    else:
        freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, opt.pseudocounts[0], pseudocount_n_single=opt.pseudocounts[1], pseudocount_n_pair=opt.pseudocount_pair_count, remove_gaps=False)


    #setup regularization properties
    centering   = ccmpred.centering.center_zero(freqs)

    if opt.center_v or opt.dev_center_v:
        centering = ccmpred.centering.center_v(freqs)


    scaling = REG_L2_SCALING[opt.scaling](msa)
    regularization = opt.regularization(msa, centering, scaling)

    init_single_potentials        = centering

    if opt.vanilla:
        freqs_for_init = ccmpred.pseudocounts.calculate_frequencies_vanilla(msa)
        init_single_potentials = ccmpred.centering.center_vanilla(freqs_for_init)
        #besides initialisation and regularization, there seems to be another difference in gradient calculation between CCMpred vanilla and CCMpred-dev-center-v
        #furthermore initialisation does NOT assure sum_a(v_ia) == 1

    #default initialisation of parameters
    raw_init = ccmpred.initialise_potentials.init(msa.shape[1], init_single_potentials)


    if opt.initrawfile:
        raw_init = ccmpred.raw.parse(opt.initrawfile)
        #only compute model frequencies and exit
        if opt.only_model_prob and opt.outmodelprobmsgpackfile:
            print("Writing msgpack-formatted model probabilties to {0}".format(opt.outmodelprobmsgpackfile))
            if opt.dev_center_v:
                freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, ccmpred.pseudocounts.constant_pseudocounts, pseudocount_n_single=1, pseudocount_n_pair=1, remove_gaps=True)
            ccmpred.model_probabilities.write_msgpack(opt.outmodelprobmsgpackfile, raw_init, weights, msa, freqs, regularization.lambda_pair)
            sys.exit(0)


    #initialise objective function
    f  =  opt.objfun(msa, freqs, weights, raw_init, regularization, *opt.objfun_args, **opt.objfun_kwargs)
    x0 = f.x0



    if opt.comparerawfile:
        craw = ccmpred.raw.parse(opt.comparerawfile)
        f.compare_raw = craw

    f.trajectory_file = opt.trajectoryfile

    alg = ALGORITHMS[opt.algorithm](opt)

    #whether to plot progress of optimization
    plotfile=None
    if(opt.plot_opt_progress):
        plotfile=os.path.dirname(opt.matfile) + "/" + protein + ".opt_progress.html"


    print("Will optimize {0} {1} variables wrt {2} and {3}".format(x0.size, x0.dtype, f, f.regularization))
    print("Optimizer: {0}".format(alg))
    fx, x, algret = alg.minimize(f, x0, plotfile)


    condition = "Finished" if algret['code'] >= 0 else "Exited"
    print("\n{0} with code {code} -- {message}".format(condition, **algret))

    meta = ccmpred.metadata.create(opt, regularization, msa, weights, f, fx, algret, alg)
    res = f.finalize(x, meta)

    if opt.cd_alnfile and hasattr(f, 'msa_sampled'):
        print("Writing sampled alignment to {0}".format(opt.cd_alnfile))
        msa_sampled = f.msa_sampled

        with open(opt.cd_alnfile, "w") as f:
            ccmpred.io.alignment.write_msa_psicov(f, msa_sampled)

    if opt.max_gap_ratio < 100:
        ccmpred.gaps.backinsert_gapped_positions(res, gapped_positions)

    if opt.outrawfile:
        print("Writing raw-formatted potentials to {0}".format(opt.outrawfile))
        ccmpred.raw.write_oldraw(opt.outrawfile, res)

    if opt.outmsgpackfile:
        print("Writing msgpack-formatted potentials to {0}".format(opt.outmsgpackfile))
        ccmpred.raw.write_msgpack(opt.outmsgpackfile, res)

    if opt.outmodelprobmsgpackfile:
        print("Writing msgpack-formatted model probabilties to {0}".format(opt.outmodelprobmsgpackfile))
        if opt.max_gap_ratio < 100:
            msa = ccmpred.io.alignment.read_msa(opt.alnfile, opt.aln_format)
            freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, opt.pseudocounts[0], pseudocount_n_single=opt.pseudocounts[1], pseudocount_n_pair=opt.pseudocount_pair_count)
        if opt.dev_center_v:
            freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, ccmpred.pseudocounts.constant_pseudocounts, pseudocount_n_single=1, pseudocount_n_pair=1, remove_gaps=True)

        ccmpred.model_probabilities.write_msgpack(opt.outmodelprobmsgpackfile, res, weights, msa, freqs, regularization.lambda_pair)

    #write contact map and meta data info matfile
    ccmpred.io.contactmatrix.write_matrix(opt.matfile, res, meta, disable_apc=opt.disable_apc)

    #perform simple checks:
    ccmpred.sanity_check.check_single_potentials(res.x_single, verbose=1)
    ccmpred.sanity_check.check_pair_potentials(res.x_pair, verbose=1)

    exitcode = 0 if algret['code'] > 0 else -algret['code']
    sys.exit(exitcode)


if __name__ == '__main__':
    main()
