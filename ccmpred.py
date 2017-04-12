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
    "gradient_descent": lambda opt: gd.gradientDescent(
        maxit=opt.maxit, alpha0=opt.alpha0, decay=opt.decay, start_decay=opt.start_decay, alpha_decay=opt.alpha_decay,
        epsilon=opt.epsilon, convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping
    ),
    "adam": lambda opt: ad.Adam(
        maxit=opt.maxit, alpha0=opt.alpha0, beta1=opt.beta1, beta2=opt.beta2, epsilon=opt.epsilon,
        convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping, decay=opt.decay,
        alpha_decay=opt.alpha_decay, start_decay=opt.start_decay, fix_v=opt.fix_v,
        group_alpha=opt.group_alpha, qij_condition=opt.qij_condition,
        decay_type=opt.decay_type
    ),
    "numerical_differentiation": lambda opt: nd.numDiff(maxit=opt.maxit, epsilon=opt.epsilon)
}


OBJ_FUNC = {
    "pll":  lambda opt, msa, freqs, weights, raw_init, regularization: pll.PseudoLikelihood(
        msa, freqs, weights, raw_init, regularization
    ),
    "cd":   lambda opt, msa, freqs, weights, raw_init, regularization: cd.ContrastiveDivergence(
        msa, freqs, weights, raw_init, regularization,
        opt.cd_gibbs_steps, opt.cd_persistent, opt.cd_min_nseq_factorl, opt.cd_pll
    ),

}


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

    parser.add_argument("-i", "--init-from-raw",        dest="initrawfile", default=None, help="Init potentials from raw file")
    parser.add_argument("-A", "--disable_apc",          dest="disable_apc", action="store_true", default=False, help="Disable average product correction (APC)")
    parser.add_argument("--aln-format",                 dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--no-logo",                    dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo")


    parser.add_argument("alnfile", help="Input alignment file to use")
    parser.add_argument("matfile", help="Output matrix file to write")

    grp_out = parser.add_argument_group("Output Options")
    grp_out.add_argument("-p", "--plot_opt_progress",       dest="plot_opt_progress", default=False, action="store_true", help="Plot optimization progress")
    grp_out.add_argument("-r", "--write-raw",               dest="outrawfile", default=None, help="Write potentials to raw file")
    grp_out.add_argument("-b", "--write-msgpack",           dest="outmsgpackfile", default=None, help="Write potentials to MessagePack file")
    grp_out.add_argument("-m", "--write-modelprob-msgpack", dest="outmodelprobmsgpackfile", default=None, help="Write model probabilities as MessagePack file")
    grp_out.add_argument("--only_model_prob",               dest="only_model_prob", action="store_true", default=False, help="Only compute model probabilties and do not optimize (-i must be specified!).")
    grp_out.add_argument("--no_centering_potentials",       dest="centering_potentials", action="store_false", default=True, help="Ensure that sum(wij)=0 by subtracting mean.")


    grp_of = parser.add_argument_group("Objective Functions")
    grp_of.add_argument("--ofn-pll",             dest="objfun", action="store_const", const="pll", default="pll", help="Use pseudo-log-likelihood (default)")
    grp_of.add_argument("--ofn-cd",              dest="objfun", action="store_const", const="cd", help="Use Contrastive Divergence. Sample at least MIN_NSEQ_FACTORL * L  sequences (taken from input MSA) with Gibbs sampling (each sequences is sampled with GIBBS_STEPS.")
    grp_of.add_argument("--cd-pll",              dest="cd_pll", action="store_true", default=False, help="Setting for CD: Sample only ONE variable per sampling step per sequence. [default: %(default)s]")
    grp_of.add_argument("--cd-persistent",       dest="cd_persistent", action="store_true",  default=False, help="Setting for CD: Use Persistent Contrastive Divergence: do not restart Markov Chain in each iteration.[default: %(default)s] ")
    grp_of.add_argument("--cd-min_nseq_factorl", dest="cd_min_nseq_factorl", default=0,      type=int, help="Setting for CD: Sample at least MIN_NSEQ_FACTORL * L  sequences (taken from input MSA).[default: %(default)s] ")
    grp_of.add_argument("--cd-gibbs_steps",      dest="cd_gibbs_steps", default=1,      type=int, help="Setting for CD: Perform GIBBS_STEPS of Gibbs sampling per sequence. [default: %(default)s]")
    grp_of.add_argument("--ofn-tree-cd", action=TreeCDAction, metavar=("TREEFILE", "ANCESTORFILE"), nargs=2, type=str, help="Use Tree-controlled Contrastive Divergence, loading tree data from TREEFILE and ancestral sequence data from ANCESTORFILE")

    grp_al = parser.add_argument_group("Algorithms")
    grp_al.add_argument("--alg-cg", dest="algorithm", action="store_const", const='conjugate_gradients', default='conjugate_gradients', help='Use conjugate gradients (default)')
    grp_al.add_argument("--alg-gd", dest="algorithm", action="store_const", const='gradient_descent', help='Use gradient descent')
    grp_al.add_argument("--alg-nd", dest="algorithm", action="store_const", const='numerical_differentiation', help='Debug gradients with numerical differentiation')
    grp_al.add_argument("--alg-ad", dest="algorithm", action="store_const", const='adam', help='Use Adam')

    grp_als = parser.add_argument_group("Algorithm specific settings")
    grp_als.add_argument("--ad-beta1",          dest="beta1",           default=0.9,        type=float,     help="Set beta 1 parameter for Adam. [default: %(default)s]")
    grp_als.add_argument("--ad-beta2",          dest="beta2",           default=0.999,      type=float,     help="Set beta 2 parameter for Adam. [default: %(default)s]")
    grp_als.add_argument("--ad-group_alpha",    dest="group_alpha",     action="store_true", default=False, help="Use min learning rate for each group v_i and w_ij. [default: %(default)s]")
    grp_als.add_argument("--alpha0",            dest="alpha0",          default=1e-3,       type=float,     help="Set initial learning rate. [default: %(default)s]")
    grp_als.add_argument("--decay",             dest="decay",           action="store_true", default=False, help="Use decaying learnign rate. Start decay when convergence criteria < START_DECAY. [default: %(default)s]")
    grp_als.add_argument("--start_decay",       dest="start_decay",     default=1e-4,       type=float,     help="Start decay when convergence criteria < START_DECAY. [default: %(default)s]")
    grp_als.add_argument("--alpha_decay",       dest="alpha_decay",     default=1e1,        type=float,     help="Set rate of decay for learning rate when --decay is on. [default: %(default)s]")
    grp_als.add_argument("--decay-type",        dest="decay_type",      default="step",     type=str,       help="Decay type. One of: step, sqrt, exp. [default: %(default)s]")


    grp_con = parser.add_argument_group("Convergence Settings")
    grp_con.add_argument("--maxit",                  dest="maxit",               default=500,    type=int, help="Stop when MAXIT number of iterations is reached. [default: %(default)s]")
    grp_con.add_argument("--early_stopping",         dest="early_stopping",      default=False,  action="store_true",  help="Apply convergence criteria instead of only maxit. [default: %(default)s]")
    grp_con.add_argument("--epsilon",                dest="epsilon",             default=1e-5,   type=float, help="Converged when relative change in f (or xnorm) in last CONVERGENCE_PREV iterations < EPSILON. [default: %(default)s]")
    grp_con.add_argument("--convergence_prev",       dest="convergence_prev",    default=5,      type=int,   help="Set CONVERGENCE_PREV parameter. [default: %(default)s]")
    grp_con.add_argument("--qij_condition",          dest="qij_condition",       action="store_true", default=False,  help="Compution of q_ij with all q_ijab > 0. [default: %(default)s]")


    grp_wt = parser.add_argument_group("Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const=ccmpred.weighting.weights_simple, default=ccmpred.weighting.weights_simple, help='Use simple weighting (default)')
    grp_wt.add_argument("--wt-henikoff",        dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff, help='Use simple Henikoff weighting')
    grp_wt.add_argument("--wt-henikoff_pair",   dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff_pair, help='Use Henikoff pair weighting ')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const=ccmpred.weighting.weights_uniform, help='Use uniform weighting')

    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-l2", dest="regularization", action=RegL2Action, type=float, nargs=2, metavar=("LAMBDA_SINGLE", "LAMBDA_PAIR_FACTOR"), default=lambda msa, centering, scaling: ccmpred.regularization.L2(10, 0.2 * scaling, centering), help='Use L2 regularization with coefficients LAMBDA_SINGLE, LAMBDA_PAIR_FACTOR * SCALING;  (default: 10 0.2)')
    grp_rg.add_argument("--reg-l2-scale_by_L",      dest="scaling", action="store_const", const="L", default="L", help="LAMBDA_PAIR = LAMBDA_PAIR_FACTOR * (L-1) (default)")
    grp_rg.add_argument("--reg-l2-scale_by_div",    dest="scaling", action="store_const", const="diversity", help="LAMBDA_PAIR = LAMBDA_PAIR_FACTOR * (L/sqrt(N))")
    grp_rg.add_argument("--reg-l2-noscaling",       dest="scaling", action="store_const", const="1", help="LAMBDA_PAIR = LAMBDA_PAIR_FACTOR")
    grp_rg.add_argument("--center-v",               dest="reg_type", action="store_const", const="center-v", default="zero", help="Use mu=v* for gaussian prior for single emissions.")
    grp_rg.add_argument("--fix-v",                  dest="fix_v",   action="store_true",    default=False, help="Use v=v* and do not optimize v.")

    grp_gp = parser.add_argument_group("Gap Treatment")
    grp_gp.add_argument("--max_gap_ratio",  dest="max_gap_ratio",   default=100, type=int, help="Remove alignment positions with > MAX_GAP_RATIO percent gaps. [default: %(default)s == no removal of gaps]")
    grp_gp.add_argument("--wt-ignore-gaps", dest="ignore_gaps",     action="store_true", default=False, help="Do not count gaps as identical amino acids during reweighting of sequences. [default: %(default)s]")

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


    args = parser.parse_args()

    if args.cd_alnfile and args.objfun not in (cd.ContrastiveDivergence, treecd.TreeContrastiveDivergence):
        parser.error("--write-cd-alignment is only supported for (tree) contrastive divergence!")

    if args.only_model_prob and not args.initrawfile:
        parser.error("--only_model_prob is only supported when -i (--init-from-raw) is specified!")


    if args.decay and args.decay_type not in ["step", "sqrt", "exp"]:
        parser.error("--decay-type must be one of: step, sqrt, exp!")

    if (args.outmodelprobmsgpackfile and args.objfun != "cd") or args.only_model_prob:
        print("Note: when computing q_ij data: couplings should be computed from full likelihood (e.g. CD)")


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


    print opt.reg_type

    #setup regularization properties
    if opt.dev_center_v or opt.reg_type == "center-v":
        centering   = ccmpred.centering.center_v(freqs)
    else:
        centering   = ccmpred.centering.center_zero(freqs)


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


    if opt.objfun == "cd" and not opt.initrawfile:
        f = OBJ_FUNC["pll"](opt, msa, freqs, weights, raw_init, regularization)
        x0 = f.x0
        alg = ALGORITHMS["conjugate_gradients"](opt)
        alg.set_epsilon(1e-2)
        alg.set_maxit(100)
        print("\n Initialize pair potentials with from pLL potentials after some iterations. \n")
        fx, x, algret = alg.minimize(f, x0, None)
        res = f.finalize(x, {})
        raw_init.x_pair = res.x_pair

    #initialise objective function
    f = OBJ_FUNC[opt.objfun](opt, msa, freqs, weights, raw_init, regularization)
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


    print("\n Will optimize {0} {1} variables wrt {2} and {3}".format(x0.size, x0.dtype, f, f.regularization))
    print("Optimizer: {0}".format(alg))
    fx, x, algret = alg.minimize(f, x0, plotfile)


    condition = "Finished" if algret['code'] >= 0 else "Exited"
    print("\n{0} with code {code} -- {message}\n".format(condition, **algret))

    meta = ccmpred.metadata.create(opt, regularization, msa, weights, f, fx, algret, alg)
    res = f.finalize(x, meta)


    if opt.centering_potentials:
        #perform checks on potentials:
        check_x_single  = ccmpred.sanity_check.check_single_potentials(res.x_single, verbose=0)
        check_x_pair  = ccmpred.sanity_check.check_pair_potentials(res.x_pair, verbose=0)

        #enforce sum(wij)=0 and sum(v_i)=0
        if not check_x_single or not check_x_pair:
            print("Enforce sum(v_i)=0 and sum(w_ij)=0 by centering potentials at zero.")
            res.x_single, res.x_pair = ccmpred.sanity_check.centering_potentials(res.x_single, res.x_pair)



    if opt.cd_alnfile and hasattr(f, 'msa_sampled'):
        print("\nWriting sampled alignment to {0}".format(opt.cd_alnfile))
        msa_sampled = f.msa_sampled

        with open(opt.cd_alnfile, "w") as f:
            ccmpred.io.alignment.write_msa_psicov(f, msa_sampled)

    if opt.max_gap_ratio < 100:
        ccmpred.gaps.backinsert_gapped_positions(res, gapped_positions)

    if opt.outrawfile:
        print("\nWriting raw-formatted potentials to {0}".format(opt.outrawfile))
        ccmpred.raw.write_oldraw(opt.outrawfile, res)

    if opt.outmsgpackfile:
        print("\nWriting msgpack-formatted potentials to {0}".format(opt.outmsgpackfile))
        ccmpred.raw.write_msgpack(opt.outmsgpackfile, res)

    if opt.outmodelprobmsgpackfile:
        print("\nWriting msgpack-formatted model probabilties to {0}".format(opt.outmodelprobmsgpackfile))
        if opt.max_gap_ratio < 100:
            msa = ccmpred.io.alignment.read_msa(opt.alnfile, opt.aln_format)
            freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, opt.pseudocounts[0], pseudocount_n_single=opt.pseudocounts[1], pseudocount_n_pair=opt.pseudocount_pair_count)
        if opt.dev_center_v:
            freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, ccmpred.pseudocounts.constant_pseudocounts, pseudocount_n_single=1, pseudocount_n_pair=1, remove_gaps=True)

        ccmpred.model_probabilities.write_msgpack(opt.outmodelprobmsgpackfile, res, weights, msa, freqs, regularization.lambda_pair, True)

    #write contact map and meta data info matfile
    ccmpred.io.contactmatrix.write_matrix(opt.matfile, res, meta, disable_apc=opt.disable_apc)

    exitcode = 0 if algret['code'] > 0 else -algret['code']
    sys.exit(exitcode)


if __name__ == '__main__':
    main()
