#!/usr/bin/env python
import argparse
import sys
import os

from ccmpred import CCMpred
import ccmpred.logo
import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.algorithm.gradient_descent as gd
import ccmpred.algorithm.conjugate_gradients as cg
import ccmpred.algorithm.numdiff as nd
import ccmpred.algorithm.adam as ad


EPILOG = """
CCMpred is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods. From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model with 21 states for amino acids and gaps. The L2 norms of the pairwise coupling potentials will be written to the output matfile.
"""

OBJ_FUNC = {
    "pll": lambda opt, ccm: pll.PseudoLikelihood(ccm),
    "cd": lambda opt, ccm : cd.ContrastiveDivergence(ccm,
        gibbs_steps=opt.cd_gibbs_steps,
        persistent=opt.cd_persistent,
        min_nseq_factorL=opt.cd_min_nseq_factorl,
        minibatch_size=opt.minibatch_size,
        pll=opt.cd_pll)
}

ALGORITHMS = {
    "conjugate_gradients": lambda opt, ccm: cg.conjugateGradient(
        ccm, maxit=opt.maxit, epsilon=opt.epsilon,
        convergence_prev=opt.convergence_prev, plotfile=opt.plotfile),
    "gradient_descent": lambda opt, ccm: gd.gradientDescent(ccm,
        maxit=opt.maxit, alpha0=opt.alpha0, decay=opt.decay, decay_start=opt.decay_start,
        decay_rate=opt.decay_rate, decay_type=opt.decay_type, epsilon=opt.epsilon,
        convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping, fix_v=opt.fix_v,
        plotfile=opt.plotfile
    ),
    "adam": lambda opt, ccm: ad.Adam(ccm,
        maxit=opt.maxit, alpha0=opt.alpha0, beta1=opt.beta1, beta2=opt.beta2,
        beta3=opt.beta3, epsilon=opt.epsilon, convergence_prev=opt.convergence_prev,
        early_stopping=opt.early_stopping, decay=opt.decay, decay_rate=opt.decay_rate,
        decay_start=opt.decay_start, fix_v=opt.fix_v, qij_condition=opt.qij_condition,
        decay_type=opt.decay_type, plotfile=opt.plotfile
    ),
    "numerical_differentiation": lambda opt, ccm: nd.numDiff(maxit=opt.maxit, epsilon=opt.epsilon)
}


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

    parser.add_argument("alnfile", help="Input alignment file to use")
    parser.add_argument("matfile", help="Output matrix file to write")

    parser.add_argument("-i", "--init-from-raw",        dest="initrawfile", default=None, help="Init potentials from raw file")
    parser.add_argument("-t", "--num_threads",          dest="num_threads", type=int, default=1, help="Specify the number of threads")
    parser.add_argument("-A", "--disable_apc",          dest="disable_apc", action="store_true", default=False, help="Disable average product correction (APC)")
    parser.add_argument("--aln-format",                 dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--no-logo",                    dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo")


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
    grp_of.add_argument("--cd-minibatch_size",   dest="minibatch_size", default=5,      type=int, help="Minibatch size as multiples of protein length L [X*L].[default: %(default)s] ")
    grp_of.add_argument("--cd-gibbs_steps",      dest="cd_gibbs_steps", default=1,      type=int, help="Setting for CD: Perform GIBBS_STEPS of Gibbs sampling per sequence. [default: %(default)s]")


    grp_al = parser.add_argument_group("Algorithms")
    grp_al.add_argument("--alg-cg", dest="algorithm", action="store_const", const='conjugate_gradients', default='conjugate_gradients', help='Use conjugate gradients (default)')
    grp_al.add_argument("--alg-gd", dest="algorithm", action="store_const", const='gradient_descent', help='Use gradient descent')
    grp_al.add_argument("--alg-nd", dest="algorithm", action="store_const", const='numerical_differentiation', help='Debug gradients with numerical differentiation')
    grp_al.add_argument("--alg-ad", dest="algorithm", action="store_const", const='adam', help='Use Adam')

    grp_als = parser.add_argument_group("Algorithm specific settings")
    grp_als.add_argument("--ad-beta1",          dest="beta1",           default=0.9,        type=float,     help="Set beta 1 parameter for Adam (moemntum). [default: %(default)s]")
    grp_als.add_argument("--ad-beta2",          dest="beta2",           default=0.999,      type=float,     help="Set beta 2 parameter for Adam (adaptivity) [default: %(default)s]")
    grp_als.add_argument("--ad-beta3",          dest="beta3",           default=0.9,      type=float,       help="Set beta 3 parameter for Adam (temporal averaging) [default: %(default)s]")
    grp_als.add_argument("--alpha0",            dest="alpha0",          default=1e-3,       type=float,     help="Set initial learning rate. [default: %(default)s]")
    grp_als.add_argument("--decay",             dest="decay",           action="store_true", default=False, help="Use decaying learnign rate. Start decay when convergence criteria < START_DECAY. [default: %(default)s]")
    grp_als.add_argument("--decay-start",       dest="decay_start",     default=1e-4,       type=float,     help="Start decay when convergence criteria < START_DECAY. [default: %(default)s]")
    grp_als.add_argument("--decay-rate",        dest="decay_rate",     default=1e1,        type=float,     help="Set rate of decay for learning rate when --decay is on. [default: %(default)s]")
    grp_als.add_argument("--decay-type",        dest="decay_type",      default="step",     type=str,       choices=['sig', 'step', 'sqrt', 'power', 'exp', 'lin'], help="Decay type. One of: step, sqrt, exp, power, lin. [default: %(default)s]")


    grp_con = parser.add_argument_group("Convergence Settings")
    grp_con.add_argument("--maxit",                  dest="maxit",               default=500,    type=int, help="Stop when MAXIT number of iterations is reached. [default: %(default)s]")
    grp_con.add_argument("--early-stopping",         dest="early_stopping",      default=False,  action="store_true",  help="Apply convergence criteria instead of only maxit. [default: %(default)s]")
    grp_con.add_argument("--epsilon",                dest="epsilon",             default=1e-5,   type=float, help="Converged when relative change in f (or xnorm) in last CONVERGENCE_PREV iterations < EPSILON. [default: %(default)s]")
    grp_con.add_argument("--convergence_prev",       dest="convergence_prev",    default=5,      type=int,   help="Set CONVERGENCE_PREV parameter. [default: %(default)s]")
    grp_con.add_argument("--qij-condition",          dest="qij_condition",       action="store_true", default=False,  help="Compution of q_ij with all q_ijab > 0. [default: %(default)s]")


    grp_wt = parser.add_argument_group("Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const="weights_simple", default="weights_simple", help='Use simple weighting (default)')
    grp_wt.add_argument("--wt-henikoff",        dest="weight", action="store_const", const="weights_henikoff", help='Use simple Henikoff weighting')
    grp_wt.add_argument("--wt-henikoff_pair",   dest="weight", action="store_const", const="weights_henikoff_pair", help='Use Henikoff pair weighting ')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const="weights_uniform", help='Use uniform weighting')
    grp_wt.add_argument("--wt-ignore-gaps",     dest="wt_ignore_gaps",  action="store_true", default=False, help="Do not count gaps as identical amino acids during reweighting of sequences. [default: %(default)s]")
    grp_wt.add_argument("--wt-cutoff",          dest="wt_cutoff",       type=float, default=0.8, help="Sequence identity threshold. [default: %(default)s]")

    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-l2-lambda-single",           dest="lambda_single",           type=float, default=10,     help='Regularization coefficient for single potentials (L2 regularization) [default: %(default)s]')
    grp_rg.add_argument("--reg-l2-lambda-pair-factor",      dest="lambda_pair_factor",      type=float, default=0.2,    help='Regularization parameter for pair potentials (L2 regularization with lambda_pair  = lambda_pair-factor * scaling) [default: %(default)s]')
    grp_rg.add_argument("--reg-l2-scale_by_L",              dest="scaling",     action="store_const", const="L", default="L",   help="lambda_pair = lambda_pair-factor * (L-1) (default)")
    grp_rg.add_argument("--reg-l2-scale_by_div",            dest="scaling",     action="store_const", const="diversity",        help="lambda_pair = lambda_pair-factor * (L/sqrt(N))")
    grp_rg.add_argument("--reg-l2-noscaling",               dest="scaling",     action="store_const", const="1",                help="lambda_pair = lambda_pair-factor")
    grp_rg.add_argument("--center-v",                       dest="reg_type",    action="store_const", const="center-v",         default="zero", help="Use mu=v* for gaussian prior for single emissions.")
    grp_rg.add_argument("--fix-v",                          dest="fix_v",       action="store_true",    default=False,          help="Use v=v* and do not optimize v.")

    grp_gp = parser.add_argument_group("Gap Treatment")
    grp_gp.add_argument("--max_gap_ratio",  dest="max_gap_ratio",   default=100, type=int, help="Remove alignment positions with > MAX_GAP_RATIO percent gaps. [default: %(default)s == no removal of gaps]")


    grp_pc = parser.add_argument_group("Pseudocounts")
    grp_pc.add_argument("--pc-uniform",     dest="pseudocounts", action="store_const", default="uniform_pseudocounts", const="uniform_pseudocounts",    help="Use uniform pseudocounts, e.g 1/21 (default)")
    grp_pc.add_argument("--pc-submat",      dest="pseudocounts", action="store_const", const="substitution_matrix_pseudocounts", help="Use substitution matrix pseudocounts")
    grp_pc.add_argument("--pc-constant",    dest="pseudocounts", action="store_const", const="constant_pseudocounts",   help="Use constant pseudocounts ")
    grp_pc.add_argument("--pc-none",        dest="pseudocounts", action="store_const", const="no_pseudocounts", help="Use no pseudocounts")
    grp_pc.add_argument("--pc-count",       dest="pseudocount_single",  default=1, type=int, help="Specify number of pseudocounts [default: %(default)s]")
    grp_pc.add_argument("--pc-pair-count",  dest="pseudocount_pair",    default=1, type=int, help="Specify number of pseudocounts for pairwise frequencies [default: %(default)s]")

    grp_db = parser.add_argument_group("Debug Options")
    grp_db.add_argument("--write-cd-alignment", dest="cd_alnfile",      default=None, metavar="ALNFILE", help="Write PSICOV-formatted sampled alignment to ALNFILE")
    grp_db.add_argument("--compare-to-raw",     dest="comparerawfile",  default=None, help="Compare potentials to raw file")
    grp_db.add_argument("--dev-center-v",       dest="dev_center_v",    action="store_true", default=False, help="Use same settings as in c++ dev-center-v version")
    grp_db.add_argument("--ccmpred-vanilla",    dest="vanilla",         action="store_true", default=False, help="Use same settings as in default c++ CCMpred")


    scores = parser.add_argument_group("Alternative Scores")
    scores.add_argument("--compute-omes",           dest="omes",                action="store_true", default=False, help="Compute OMES scores as in Kass and Horovitz . [default: %(default)s]")
    scores.add_argument("--omes-fodoraldrich",      dest="omes_fodoraldrich",   action="store_true", default=False, help="Comoute OMES as in Fodor & Aldrich 2004 . [default: %(default)s]")
    scores.add_argument("--compute-mi",             dest="mi",                  action="store_true", default=False, help="Compute MI scores . [default: %(default)s]")
    scores.add_argument("--mi-normalized",          dest="mi_normalized",       action="store_true", default=False, help="Compute normalized MI according to Martin et al 2005 . [default: %(default)s]")
    scores.add_argument("--mi-pseudocounts",        dest="mi_pseudocounts",     action="store_true", default=False, help="Compute MI with pseudocounts . [default: %(default)s]")



    args = parser.parse_args()

    if args.cd_alnfile and args.objfun != "cd":
        parser.error("--write-cd-alignment is only supported for contrastive divergence!")

    if args.only_model_prob and not args.initrawfile:
        parser.error("--only_model_prob is only supported when -i (--init-from-raw) is specified!")

    if args.objfun == "pll" and args.algorithm != "conjugate_gradients":
        parser.error("pseudo-log-likelihood (--ofn-pll) needs to be optimized with conjugate gradients (--alg-cg)!")

    if (args.outmodelprobmsgpackfile and args.objfun != "cd") or args.only_model_prob:
        print("Note: when computing q_ij data: couplings should be computed from full likelihood (e.g. CD)")


    args.plotfile=None
    if args.plot_opt_progress:
        args.plotfile="".join(args.matfile.split(".")[:-1])+".opt_progress.html"

    return args


def main():

    opt = parse_args()

    if opt.logo:
        ccmpred.logo.logo()

    #set OMP environment variable for number of threads
    os.environ['OMP_NUM_THREADS'] = str(opt.num_threads)
    print("Using {0} threads for OMP parallelization.".format(os.environ["OMP_NUM_THREADS"]))


    ccm = CCMpred(opt.alnfile, opt.matfile)

    #read alignment and compute frequency and counts
    ccm.read_alignment(opt.aln_format, opt.max_gap_ratio)
    ccm.compute_sequence_weights(opt.weight, opt.wt_ignore_gaps, opt.wt_cutoff)
    ccm.compute_frequencies(opt.pseudocounts, opt.pseudocount_single,  opt.pseudocount_pair, dev_center_v=opt.dev_center_v)

    if opt.omes:
        ccm.compute_omes(opt.omes_fodoraldrich)
        ccm.write_mat()
        sys.exit(0)

    if opt.mi:
        ccm.compute_mutual_info(opt.mi_normalized, opt.mi_pseudocounts)
        ccm.write_mat()
        sys.exit(0)

    #setup L2 regularization
    ccm.specify_regularization(opt.lambda_single, opt.lambda_pair_factor, reg_type=opt.reg_type, scaling=opt.scaling, dev_center_v=opt.dev_center_v)

    #intialise potentials
    ccm.intialise_potentials(opt.initrawfile, opt.vanilla)

    #only compute model frequencies and exit
    if opt.only_model_prob and opt.outmodelprobmsgpackfile:
        ccm.write_binary_modelprobs(opt.outmodelprobmsgpackfile)
        sys.exit(0)

    #specify objective function
    objfun = OBJ_FUNC[opt.objfun](opt, ccm)

    #specify optimizer
    alg = ALGORITHMS[opt.algorithm](opt, ccm)

    #optimize OBJFUN with ALGORITHM
    ccm.minimize(objfun, alg)

    #center the variables
    if opt.centering_potentials:
        ccm.recenter_potentials()

    #specify meta data, apply apc and write contact matrix
    ccm.write_mat(apc=not opt.disable_apc)

    if opt.cd_alnfile and hasattr(ccm.f, 'msa_sampled'):
        ccm.write_sampled_alignment(opt.cd_alnfile)

    if opt.outrawfile:
        ccm.write_raw(opt.outrawfile)

    if opt.outmsgpackfile:
        ccm.write_binary_raw(opt.outmsgpackfile)

    if opt.outmodelprobmsgpackfile:
        ccm.write_binary_modelprobs(opt.outmodelprobmsgpackfile)

    #print("\nCCMpred was running with following settings:")
    #print(ccm)


    exitcode = 0 if ccm.algret['code'] > 0 else -ccm.algret['code']
    sys.exit(exitcode)



if __name__ == '__main__':
    main()
