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
CCMpredPy is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods. 
From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model with 21 states for amino acids and gaps. 
The L2 norms of the pairwise coupling potentials will be written to the output matfile.
"""

OBJ_FUNC = {
    "pll": lambda opt, ccm: pll.PseudoLikelihood(ccm),
    "cd": lambda opt, ccm : cd.ContrastiveDivergence(
        ccm,
        gibbs_steps=opt.cd_gibbs_steps,
        sample_size=opt.sample_size,
        sample_ref=opt.sample_ref
    )
}

ALGORITHMS = {
    "conjugate_gradients": lambda opt, ccm: cg.conjugateGradient(
        ccm,
        maxit=opt.maxit, epsilon=opt.epsilon,
        convergence_prev=opt.convergence_prev, plotfile=opt.plotfile),
    "gradient_descent": lambda opt, ccm: gd.gradientDescent(
        ccm,
        maxit=opt.maxit, alpha0=opt.alpha0, decay=opt.decay, decay_start=opt.decay_start,
        decay_rate=opt.decay_rate, decay_type=opt.decay_type, epsilon=opt.epsilon,
        convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping, fix_v=opt.fix_v,
        plotfile=opt.plotfile
    ),
    "adam": lambda opt, ccm: ad.Adam(
        ccm,
        maxit=opt.maxit, alpha0=opt.alpha0, beta1=opt.beta1, beta2=opt.beta2,
        beta3=opt.beta3, epsilon=opt.epsilon, convergence_prev=opt.convergence_prev,
        early_stopping=opt.early_stopping, decay=opt.decay, decay_rate=opt.decay_rate,
        decay_start=opt.decay_start, fix_v=opt.fix_v,
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

    parser.add_argument("-i", "--init-from-raw",        dest="initrawfile", default=None, help="Init single and pair potentials from a binary raw file")
    parser.add_argument("-t", "--num_threads",          dest="num_threads", type=int, default=1, help="Specify the number of threads")
    parser.add_argument("--aln-format",                 dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--no-logo",                    dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo [default: %(default)s]")

    grp_contact_score = parser.add_argument_group("Contact Score and Corrections")
    grp_contact_score.add_argument("--frobenius",         dest="frob",     action="store_true", default=True,  help="Map 20x20 dimensional coupling matrices to Frobenius norm. [default: %(default)s]")
    grp_contact_score.add_argument("--no_centering_potentials", dest="centering_potentials", action="store_false", default=True, help="Ensure sum(v_i)=0 and sum(wij)=0 by subtracting mean. [default: %(default)s]")

    grp_corr = parser.add_argument_group("Corrections")
    grp_corr.add_argument("--apc",                  dest="apc",  action="store_true", default=False,  help="Apply average product correction (APC). [default: %(default)s] ")
    grp_corr.add_argument("--entropy-correction",   dest="entropy_correction", action="store_true", default=False, help="Apply entropy correction. [default: %(default)s]")
    grp_corr.add_argument("--joint-entropy-correction", dest="joint_entropy_correction", action="store_true", default=False, help="Apply joint entropy correction. [default: %(default)s]")
    grp_corr.add_argument("--sergeys-joint-entropy-correction", dest="sergeys-joint_entropy_correction", action="store_true", default=False, help="Apply sergeys joint entropy correction. [default: %(default)s]")


    grp_out = parser.add_argument_group("Output Options")
    grp_out.add_argument("-p", "--plot_opt_progress",       dest="plot_opt_progress", default=False, action="store_true", help="Plot optimization progress as interactive html. [default: %(default)s]")
    grp_out.add_argument("-b", "--write-binary-raw",        dest="out_binary_raw_file",  default=None, help="Write potentials as binary MessagePack file. [default: %(default)s]")


    grp_of = parser.add_argument_group("Objective Functions")
    grp_of.add_argument("--ofn-pll",             dest="objfun", action="store_const", const="pll", default="pll", help="Use pseudo-log-likelihood(pLL) [default: %(default)s]")
    grp_of.add_argument("--ofn-cd",              dest="objfun", action="store_const", const="cd", help="Use contrastive divergence (CD)")
    grp_of.add_argument("--cd-sample_size",      dest="sample_size",    default=10.0,   type=float, help="Setting for CD: multiplier X for sampling at max X * SAMPLE_REF sequences per iteration. [default: %(default)s] ")
    grp_of.add_argument("--cd-sample_ref",       dest="sample_ref",     default="L",    type=str, choices=['L', 'Neff'], help="Setting for CD: Sample at max SAMPLE_SIZE * [L|Neff] sequences per iteration. Reduces runtime. [default: %(default)s] ")
    grp_of.add_argument("--cd-gibbs_steps",      dest="cd_gibbs_steps", default=1,      type=int,  help="Setting for CD: Perform this many steps of Gibbs sampling per sequence. [default: %(default)s]")
    grp_of.add_argument("--cd-fix-v",            dest="fix_v",  action="store_true",  default=False,  help="Set single potentials v=v* and do not optimize single potentials v. [default: %(default)s]")

    grp_al = parser.add_argument_group("Optimization Algorithms")
    grp_al.add_argument("--alg-cg", dest="algorithm", action="store_const", const='conjugate_gradients', default='conjugate_gradients', help='Use conjugate gradients (CG) (standard for pLL (default) [default: %(default)s] ')
    grp_al.add_argument("--alg-gd", dest="algorithm", action="store_const", const='gradient_descent', help='Use gradient descent (GD) (standard for CD) ')
    grp_al.add_argument("--alg-nd", dest="algorithm", action="store_const", const='numerical_differentiation', help='Debug gradients with numerical differentiation ')
    grp_al.add_argument("--alg-ad", dest="algorithm", action="store_const", const='adam', help='Use ADAM (alternative for CD) according to Kingma & Ba, 2017 ')

    grp_als = parser.add_argument_group("Algorithm specific settings")
    grp_als.add_argument("--ad-beta1",          dest="beta1",           default=0.9,        type=float,     help="ADAM: Set beta 1 parameter (moemntum). [default: %(default)s]")
    grp_als.add_argument("--ad-beta2",          dest="beta2",           default=0.999,      type=float,     help="ADAM:Set beta 2 parameter (adaptivity) [default: %(default)s]")
    grp_als.add_argument("--ad-beta3",          dest="beta3",           default=0.9,        type=float,       help="ADAM:Set beta 3 parameter (temporal averaging) [default: %(default)s]")
    grp_als.add_argument("--alpha0",            dest="alpha0",          default=1e-3,       type=float,     help="ADAM and GD: Set initial learning rate. [default: %(default)s]")
    grp_als.add_argument("--decay",             dest="decay",           action="store_true", default=False, help="ADAM and GD: Use decaying learnign rate. Start decay when convergence criteria < START_DECAY. [default: %(default)s]")
    grp_als.add_argument("--decay-start",       dest="decay_start",     default=1e-4,       type=float,     help="ADAM and GD: Start decay when convergence criteria < START_DECAY. Only when --decay. [default: %(default)s]")
    grp_als.add_argument("--decay-rate",        dest="decay_rate",      default=1e1,        type=float,     help="ADAM and GD: Set rate of decay for learning rate. Only when --decay. [default: %(default)s]")
    grp_als.add_argument("--decay-type",        dest="decay_type",      default="step",     type=str,       choices=['sig', 'step', 'sqrt', 'power', 'exp', 'lin', 'keras'], help="ADAM and GD: Decay type. One of: step, sqrt, exp, power, lin. Only when --decay. [default: %(default)s]")


    grp_con = parser.add_argument_group("Convergence Settings")
    grp_con.add_argument("--maxit",                  dest="maxit",               default=500,    type=int, help="Stop when MAXIT number of iterations is reached. [default: %(default)s]")
    grp_con.add_argument("--early-stopping",         dest="early_stopping",      default=False,  action="store_true",  help="Apply convergence criteria instead of only maxit. [default: %(default)s]")
    grp_con.add_argument("--epsilon",                dest="epsilon",             default=1e-5,   type=float, help="Converged when relative change in f (or xnorm) in last CONVERGENCE_PREV iterations < EPSILON. [default: %(default)s]")
    grp_con.add_argument("--convergence_prev",       dest="convergence_prev",    default=5,      type=int,   help="Set CONVERGENCE_PREV parameter. [default: %(default)s]")


    grp_wt = parser.add_argument_group("Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const="weights_simple", default="weights_simple", help='Use simple weighting  [default: %(default)s]')
    grp_wt.add_argument("--wt-henikoff",        dest="weight", action="store_const", const="weights_henikoff", help='Use simple Henikoff weighting')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const="weights_uniform", help='Use uniform weighting')
    grp_wt.add_argument("--wt-ignore-gaps",     dest="wt_ignore_gaps",  action="store_true", default=False, help="Do not count gaps as identical amino acids during reweighting of sequences. [default: %(default)s]")
    grp_wt.add_argument("--wt-cutoff",          dest="wt_cutoff",       type=float, default=0.8, help="Sequence identity threshold. [default: %(default)s]")

    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-l2-lambda-single",           dest="lambda_single",           type=float, default=10,     help='Regularization coefficient for single potentials (L2 regularization) [default: %(default)s]')
    grp_rg.add_argument("--reg-l2-lambda-pair-factor",      dest="lambda_pair_factor",      type=float, default=0.2,    help='Regularization parameter for pair potentials (L2 regularization with lambda_pair  = lambda_pair-factor * scaling) [default: %(default)s]')
    grp_rg.add_argument("--reg-l2-scale_by_L",              dest="scaling",     action="store_const", const="L", default="L",   help="lambda_pair = lambda_pair-factor * (L-1) [default: %(default)s]")
    grp_rg.add_argument("--reg-l2-noscaling",               dest="scaling",     action="store_const", const="1",                help="lambda_pair = lambda_pair-factor")
    grp_rg.add_argument("--v-center",                       dest="reg_type",    action="store_const", const="v-center", default="v-center", help="Use mu=v* in Gaussian prior for single emissions and initialization. [default: %(default)s]")
    grp_rg.add_argument("--v-zero",                         dest="reg_type",    action="store_const", const="v-zero",           help="Use mu=0 in Gaussian prior for single emissions and initialisation.")


    grp_gp = parser.add_argument_group("Gap Treatment")
    grp_gp.add_argument("--max_gap_ratio",  dest="max_gap_ratio",   default=100, type=int, help="Ignore alignment positions with > MAX_GAP_RATIO percent gaps. [default: %(default)s == no removal of gaps]")


    grp_pc = parser.add_argument_group("Pseudocounts")
    grp_pc.add_argument("--pc-uniform",     dest="pseudocounts", action="store_const", const="uniform_pseudocounts", default="uniform_pseudocounts",     help="Use uniform pseudocounts, e.g 1/21 [default: %(default)s]")
    grp_pc.add_argument("--pc-submat",      dest="pseudocounts", action="store_const", const="substitution_matrix_pseudocounts", help="Use substitution matrix pseudocounts")
    grp_pc.add_argument("--pc-constant",    dest="pseudocounts", action="store_const", const="constant_pseudocounts",   help="Use constant pseudocounts ")
    grp_pc.add_argument("--pc-none",        dest="pseudocounts", action="store_const", const="no_pseudocounts", help="Use no pseudocounts")
    grp_pc.add_argument("--pc-count",       dest="pseudocount_single",  default=1, type=int, help="Specify number of pseudocounts [default: %(default)s]")
    grp_pc.add_argument("--pc-pair-count",  dest="pseudocount_pair",    default=1, type=int, help="Specify number of pseudocounts for pairwise frequencies [default: %(default)s]")


    grp_db = parser.add_argument_group("Debug Options")
    grp_db.add_argument("--write-cd-alignment", dest="cd_alnfile",      default=None, metavar="ALNFILE", help="Write PSICOV-formatted sampled alignment to ALNFILE")
    grp_db.add_argument("--do-not-optimize",    dest="optimize",        action="store_false", default=True, help="Do not optimize potentials. Only available when providing raw couplings file with -i.")


    scores = parser.add_argument_group("Alternative Coevolution Scores")
    scores.add_argument("--compute-omes",       dest="omes",                action="store_true", default=False, help="Compute OMES scores as in Kass and Horovitz 2002. [default: %(default)s]")
    scores.add_argument("--omes-fodoraldrich",  dest="omes_fodoraldrich",   action="store_true", default=False, help="OMES option: according to Fodor & Aldrich 2004. [default: %(default)s]")
    scores.add_argument("--compute-mi",         dest="mi",                  action="store_true", default=False, help="Compute mutual information (MI) . [default: %(default)s]")
    scores.add_argument("--mi-normalized",      dest="mi_normalized",       action="store_true", default=False, help="MI option: Compute normalized MI according to Martin et al 2005 . [default: %(default)s]")
    scores.add_argument("--mi-pseudocounts",    dest="mi_pseudocounts",     action="store_true", default=False, help="MI option: Compute MI with pseudocounts . [default: %(default)s]")



    args = parser.parse_args()

    if args.cd_alnfile and args.objfun != "cd":
        parser.error("--write-cd-alignment is only supported for contrastive divergence!")

    if not args.optimize and not args.initrawfile:
        parser.error("--do-not-optimize is only supported when -i (--init-from-raw) is specified!")

    if args.objfun == "pll" and (args.algorithm != "conjugate_gradients" and args.algorithm != "numerical_differentiation"):
        parser.error("pseudo-log-likelihood (--ofn-pll) needs to be optimized with conjugate gradients (--alg-cg)!")

    if args.objfun == "cd" and (args.algorithm != "gradient_descent" and args.algorithm != "adam"):
        parser.error("contrastive divergence (--ofn-cd) needs to be optimized with gradient descent (--alg-gd) or the ADAM optimizer (--alg-ad)!")


    args.plotfile=None
    if args.plot_opt_progress:
        args.plotfile=".".join(args.matfile.split(".")[:-1])+".opt_progress.html"

    return args


def main():

    #Read command line options
    opt = parse_args()

    if opt.logo:
        ccmpred.logo.logo()

    #set OMP environment variable for number of threads
    os.environ['OMP_NUM_THREADS'] = str(opt.num_threads)
    print("Using {0} threads for OMP parallelization.".format(os.environ["OMP_NUM_THREADS"]))


    ccm = CCMpred(opt.alnfile, opt.matfile)

    ##############################
    ### Setup
    ##############################

    #read alignment and compute amino acid counts and frequencies
    ccm.read_alignment(opt.aln_format, opt.max_gap_ratio)
    ccm.compute_sequence_weights(opt.weight, opt.wt_ignore_gaps, opt.wt_cutoff)
    ccm.compute_frequencies(opt.pseudocounts, opt.pseudocount_single,  opt.pseudocount_pair)

    #if alternative scores are specified: compute these and exit
    if opt.omes:
        ccm.compute_omes(opt.omes_fodoraldrich)
        ccm.write_matrix()
        sys.exit(0)

    if opt.mi:
        ccm.compute_mutual_info(opt.mi_normalized, opt.mi_pseudocounts)
        ccm.write_matrix()
        sys.exit(0)

    #setup L2 regularization
    ccm.specify_regularization(opt.lambda_single, opt.lambda_pair_factor, reg_type=opt.reg_type, scaling=opt.scaling)

    #intialise single and pair potentials either:
    #   - according to regularization priors
    #   - from file
    ccm.intialise_potentials(opt.initrawfile)


    ##############################
    ### Optimize objective function (pLL or CD) with optimization algorithm (CG, GD or ADAM)
    ##############################
    if opt.optimize:

        # specify objective function
        objfun = OBJ_FUNC[opt.objfun](opt, ccm)

        # specify optimizer
        alg = ALGORITHMS[opt.algorithm](opt, ccm)

        #minimize objective function with optimizer
        ccm.minimize(objfun, alg)
    else:
        print("\nDo not optimize but load couplings from binary raw file {0}\n".format(opt.initrawfile))

    ##############################
    ### Post Processing
    ##############################

    # Compute contact score (frobenius norm) by possibly recentering potentials
    # TODO: other scores can be added ...
    ccm.compute_contact_matrix(recenter_potentials=opt.centering_potentials, frob=opt.frob)

    # and bias correction to contact score
    ccm.compute_correction(
        apc=opt.apc,
        entropy_correction=opt.entropy_correction,
        joint_entropy=opt.joint_entropy_correction,
        sergeys_jec=opt.sergeys_joint_entropy_correction
    )

    #specify meta data, and write (corrected) contact matrices to files
    ccm.write_matrix()

    if opt.cd_alnfile and hasattr(ccm.f, 'msa_sampled'):
        ccm.write_sampled_alignment(opt.cd_alnfile)


    if opt.out_binary_raw_file:
        ccm.write_binary_raw(opt.out_binary_raw_file)

    exitcode = 0
    if opt.optimize:
        if ccm.algret['code'] < 0:
            exitcode =-ccm.algret['code']
    sys.exit(exitcode)



if __name__ == '__main__':
    main()
