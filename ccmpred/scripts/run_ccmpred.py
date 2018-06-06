#!/usr/bin/env python
import argparse
import sys
import os

from ccmpred import CCMpred
import ccmpred.logo


EPILOG = """
CCMpredPy is a fast python implementation of contact prediction method based on correlated mutations. 
From an alignment given as alnfile, it will infer the parameters of a Potts model with 21 states for amino acids and gaps. 
Either pseudo-likelihood maximization or contrastive divergence can be chosen as inference algorithm. 
The L2 norms of the pairwise coupling potentials will be written to the output matfile.
"""


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

    grp_general = parser.add_argument_group("General Options")
    grp_general.add_argument("--num-threads",          dest="num_threads", type=int, default=1,
                        help="Specify the number of threads. [default: %(default)s]")
    grp_general.add_argument("--aln-format",                 dest="aln_format", default="fasta",
                        help="File format for MSAs [default: \"%(default)s\"]")
    grp_general.add_argument("--no-logo",                    dest="logo", default=True, action="store_false",
                        help="Disable showing the CCMpred logo [default: %(default)s]")


    grp_out = parser.add_argument_group("Output Options")
    grp_out.add_argument("-m", "--mat-file", dest="matfile", type=str,
                         help="Write contact score matrix to file. [default: %(default)s]")
    grp_out.add_argument("-b", "--write-binary-raw", dest="out_binary_raw_file", type=str,
                         help="Write single and pairwise potentials as binary MessagePack file. [default: %(default)s]")
    grp_out.add_argument("--plot-opt-progress", dest="plot_opt_progress", type=str,
                         help="Continously plot optimization progress as an interactive HTML. [default: %(default)s]")


    grp_in = parser.add_argument_group("Optional Input Options")
    grp_in.add_argument("-i", "--init-from-raw",        dest="initrawfile", default=None,
                        help="Init single and pair potentials from a binary raw file")
    grp_in.add_argument("--do-not-optimize",            dest="optimize", action="store_false", default=True,
                        help="Do not optimize potentials. Requires providing initial model parameters with -i.")



    grp_pll = parser.add_argument_group("Pseudo-Likelihood Options")
    grp_pll.add_argument("--ofn-pll", dest="objfun", action="store_const", const="pll", default="pll",
                         help="Use pseudo-log-likelihood(pLL)")
    grp_pll.add_argument("--lbfgs-ftol", dest="ftol",  default=1e-4, type=float,
                         help="LBFGS: convergence criterion ftol. [default: %(default)s]")
    grp_pll.add_argument("--lbfgs-max-linesearch", dest="max_linesearch", default=5, type=int,
                         help="LBFGS: maximum number of linesearch steps. [default: %(default)s]")
    grp_pll.add_argument("--lbfgs-maxcor", dest="max_cor", default=5, type=int,
                         help="LBFGS: maximum number of corrections for memory. [default: %(default)s]")


    grp_cd = parser.add_argument_group("(Persistent) Contrastive Divergence Options")
    grp_cd.add_argument("--ofn-cd",dest="objfun",action="store_const",const="cd",help="Use contrastive divergence (CD)")
    grp_cd.add_argument("--nr-markov-chains", dest="nr_seq_sample", type=int, default=500, help="Number of parallel "
                        "Markov chains used for sampling at each iteration. [default: %(default)s] ")
    grp_cd.add_argument("--gibbs_steps", dest="cd_gibbs_steps", type=int, default=1,
                         help="Number of Gibbs steps used to evolve each Markov chain "
                              "in each iteration of the optimization.  [default: %(default)s]")
    grp_cd.add_argument("--persistent", dest="cd_persistent", action="store_true", default=False, help="Switch on "
                        "PERSISTENT CD once the learning rate is small enough (< alpha_0 / 10) [default: %(default)s]")
    grp_cd.add_argument("--alpha0", dest="alpha0", default=1e-3, type=float,
                        help="GD: Set initial learning rate. [default: %(default)s]")
    grp_cd.add_argument("--no-decay", dest="decay", action="store_false", default=True,
                        help="GD: Do not use decaying learning rate** (`--no-decay`): Do not use decaying learnign "
                             "rates. Decay is started when convergence criteria falls below value of START_DECAY. "
                             "[default: %(default)s]")
    grp_cd.add_argument("--decay-start", dest="decay_start", default=1e-1, type=float,
                        help="GD: Start decay when convergence criteria < START_DECAY."
                             "[default: %(default)s]")
    grp_cd.add_argument("--decay-rate", dest="decay_rate", default=5e-6, type=float,
                        help="GD: Set rate of decay for learning rate. [default: %(default)s]")
    grp_cd.add_argument("--decay-type", dest="decay_type", default="sig", type=str,
                        choices=['sig', 'sqrt', 'exp', 'lin'],
                        help="GD: Decay type. [default: %(default)s]")


    grp_con = parser.add_argument_group("Convergence Settings")
    grp_con.add_argument("--maxit", dest="maxit", default=2000, type=int,
                         help="Stop when MAXIT number of iterations is reached. [default: %(default)s]")
    grp_con.add_argument("--early-stopping", dest="early_stopping", default=False, action="store_true",
                         help="Apply convergence criteria instead of only maxit. [default: %(default)s]")
    grp_con.add_argument("--epsilon", dest="epsilon", default=1e-5, type=float,
                         help="Converged when relative change in f (or xnorm) in last CONVERGENCE_PREV iterations "
                              "< EPSILON. [default: %(default)s]")
    grp_con.add_argument("--convergence_prev", dest="convergence_prev", default=5, type=int,
                         help="Set CONVERGENCE_PREV parameter. [default: %(default)s]")



    grp_constraints = parser.add_argument_group("Use with Contraints (non-contacts will obtain zero couplings)")
    grp_constraints.add_argument("--pdb-file", dest="pdbfile", help="Input PDB file")
    grp_constraints.add_argument("--contact-threshold", dest="contact_threshold", type=int, default=8,
                           help="Definition of residue pairs forming a contact wrt distance of their Cbeta atoms in "
                                "angstrom. [default: %(default)s]")



    grp_corr = parser.add_argument_group("Corrections applied to Contact Score")
    grp_corr.add_argument("--apc",                  dest="apc_file",  type=str, default=None,
                          help="Path to contact matrix file corrected with average product correction (APC). "
                               "[default: %(default)s] ")
    grp_corr.add_argument("--entropy-correction",   dest="entropy_correction_file",  type=str, default=None,
                          help="Path to contact matrix file corrected with entropy correction. "
                               "[default: %(default)s]")


    grp_wt = parser.add_argument_group("Sequence Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const="simple",
                        default="simple", help='Use simple weighting  [default: %(default)s]')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const="uniform",
                        help='Use uniform weighting')
    grp_wt.add_argument("--wt-cutoff",          dest="wt_cutoff",       type=float, default=0.8,
                        help="Sequence identity threshold. [default: %(default)s]")


    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-lambda-single",          dest="lambda_single",           type=float, default=10,
                        help='Regularization coefficient for single potentials (L2 regularization) '
                             '[default: %(default)s]')
    grp_rg.add_argument("--reg-lambda-pair-factor",     dest="lambda_pair_factor",      type=float, default=0.2,
                        help='Regularization parameter for pair potentials (L2 regularization with '
                             'lambda_pair  = lambda_pair-factor * scaling) [default: %(default)s]')
    grp_rg.add_argument("--v-center", dest="single_prior", action="store_const", const="v-center", default="v-center",
                        help="Use mu=v* in Gaussian prior for single emissions and initialization. [default: %(default)s]")
    grp_rg.add_argument("--v-zero", dest="single_prior", action="store_const", const="v-zero",
                        help="Use mu=0 in Gaussian prior for single emissions and initialisation.")



    grp_gap = parser.add_argument_group("Gap Treatment")
    grp_gap.add_argument("--max-gap-pos",  dest="max_gap_pos",  default=100, type=int,
                        help="Ignore alignment positions with > MAX_GAP_POS percent gaps. "
                             "[default: %(default)s == no removal of positions]")
    grp_gap.add_argument("--max-gap-seq",  dest="max_gap_seq",  default=100, type=int,
                        help="Remove sequences with > MAX_GAP_SEQ percent gaps. [default: %(default)s == no removal of sequences]")


    grp_pc = parser.add_argument_group("Pseudocounts")
    grp_pc.add_argument("--pc-uniform",     dest="pseudocounts", action="store_const", const="uniform_pseudocounts",
                        default="uniform_pseudocounts",
                        help="Use uniform pseudocounts, e.g 1/21 [default: %(default)s]")
    grp_pc.add_argument("--pc-submat",      dest="pseudocounts", action="store_const",
                        const="substitution_matrix_pseudocounts", help="Use substitution matrix pseudocounts")
    grp_pc.add_argument("--pc-constant",    dest="pseudocounts", action="store_const",
                        const="constant_pseudocounts",   help="Use constant pseudocounts ")
    grp_pc.add_argument("--pc-none",        dest="pseudocounts", action="store_const",
                        const="no_pseudocounts", help="Use no pseudocounts")
    grp_pc.add_argument("--pc-count-single",       dest="pseudocount_single",  default=1, type=int,
                        help="Specify number of pseudocounts [default: %(default)s]")
    grp_pc.add_argument("--pc-pair-count",  dest="pseudocount_pair",    default=1, type=int,
                        help="Specify number of pseudocounts for pairwise frequencies [default: %(default)s]")


    scores = parser.add_argument_group("Alternative Coevolution Scores")
    scores.add_argument("--compute-omes",       dest="omes",                action="store_true", default=False,
                        help="Compute OMES scores as in Kass and Horovitz 2002. [default: %(default)s]")
    scores.add_argument("--omes-fodoraldrich",  dest="omes_fodoraldrich",   action="store_true", default=False,
                        help="OMES option: according to Fodor & Aldrich 2004. [default: %(default)s]")
    scores.add_argument("--compute-mi",         dest="mi",                  action="store_true", default=False,
                        help="Compute mutual information (MI) . [default: %(default)s]")
    scores.add_argument("--mi-normalized",      dest="mi_normalized",       action="store_true", default=False,
                        help="MI option: Compute normalized MI according to Martin et al 2005 . [default: %(default)s]")
    scores.add_argument("--mi-pseudocounts",    dest="mi_pseudocounts",     action="store_true", default=False,
                        help="MI option: Compute MI with pseudocounts . [default: %(default)s]")



    args = parser.parse_args()


    if not args.optimize and not args.initrawfile:
        parser.error("--do-not-optimize is only supported when -i (--init-from-raw) is specified!")

    return args


def main():

    # read command line options
    opt = parse_args()

    # print logo
    if opt.logo:
        ccmpred.logo.logo()

    # set OMP environment variable for number of threads
    os.environ['OMP_NUM_THREADS'] = str(opt.num_threads)
    print("Using {0} threads for OMP parallelization.".format(os.environ["OMP_NUM_THREADS"]))

    # instantiate CCMpred
    ccm = CCMpred()

    # specify possible file paths
    ccm.set_alignment_file(opt.alnfile)
    ccm.set_matfile(opt.matfile)
    ccm.set_pdb_file(opt.pdbfile)
    ccm.set_initraw_file(opt.initrawfile)

    # read alignment and possible remove gapped sequences and positions
    ccm.read_alignment(opt.aln_format, opt.max_gap_pos, opt.max_gap_seq)

    # compute sequence weights (in order to reduce sampling bias)
    ccm.compute_sequence_weights(opt.weight, opt.wt_cutoff)

    # compute amino acid counts and frequencies adding pseudo counts for non-observed amino acids
    ccm.compute_frequencies(opt.pseudocounts, opt.pseudocount_single,  opt.pseudocount_pair)

    # read pdb file if CCMpred is setup as a constrained run
    if opt.pdbfile:
        ccm.read_pdb(opt.contact_threshold)


    # if alternative scores are specified: compute these and exit
    if opt.omes:
        ccm.compute_omes(opt.omes_fodoraldrich)
        ccm.write_matrix()
        sys.exit(0)

    if opt.mi:
        ccm.compute_mutual_info(opt.mi_normalized, opt.mi_pseudocounts)
        ccm.write_matrix()
        sys.exit(0)

    # setup L2 regularization
    ccm.specify_regularization(opt.lambda_single, opt.lambda_pair_factor,
                               reg_type="L2", scaling="L", single_prior=opt.single_prior)

    # intialise single and pair potentials either:
    #   - according to regularization priors
    #   - from initrawfile (accounting for removal of many gapped positions, if applicable)
    ccm.intialise_potentials()


    # optimize objective function (pLL or CD/PCD) with optimization algorithm (LBFGS, CG, GD or ADAM)
    if opt.optimize:

        #initialize log object
        ccm.initiate_logging(opt.plot_opt_progress)

        #minimize objective function with corresponding optimization algorithm
        ccm.minimize(opt)
    else:
        print("\nDo not optimize but use model parameters provided by {0}\n".format(opt.initrawfile))




    ### Post Processing


    #specify meta data, and write (corrected) contact matrices to files
    if opt.matfile:

        # Compute contact score (frobenius norm) by recentering potentials
        # TODO: other scores can be added ...
        ccm.compute_contact_matrix(recenter_potentials=True, frob=True)

        # compute corrected contact maps (removing entropy/phylogenetic biases)
        # TODO: other corrections can be added ...
        ccm.compute_correction(
            apc_file=opt.apc_file,
            entropy_correction_file=opt.entropy_correction_file
        )

        ccm.write_matrix()

    # write model parameters in binary format
    if opt.out_binary_raw_file:
        ccm.write_binary_raw(opt.out_binary_raw_file)


    exitcode = 0
    if opt.optimize:
        if ccm.algret['code'] < 0:
            exitcode =-ccm.algret['code']
    sys.exit(exitcode)



if __name__ == '__main__':
    main()
