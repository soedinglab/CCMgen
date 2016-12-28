#!/usr/bin/env python
import argparse
import numpy as np
import sys
import json

import ccmpred.metadata
import ccmpred.weighting
import ccmpred.scoring
import ccmpred.pseudocounts
import ccmpred.raw
import ccmpred.logo
import ccmpred.io.alignment as aln
import ccmpred.centering
import ccmpred.regularization
import ccmpred.model_probabilities
import ccmpred.gaps

import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.objfun.treecd as treecd

import ccmpred.algorithm.gradient_descent
import ccmpred.algorithm.conjugate_gradients
import ccmpred.algorithm.numdiff

EPILOG = """
CCMpred is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods. From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model with 21 states for amino acids and gaps. The L2 norms of the pairwise coupling potentials will be written to the output matfile.
"""

REG_L2_SCALING= {
    "L" : lambda msa : msa.shape[1] - 1,
    "diversity" : lambda msa: msa.shape[1] / np.sqrt(msa.shape[0]),
    "1": lambda msa: 1
}

ALGORITHMS = {
    "gradient_descent": lambda of, x0, opt: ccmpred.algorithm.gradient_descent.minimize(of, x0, opt.numiter, alpha0=5e-3, alpha_decay=1e1),
    "conjugate_gradients": lambda of, x0, opt: ccmpred.algorithm.conjugate_gradients.minimize(of, x0, opt.numiter, epsilon=1e-5),
    "numerical_differentiation": lambda of, x0, opt: ccmpred.algorithm.numdiff.numdiff(of, x0),
}


class TreeCDAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import Bio.Phylo
        treefile, seq0file = values

        tree = Bio.Phylo.read(treefile, "newick")
        seq0, id0 = aln.read_msa(seq0file, parser.values.aln_format, return_identifiers=True)

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

    parser.add_argument("-n", "--num-iterations", dest="numiter", default=100, type=int, help="Specify the number of iterations [default: %(default)s]")
    parser.add_argument("-i", "--init-from-raw", dest="initrawfile", default=None, help="Init potentials from raw file")
    parser.add_argument("-r", "--write-raw", dest="outrawfile", default=None, help="Write potentials to raw file")
    parser.add_argument("-b", "--write-msgpack", dest="outmsgpackfile", default=None, help="Write potentials to MessagePack file")
    parser.add_argument("-m", "--write-modelprob-msgpack", dest="outmodelprobmsgpackfile", default=None, help="Write model probabilities as MessagePack file")
    parser.add_argument("-A", "--disable_apc",  dest="disable_apc", action="store_true", default=False, help="Disable average product correction (APC)")
    parser.add_argument("--aln-format", dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--no-logo", dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo")

    parser.add_argument("alnfile", help="Input alignment file to use")
    parser.add_argument("matfile", help="Output matrix file to write")

    grp_of = parser.add_argument_group("Objective Functions")
    grp_of.add_argument("--ofn-pll", dest="objfun", action="store_const", const=pll.PseudoLikelihood, default=pll.PseudoLikelihood, help="Use pseudo-log-likelihood (default)")
    grp_of.add_argument("--ofn-pcd", dest="objfun", action="store_const", const=cd.ContrastiveDivergence, help="Use Persistent Contrastive Divergence")
    grp_of.add_argument("--ofn-tree-cd", action=TreeCDAction, metavar=("TREEFILE", "ANCESTORFILE"), nargs=2, type=str, help="Use Tree-controlled Contrastive Divergence, loading tree data from TREEFILE and ancestral sequence data from ANCESTORFILE")

    grp_al = parser.add_argument_group("Algorithms")
    grp_al.add_argument("--alg-gd", dest="algorithm", action="store_const", const=ALGORITHMS['gradient_descent'], default=ALGORITHMS['gradient_descent'], help='Use gradient descent (default)')
    grp_al.add_argument("--alg-cg", dest="algorithm", action="store_const", const=ALGORITHMS['conjugate_gradients'], help='Use conjugate gradients')
    grp_al.add_argument("--alg-nd", dest="algorithm", action="store_const", const=ALGORITHMS['numerical_differentiation'], help='Debug gradients with numerical differentiation')

    grp_wt = parser.add_argument_group("Weighting")
    grp_wt.add_argument("--wt-simple",          dest="weight", action="store_const", const=ccmpred.weighting.weights_simple, default=ccmpred.weighting.weights_simple, help='Use simple weighting (default)')
    grp_wt.add_argument("--wt-henikoff",        dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff, help='Use simple Henikoff weighting')
    grp_wt.add_argument("--wt-henikoff_pair",   dest="weight", action="store_const", const=ccmpred.weighting.weights_henikoff_pair, help='Use Henikoff pair weighting ')
    grp_wt.add_argument("--wt-uniform",         dest="weight", action="store_const", const=ccmpred.weighting.weights_uniform, help='Use uniform weighting')

    grp_rg = parser.add_argument_group("Regularization")
    grp_rg.add_argument("--reg-l2", dest="regularization", action=RegL2Action, type=float, nargs=2, metavar=("LAMBDA_SINGLE", "LAMBDA_PAIR"), default=lambda msa, centering, scaling: ccmpred.regularization.L2(10, 0.2 * scaling, centering), help='Use L2 regularization with coefficients LAMBDA_SINGLE, LAMBDA_PAIR * SCALING;  (default: 10 0.2)')
    grp_rg.add_argument("--reg-l2-scale_by_L",      dest="scaling", action="store_const", const=REG_L2_SCALING["L"], default=REG_L2_SCALING["L"], help=" LAMBDA_PAIR * (L-1) (default)")
    grp_rg.add_argument("--reg-l2-scale_by_div",    dest="scaling", action="store_const", const=REG_L2_SCALING["diversity"], help="LAMBDA_PAIR * (L/sqrt(N))")
    grp_rg.add_argument("--reg-l2-noscaling",       dest="scaling", action="store_const", const=REG_L2_SCALING["1"], help="LAMBDA_PAIR")

    grp_gp = parser.add_argument_group("Gap Treatment")
    grp_gp.add_argument("--max_gap_ratio",  dest="max_gap_ratio", default=100, type=int, help="Remove alignment positions with >x% gaps [default: %(default)s  = no removal]")
    grp_gp.add_argument("--wt-ignore-gaps", dest="ignore_gaps", action="store_true", default=False, help="Do not count gaps as identical amino acids during reweighting of sequences.")

    grp_pc = parser.add_argument_group("Pseudocounts")
    grp_pc.add_argument("--pc-submat",      dest="pseudocounts", action=StoreConstParametersAction, default=ccmpred.pseudocounts.substitution_matrix_pseudocounts, const=ccmpred.pseudocounts.substitution_matrix_pseudocounts, nargs="?", metavar="N", type=float, arg_default=1, help="Use N substitution matrix pseudocounts (default) (by default, N=1)")
    grp_pc.add_argument("--pc-constant",    dest="pseudocounts", action=StoreConstParametersAction, const=ccmpred.pseudocounts.constant_pseudocounts,   metavar="N", nargs="?", type=float, arg_default=1, help="Use N constant pseudocounts (by default, N=1)")
    grp_pc.add_argument("--pc-uniform",     dest="pseudocounts", action=StoreConstParametersAction, const=ccmpred.pseudocounts.uniform_pseudocounts,    metavar="N", nargs="?", type=float, arg_default=1, help="Use N uniform pseudocounts, e.g 1/21 (by default, N=1)")
    grp_pc.add_argument("--pc-none",        dest="pseudocounts", action="store_const", const=[ccmpred.pseudocounts.no_pseudocounts, 0], help="Use no pseudocounts")
    grp_pc.add_argument("--pc-pair-count",  dest="pseudocount_pair_count", default=None, type=int, help="Specify a separate number of pseudocounts for pairwise frequencies (default: use same as single counts)")

    grp_db = parser.add_argument_group("Debug Options")
    grp_db.add_argument("--write-trajectory", dest="trajectoryfile", default=None, help="Write trajectory to files with format expression")
    grp_db.add_argument("--write-cd-alignment", dest="cd_alnfile", default=None, metavar="ALNFILE", help="Write PSICOV-formatted sampled alignment to ALNFILE")
    grp_db.add_argument("-c", "--compare-to-raw", dest="comparerawfile", default=None, help="Compare potentials to raw file")

    args = parser.parse_args()

    if args.cd_alnfile and args.objfun not in (cd.ContrastiveDivergence, treecd.TreeContrastiveDivergence):
        parser.error("--write-cd-alignment is only supported for (tree) contrastive divergence!")

    return args


def main():

    opt = parse_args()

    if opt.logo:
        ccmpred.logo.logo()

    msa = aln.read_msa(opt.alnfile, opt.aln_format)
    msa, gapped_positions = ccmpred.gaps.remove_gapped_positions(msa, opt.max_gap_ratio)

    weights = opt.weight(msa, opt.ignore_gaps)

    print("Reweighted {0} sequences to Neff={1:g} (min={2:g}, mean={3:g}, max={4:g}) using {5} and ignore_gaps={6}".format(msa.shape[0], np.sum(weights), np.min(weights), np.mean(weights), np.max(weights), opt.weight.__name__, opt.ignore_gaps))

    if not hasattr(opt, "objfun_args"):
        opt.objfun_args = []

    if not hasattr(opt, "objfun_kwargs"):
        opt.objfun_kwargs = {}

    freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, opt.pseudocounts[0], pseudocount_n_single=opt.pseudocounts[1], pseudocount_n_pair=opt.pseudocount_pair_count)

    if opt.initrawfile:
        raw = ccmpred.raw.parse(opt.initrawfile)
        centering = raw.x_single.copy()
        regularization = opt.regularization(msa, centering, opt.scaling(msa))
        x0, f = opt.objfun.init_from_raw(msa, freqs, weights, raw, regularization, *opt.objfun_args, **opt.objfun_kwargs)

    else:
        centering = ccmpred.centering.calculate(msa, freqs)
        regularization = opt.regularization(msa, centering, opt.scaling(msa))
        x0, f = opt.objfun.init_from_default(msa, freqs, weights, regularization, *opt.objfun_args, **opt.objfun_kwargs)

    if opt.comparerawfile:
        craw = ccmpred.raw.parse(opt.comparerawfile)
        f.compare_raw = craw

    f.trajectory_file = opt.trajectoryfile

    print("Will optimize {0} {1} variables with {2}\n".format(x0.size, x0.dtype, f))

    fx, x, algret = opt.algorithm(f, x0, opt)

    condition = "Finished" if algret['code'] >= 0 else "Exited"

    print("\n{0} with code {code} -- {message}".format(condition, **algret))

    meta = ccmpred.metadata.create(opt, regularization, msa, weights, f, fx, algret)
    res = f.finalize(x, meta)

    if opt.cd_alnfile and hasattr(f, 'msa_sampled'):
        print("Writing sampled alignment to {0}".format(opt.cd_alnfile))
        msa_sampled = f.msa_sampled

        with open(opt.cd_alnfile, "w") as f:
            aln.write_msa_psicov(f, msa_sampled)

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
        ccmpred.model_probabilities.write_msgpack(opt.outmodelprobmsgpackfile, res, msa, weights, freqs[1], regularization.lambda_pair)

    print("Writing summed score matrix (with APC={0}) to {1}".format(not opt.disable_apc, opt.matfile))
    mat = ccmpred.scoring.frobenius_score(res.x_pair)
    if not opt.disable_apc:
        mat = ccmpred.scoring.apc(mat)
    np.savetxt(opt.matfile, mat)
    with open(opt.matfile,'a') as f:
        f.write("#>META> ".encode("utf-8") + json.dumps(meta).encode("utf-8") + b"\n")


    exitcode = 0 if algret['code'] > 0 else -algret['code']
    sys.exit(exitcode)


if __name__ == '__main__':
    main()
