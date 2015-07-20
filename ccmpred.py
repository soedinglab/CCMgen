#!/usr/bin/env python
import optparse
import numpy as np

import ccmpred.weighting
import ccmpred.scoring
import ccmpred.raw
import ccmpred.logo
import ccmpred.io.alignment as aln

import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.objfun.treecd as treecd

import ccmpred.algorithm.gradient_descent
import ccmpred.algorithm.conjugate_gradients

ALGORITHMS = {
    "gradient_descent": lambda of, x0, opt: ccmpred.algorithm.gradient_descent.minimize(of, x0, opt.numiter, alpha0=1e-4, alpha_decay=1000),
    "conjugate_gradients": lambda of, x0, opt: ccmpred.algorithm.conjugate_gradients.minimize(of, x0, opt.numiter),
}


def cb_treecd(option, opt, value, parser):
    import Bio.Phylo
    treefile, seq0file = value

    tree = Bio.Phylo.read(treefile, "newick")
    seq0, id0 = aln.read_msa(seq0file, parser.values.aln_format, return_identifiers=True)

    parser.values.objfun_args = [tree, seq0, id0]
    parser.values.objfun = treecd.TreeContrastiveDivergence


def main():
    parser = optparse.OptionParser(usage="%prog [options] alnfile matfile")
    parser.add_option("-n", "--num-iterations", dest="numiter", default=100, type=int, help="Specify the number of iterations [default: %default]")

    parser.add_option("-i", "--init-from-raw", dest="initrawfile", default=None, help="Init potentials from raw file")
    parser.add_option("-r", "--write-raw", dest="outrawfile", default=None, help="Write potentials to raw file")
    parser.add_option("-b", "--write-msgpack", dest="outmsgpackfile", default=None, help="Write potentials to MessagePack file")
    parser.add_option("--aln-format", dest="aln_format", default="psicov", help="File format for MSAs [default: \"%default\"]")
    parser.add_option("--no-logo", dest="logo", default=True, action="store_false", help="Disable showing the CCMpred logo")

    grp_of = parser.add_option_group("Objective Functions")
    grp_of.add_option("--ofn-pll", dest="objfun", action="store_const", const=pll.PseudoLikelihood, default=pll.PseudoLikelihood, help="Use pseudo-log-likelihood (default)")
    grp_of.add_option("--ofn-pcd", dest="objfun", action="store_const", const=cd.ContrastiveDivergence, help="Use Persistent Contrastive Divergence")
    grp_of.add_option("--ofn-tree-cd", action="callback", metavar="TREEFILE ANCESTORFILE", callback=cb_treecd, nargs=2, type=str, help="Use Tree-controlled Contrastive Divergence, loading tree data from TREEFILE and ancestral sequence data from ANCESTORFILE")

    grp_al = parser.add_option_group("Algorithms")
    grp_al.add_option("--alg-gd", dest="algorithm", action="store_const", const=ALGORITHMS['gradient_descent'], default=ALGORITHMS['gradient_descent'], help='Use gradient descent (default)')
    grp_al.add_option("--alg-cg", dest="algorithm", action="store_const", const=ALGORITHMS['conjugate_gradients'], help='Use conjugate gradients')

    grp_wt = parser.add_option_group("Weighting")
    grp_wt.add_option("--wt-simple", dest="weight", action="store_const", const=ccmpred.weighting.weights_simple, default=ccmpred.weighting.weights_simple, help='Use simple weighting (default)')
    grp_wt.add_option("--wt-uniform", dest="weight", action="store_const", const=ccmpred.weighting.weights_uniform, help='Use uniform weighting')

    opt, args = parser.parse_args()

    if len(args) != 2:
        parser.error("Need exactly 2 positional arguments!")

    if opt.logo:
        ccmpred.logo.logo()

    alnfile, matfile = args

    msa = aln.read_msa(alnfile, opt.aln_format)
    weights = opt.weight(msa)

    if not hasattr(opt, "objfun_args"):
        opt.objfun_args = []

    if not hasattr(opt, "objfun_kwargs"):
        opt.objfun_kwargs = {}

    if opt.initrawfile:
        raw = ccmpred.raw.parse(opt.initrawfile)
        x0, f = opt.objfun.init_from_raw(msa, weights, raw, *opt.objfun_args, **opt.objfun_kwargs)

    else:
        x0, f = opt.objfun.init_from_default(msa, weights, *opt.objfun_args, **opt.objfun_kwargs)

    fx, x = opt.algorithm(f, x0, opt)

    print("Finished with fx = {fx}".format(fx=fx))

    res = f.finalize(x)

    if opt.outrawfile:
        ccmpred.raw.write_oldraw(opt.outrawfile, res)

    if opt.outmsgpackfile:
        ccmpred.raw.write_msgpack(opt.outmsgpackfile, res)

    mat = ccmpred.scoring.frobenius_score(res.x_pair)

    np.savetxt(matfile, mat)


if __name__ == '__main__':
    main()
