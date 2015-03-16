#!/usr/bin/env python
import optparse

import ccmpred.weighting as cw
import ccmpred.io.alignment as aln

import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd

import ccmpred.algorithm.gradient_descent as gd
import ccmpred.algorithm.conjugate_gradients as cg

OBJECTIVE_FUNCTIONS = {
    "pll": pll.PseudoLikelihood,
    "pcd": cd.ContrastiveDivergence,
}

ALGORITHMS = {
    "gradient_descent": lambda of, x0, opt: gd.minimize(of, x0, opt.numiter, step=lambda i: 1e-3 / (1 + i / 25)),
    "conjugate_gradients": lambda of, x0, opt: cg.minimize(of, x0, opt.numiter),
}


def main():
    parser = optparse.OptionParser(usage="%prog [options] alnfile matfile")
    parser.add_option("--objective-function", dest="objfun", default="pcd", choices=list(OBJECTIVE_FUNCTIONS.keys()), help="Specify the objective function ({0}) to optimize [default: \"%default\"]".format(", ".join(OBJECTIVE_FUNCTIONS.keys())))
    parser.add_option("--algorithm", dest="algorithm", default="gradient_descent", choices=list(ALGORITHMS.keys()), help="Specify the algorithm ({0}) for optimization [default: \"%default\"]".format(", ".join(ALGORITHMS.keys())))
    parser.add_option("-n", "--num-iterations", dest="numiter", default=100, type=int, help="Specify the number of iterations [default: %default]")

    opt, args = parser.parse_args()
    if len(args) != 2:
        parser.error("Need exactly 2 positional arguments!")

    alnfile, matfile = args

    print(opt)

    msa = aln.read_msa_psicov(alnfile)
    weights = cw.weights_simple(msa)

    x0, of = OBJECTIVE_FUNCTIONS[opt.objfun].init_from_default(msa, weights)

    fx, x = ALGORITHMS[opt.algorithm](of, x0, opt)

    print("Finished with fx = {fx}".format(fx=fx))


if __name__ == '__main__':
    main()
