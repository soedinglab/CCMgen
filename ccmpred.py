#!/usr/bin/env python

import ccmpred.io.alignment as aln
# import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.algorithm.gradient_descent as gd

msa = aln.read_msa_psicov("data/1atzA.aln")

# x0, of = pll.PseudoLikelihood.init_from_default(msa)
x0, of = cd.ContrastiveDivergence.init_from_default(msa)

fx, x = gd.minimize(of, x0, 100, step=lambda i: 1e-3 / (1 + i / 25))

print("Finished with fx = {fx}".format(fx=fx))
