#!/usr/bin/env python

import ccmpred.io.alignment as aln
import ccmpred.objfun.pll as pll
import ccmpred.algorithm.conjugate_gradients as cg

msa = aln.read_msa_psicov("data/1atzA.aln")

x0, of = pll.PseudoLikelihood.init_from_default(msa)

fx, x = cg.minimize(of, x0, 100)

print("Finished with fx = {fx}".format(fx=fx))
