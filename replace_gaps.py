#!/usr/bin/env python
import optparse
import numpy as np

import ccmpred.io.alignment
import ccmpred.raw
import ccmpred.counts
import ccmpred.gaps


def replacement_mrf(option, opt, value, parser):
    raw = ccmpred.raw.parse(value)

    def inner(msa):
        # TODO compute probs
        return msa

    parser.values.replacement = inner


def replacement_consensus(option, opt, value, parser):
    parser.values.replacement = ccmpred.gaps.remove_gaps_consensus


def replacement_col_freqs(option, opt, value, parser):

    def inner(msa):
        counts = ccmpred.counts.single_counts(msa)
        counts[:, 20] = 0

        counts /= np.sum(counts, axis=1)[:, np.newaxis]

        return ccmpred.gaps.remove_gaps_probs(msa, counts)

    parser.values.replacement = inner


def main():
    parser = optparse.OptionParser(usage="%prog [options] msa_in_file msa_out_file")

    parser.add_option("--with-mrf", dest="replacement", action="callback", callback=replacement_mrf, type=str, nargs=1, metavar="MRFFILE", help="Remove gaps with character probabilities from MRF model")
    parser.add_option("--with-consensus", dest="replacement", action="callback", callback=replacement_consensus, help="Remove gaps with consensus characters")
    parser.add_option("--with-col-freqs", dest="replacement", action="callback", callback=replacement_col_freqs, help="Remove gaps with column character frequencies")
    parser.add_option("--msa-in-format", dest="msa_in_format", default="psicov", help="Input alignment format [default: '%default']")

    opt, args = parser.parse_args()

    if not opt.replacement:
        parser.error("Need to specify one of the --with-* options!")

    if not len(args) == 2:
        parser.error("Need exactly two positional arguments!")

    msa_in_file, msa_out_file = args

    msa = ccmpred.io.alignment.read_msa(msa_in_file, opt.msa_in_format)
    msa_nogaps = opt.replacement(msa)

    with open(msa_out_file, "w") as f:
        ccmpred.io.alignment.write_msa_psicov(f, msa_nogaps)


if __name__ == '__main__':
    main()
