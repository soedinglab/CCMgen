#!/usr/bin/env python

import argparse
import ccmpred.io.alignment


def main():

    parser = argparse.ArgumentParser(description='Convert Fasta to Psicov format and vice versa.')


    parser.add_argument("infile",  type=str,   help="MSA input file")
    parser.add_argument("outfile", type=str,   help="MSA output file")
    parser.add_argument("--msa-in-format",  dest="msa_in_format", default="psicov",
                        help="Input alignment format [default: '%default']")
    parser.add_argument("--msa-out-format", dest="msa_out_format", default="fasta",
                        help="Output alignment format [default: '%default']")

    args = parser.parse_args()


    msa = ccmpred.io.alignment.read_msa(args.infile, args.msa_in_format)

    with open(args.outfile, "w") as f:
        ccmpred.io.alignment.write_msa(f, msa,
                                       ids=["seq_"+str(i) for i in range(msa.shape[0])],
                                       format=args.msa_out_format
                                       )



if __name__ == '__main__':
    main()
