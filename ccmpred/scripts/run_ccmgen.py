#!/usr/bin/env python

import argparse
import os
from ccmpred import CCMpred
import ccmpred.logo
import ccmpred.io.alignment
import ccmpred.raw
import ccmpred.weighting
import ccmpred.sampling
import ccmpred.gaps
import ccmpred.trees
import ccmpred.parameter_handling
import numpy as np

EPILOG = """
Generate a realistic synthetic multiple sequence alignment (MSA) of protein sequences 
complying constraints from a Markov Random Field model.

In a first step, a Markov Random Field Model will have to be learned from a source protein MSA using 
e.g. CCMpredPy with the -b command. 
This learned model can then be passed to the CCMgen call as RAWFILE.

"""



def parse_args():
    parser = argparse.ArgumentParser(epilog=EPILOG)

    parser.add_argument("rawfile", help="Raw coupling potential file as generated by the CCMpredPy -b option")
    parser.add_argument("outalnfile", help="Output alignment file for sampled sequences.")



    grp_opt = parser.add_argument_group("General Options")
    grp_opt.add_argument("--alnfile", dest="alnfile", metavar="ALN_FILE", type=str,
                               help="Reference alignment file that is used to specify NEFF and NSEQ")
    grp_opt.add_argument("--num-sequences", dest="nseq", type=int, default=2**10,
                        help="Specify the number of sequences to generate to NSEQ "
                             "(does not apply when newick file is specified) [default: %(default)s]")
    grp_opt.add_argument("--max-gap-pos",  dest="max_gap_pos", default=100, type=int,
                        help="Ignore alignment positions with > MAX_GAP_POS percent gaps when reading ALN_FILE. "
                             "[default: %(default)s == no removal of gaps]")
    grp_opt.add_argument("--max-gap-seq",  dest="max_gap_seq",  default=100, type=int,
                        help="Remove sequences with >X percent gaps when reading ALN_FILE. "
                             "[default: %(default)s == no removal of sequences]")
    grp_opt.add_argument("--aln-format", dest="aln_format", type=str, default="fasta",
                        help="Specify format for alignment files [default: %(default)s]")
    grp_opt.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                        help="Specify the number of threads. [default: %(default)s]")




    grp_tr = parser.add_argument_group("Phylogenetic Tree Options")
    grp_tr_me = grp_tr.add_mutually_exclusive_group()
    grp_tr_me.add_argument("--tree-newick",    dest="tree_file", type=str,
                        help="Load tree from newick-formatted file")
    grp_tr_me.add_argument("--tree-binary",    dest="tree_source", action="store_const", const="binary",
                        help="Generate a binary tree with equally distributed branch lengths.")
    grp_tr_me.add_argument("--tree-star",      dest="tree_source", action="store_const", const="star",
                        help="Generate a tree where all leaf nodes are direct descendants of the root node.")
    grp_tr_me.add_argument("--mcmc-sampling",      dest="mcmc", action="store_true", default=False,
                        help="Generate MCMC sample without following tree topology.")



    grp_tr_opt = parser.add_argument_group("Tree Sampling Options")
    grp_tr_opt_me = grp_tr_opt.add_mutually_exclusive_group()
    grp_tr_opt_me.add_argument("--mutation-rate", dest="mutation_rate", type=float,
                        help="Specify constant mutation rate")
    grp_tr_opt_me.add_argument("--mutation-rate-neff", dest="neff", nargs='?', type=float, const=0, default=None,
                        help="Set the mutation rate to approximately hit a target number of effective sequences, Neff "
                             "(calculated as in the HHsuite package (https://github.com/soedinglab/hh-suite)). "
                             "Without specifying NEFF, the value will be determined from ALN_FILE." )


    grp_s0 = parser.add_argument_group("Initial Sequence Options")
    grp_s0_me = grp_s0.add_mutually_exclusive_group()
    grp_s0_me.add_argument("--seq0-mrf",   dest="seq0_mrf", metavar="NMUT", type=int, default=10,
                        help="Start out with an all-alanine sequence and use the MRF model to evolve "
                             "the sequence for NMUT Gibbs steps. [default: NMUT=%(default)s]")
    grp_s0_me.add_argument("--seq0-file",  dest="seq0_file",   metavar="SEQ_FILE", type=str,
                        help="Specify ancestor sequence in SEQ_FILE.")



    grp_mcmc = parser.add_argument_group("MCMC Sampling Options")
    grp_mcmc_me = grp_mcmc.add_mutually_exclusive_group()
    grp_mcmc_me.add_argument("--mcmc-sample-random-gapped",   dest="mcmc_sample_type", action="store_const", const="random-gapped",
                          default="random-gapped",
                          help="Sample sequences starting from random sequences. Gap structure of randomly selected "
                               "input sequences will be copied. Gap positions are not sampled. "
                               "(requires --alnfile option)[default]")
    grp_mcmc_me.add_argument("--mcmc-sample-random",   dest="mcmc_sample_type", action="store_const", const="random",
                          help="Sample sequences starting from random sequences comprised of 20 amino acids. ")
    grp_mcmc_me.add_argument("--mcmc-sample-aln",  dest="mcmc_sample_type", action="store_const", const="aln",
                          help="Sample sequences starting from original sequences (requires setting ALN_FILE).")
    grp_mcmc.add_argument("--mcmc-burn-in", dest="mcmc_burn_in", type=int, default=500,
                          help="Number of Gibbs sampling steps to evolve a Markov chain before a sample is obtained.")




    opt = parser.parse_args()

    if not opt.mcmc:

        if not opt.tree_source and not opt.tree_file:
            parser.error("Need one of the --tree-* options or --mcmc-sampling!")

        if not opt.mutation_rate and opt.neff is None:
            parser.error("Need one of the --mutation-rate* options!")

        if not opt.mutation_rate and opt.neff == 0 and not opt.alnfile:
            parser.error("Need to specify Neff with either --mutation-rate-neff or via an alignment file (--alnfile)!")


    if opt.mcmc:
        if (opt.mcmc_sample_type == "aln" or opt.mcmc_sample_type == "random-gapped") and not opt.alnfile:
            parser.error("Need an alignment file (--alnfile) for use with "
                         "--mcmc-sample-aln and  --mcmc-sample-random-gapped!")

    return opt



def main():
    
    def read_root_sequence(seq0_file, aln_format, print_sequence=True):
        seq0 = ccmpred.io.alignment.read_msa(seq0_file, aln_format)
        seq_N, seq_L = seq0.shape

        if seq_L != ncol:
            print("Length of ancestor sequence must match dimension of MRF model!")
            exit(0)

        if seq_N>1:
            print("You passed a fasta file with more than one sequence as a root sequences! We took the first sequence.")
            print_sequence = True

        if print_sequence:
            print("Ancestor sequence:\n{0}".format("".join([ccmpred.io.alignment.AMINO_ACIDS[c] for c in seq0[0]])))

        return seq0

    # read command line options
    opt = parse_args()

    ccmpred.logo.logo(what_for="ccmgen")

    # set OMP environment variable for number of threads
    os.environ['OMP_NUM_THREADS'] = str(opt.num_threads)
    print("Using {0} threads for OMP parallelization.".format(os.environ["OMP_NUM_THREADS"]))

    # instantiate CCMpred
    ccm = CCMpred()

    # specify possible file paths
    ccm.set_initraw_file(opt.rawfile)


    # read alignment and remove gapped sequences and positions
    if opt.alnfile:
        ccm.set_alignment_file(opt.alnfile)
        ccm.read_alignment(opt.aln_format, opt.max_gap_pos, opt.max_gap_seq)


    #read potentials from binary raw file (possibly remove positions with many gaps)
    ccm.intialise_potentials()
    x = ccmpred.parameter_handling.structured_to_linear(ccm.x_single, ccm.x_pair, nogapstate=True, padding=False)
    ncol = ccm.x_single.shape[0]


    #if MCMC sampling is specified
    if opt.mcmc:
        msa_sampled, neff = ccmpred.sampling.generate_mcmc_sample(
            x, ncol, ccm.msa, size=opt.nseq, burn_in=opt.mcmc_burn_in, sample_type=opt.mcmc_sample_type)

        ids = ["seq {0}".format(i) for i in range(msa_sampled.shape[0])]

    else:

        tree = ccmpred.trees.CCMTree()

        #prepare tree topology
        if opt.tree_file:

            tree.load_tree(opt.tree_file)
            nseq = tree.n_leaves

        else:

            if opt.alnfile:
                nseq = ccm.N
            else:
                nseq = opt.nseq
            tree.specify_tree(nseq, opt.tree_source)


        ids = tree.ids


        # sample alignment with specified mutation rate
        if opt.mutation_rate:
            seq0 = np.zeros((1, ncol), dtype="uint8")
            
            if opt.seq0_mrf and not opt.seq0_file:
                seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, opt.seq0_mrf)
                print("Ancestor sequence (polyA --> {0} gibbs steps --> seq0) :\n{1}".format(
                    opt.seq0_mrf, "".join([ccmpred.io.alignment.AMINO_ACIDS[c] for c in seq0[0]])))

            elif opt.seq0_file:
                seq0 = read_root_sequence(opt.seq0_file, opt.aln_format)

            msa_sampled, neff = ccmpred.sampling.sample_with_mutation_rate(
                tree, nseq, seq0, x, opt.mutation_rate)

        # sample an alignment that has approximately the specified Neff
        else:            
            seq0 = None
            
            if opt.alnfile:
                neff = ccm.neff_entropy
            else:
                neff = opt.neff
            
            if opt.seq0_file:              
                seq0 = read_root_sequence(opt.seq0_file, opt.aln_format)
                    
            msa_sampled, neff = ccmpred.sampling.sample_to_neff_increasingly(
            tree, nseq, neff, ncol, x, opt.seq0_mrf, root_seq=seq0)
                


    # if gappy positions have been removed
    # insert columns with gaps at that position
    if ccm.max_gap_pos < 100:
        msa_sampled = ccmpred.gaps.backinsert_gapped_positions_aln(
            msa_sampled, ccm.gapped_positions
        )


    print("\nWriting sampled alignment to {0}".format(opt.outalnfile))
    with open(opt.outalnfile, "w") as f:
        descs=["synthetic sequence generated with CCMgen" for _ in range(msa_sampled.shape[0])]
        ccmpred.io.alignment.write_msa(f, msa_sampled, ids, is_indices=True, format=opt.aln_format, descriptions=descs)

        
if __name__ == '__main__':
    main()
