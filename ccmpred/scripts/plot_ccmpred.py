#!/usr/bin/env python

"""
Usage: plot_ccmpred.py

Various plotting functionalities
"""

import os
import sys
import argparse
import ccmpred.raw as raw
import ccmpred.weighting
from ccmpred.pseudocounts import PseudoCounts
import ccmpred.io as io
import ccmpred.io.contactmatrix as io_cm
import ccmpred.plotting as plot
import pandas as pd
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='Various Plotting Functionalities.')
    subparsers = parser.add_subparsers(title="Plot types", dest="plot_types")


    #parent parsers for common flags
    parent_parser_out = argparse.ArgumentParser(add_help=False)
    requiredNamed = parent_parser_out.add_argument_group('Required Output Arguments')
    requiredNamed.add_argument('-o', '--plot-file', dest='plot_file', type=str, required=True,
                               help='Path to plot file')



    #parser for contact map
    parser_cmap = subparsers.add_parser('cmap', parents=[parent_parser_out],
                                        help="Specify options for plotting a Contact Map")

    cmap_in_req = parser_cmap.add_argument_group('Required Inputs')
    mutual_excl = cmap_in_req.add_mutually_exclusive_group(required=True)
    mutual_excl.add_argument('--mat-file', dest='mat_file', type=str, help='path to mat file')
    mutual_excl.add_argument('--braw-file', dest='braw_file', type=str,help='path to binary raw coupling file')

    cmap_in = parser_cmap.add_argument_group('Optional Inputs')
    cmap_in.add_argument('-p', '--pdb-file', dest='pdb_file', type=str, default=None,
                        help=' PDB file (renumbered starting from 1) for distance matrix.')
    cmap_in.add_argument('-a', '--alignment-file', dest='aln_file', type=str, default=None,
                         help='path to alignment file')
    cmap_in.add_argument("--aln-format", dest="aln_format", default="fasta",
                                   help="File format for MSAs [default: \"%(default)s\"]")

    cmap_options = parser_cmap.add_argument_group('Further Settings for Contact Map Plot')
    cmap_options.add_argument('--seq-sep', dest='seqsep', type=int, default=6, help='Minimal sequence separation')
    cmap_options.add_argument('--contact-threshold', dest='contact_threshold', type=int, default=8,
                        help='Contact definition as maximal C_beta distance between residue pairs.')
    cmap_options.add_argument("--apc", action="store_true", default=False, help="Apply average product correction")
    cmap_options.add_argument("--entropy-correction", dest='entropy_correction', action="store_true", default=False, help="Apply entropy correction")


    # parser for aa distribution plot
    parser_aa_dist = subparsers.add_parser('aa-dist', parents=[parent_parser_out],
                                           help="Specify options for plotting the amino acid distribution in an alignment")

    aadist_in_req = parser_aa_dist.add_argument_group('Required Inputs')
    aadist_in_req.add_argument('-a', '--alignment-file', dest='aln_file', type=str, required=True,
                               help='path to alignment file')
    aadist_in_req.add_argument("--aln-format", dest="aln_format", default="psicov",
                               help="File format for MSAs [default: \"%(default)s\"]")


    # parser for alignment statistics plot
    parser_aln_stats = subparsers.add_parser(
        'aln-stats', parents=[parent_parser_out],
        help="Specify options for plotting the alignment statistics of two alignments against each other")

    alnstats_in_req = parser_aln_stats.add_argument_group('Required Inputs')
    alnstats_in_req.add_argument('-a', '--alignment-file', dest='aln_file', type=str, required=True,
                               help='path to alignment file')
    alnstats_in_req.add_argument("--aln-format", dest="aln_format", default="psicov",
                               help="File format for MSAs [default: \"%(default)s\"]")
    alnstats_in_req.add_argument('-s', '--sampled-alignment-file', dest='sample_aln_file', type=str, required=True,
                               help='path to sampled alignment' )


    args = parser.parse_args()

    if args.plot_types == "cmap":
        if args.entropy_correction and args.alignment_file is None:
            print("Alignment file (-a) must be specified to compute entropy correction!")

        if args.entropy_correction and args.braw_file is None:
            print("Binary Raw file (-b) must be specified to compute entropy correction!")

    return args

def plot_contact_map(alignment_file, aln_format, braw_file, mat_file, pdb_file, plot_file,
                     entropy_correction, apc, seqsep, contact_threshold):

    pseudocounts = None
    mat = None
    gaps_percentage_plot = None
    protein = None


    if entropy_correction and (alignment_file is None or braw_file is None):
        print("Entropy correction requires specification of alignment file and binary raw couplign file!")
        sys.exit(1)

    if alignment_file is not None:
        protein = os.path.basename(alignment_file).split(".")[0]
        alignment = io.read_msa(alignment_file, aln_format)

        # compute sequence weights
        weights = ccmpred.weighting.weights_simple(alignment, 0.8)

        # compute frequencies
        pseudocounts = PseudoCounts(alignment, weights)
        pseudocounts.calculate_frequencies(
            'uniform_pseudocounts', 1, 1, remove_gaps=False
        )

        gaps_percentage_plot = plot.plot_percentage_gaps_per_position(pseudocounts.counts[0], plot_file=None)

    if braw_file is not None:

        protein = os.path.basename(braw_file).split(".")[0]

        braw = raw.parse_msgpack(braw_file)
        meta_info = braw.meta

        # compute frobenius score from couplings
        mat = io_cm.frobenius_score(braw.x_pair)

        if entropy_correction:

            scaling_factor_eta, mat = io_cm.compute_local_correction(
                pseudocounts.freqs[0],
                braw.x_pair,
                meta_info['workflow'][0]['msafile']['neff'],
                meta_info['workflow'][0]['regularization']['lambda_pair'],
                mat,
                entropy=True
            )
        elif apc:
            mat = io_cm.apc(mat)

    if mat_file is not None:

        protein = os.path.basename(mat_file).split(".")[0]

        mat, meta_info = io_cm.read_matrix(mat_file)

        if apc:
            mat = io_cm.apc(mat)

    L = len(mat)
    indices_upper_tri_i, indices_upper_tri_j = np.triu_indices(L, seqsep)

    plot_matrix = pd.DataFrame()
    plot_matrix['residue_i'] = indices_upper_tri_i + 1
    plot_matrix['residue_j'] = indices_upper_tri_j + 1
    plot_matrix['confidence'] = mat[indices_upper_tri_i, indices_upper_tri_j]

    if pdb_file is not None:
        # compute distance map from pdb file
        observed_distances = io.distance_map(pdb_file, L)
        plot_matrix['distance'] = observed_distances[indices_upper_tri_i, indices_upper_tri_j]
        plot_matrix['contact'] = ((plot_matrix.distance < contact_threshold) * 1).tolist()


    plot_title="Contact Map for protein {0}".format(protein)

    # Plot Contact Map
    plot.plot_contact_map_someScore_plotly(plot_matrix, plot_title, seqsep, gaps_percentage_plot, plot_file)

def plot_aminoacid_distribution(alignment_file, aln_format, plot_file):

    protein = os.path.basename(alignment_file).split(".")[0]

    #read alignment
    try:
        alignment = io.read_msa(alignment_file, aln_format)
    except OSError as e:
        print("Problems reading alignment file {0}: {1}!".format(alignment_file, e))
        sys.exit(0)

    N = alignment.shape[0]
    L = alignment.shape[1]
    diversity = np.sqrt(N) / L

    # compute sequence weights
    weights = ccmpred.weighting.weights_simple(alignment, 0.8, False)

    # compute frequencies
    pseudocounts = PseudoCounts(alignment, weights)
    pseudocounts.calculate_frequencies(
        'uniform_pseudocounts', 1, 1, remove_gaps=False
    )

    #plot
    plot.plot_alignment(
        pseudocounts.counts[0],
        "Amino Acid Distribution in Alignment for {0} (N={1}, L={2}, diversity={3})".format(
            protein, N, L, np.round(diversity, decimals=3)), plot_file
    )

def plot_alignment_statistics(alignment_file, sample_aln_file, aln_format, plot_file):


    #read alignment
    try:
        alignment = io.read_msa(alignment_file, aln_format)
    except OSError as e:
        print("Problems reading alignment file {0}: {1}!".format(alignment_file, e))
        sys.exit(0)

    try:
        sampled_alignment = io.read_msa(sample_aln_file, aln_format)
    except OSError as e:
        print("Problems reading alignment file {0}: {1}!".format(sample_aln_file, e))
        sys.exit(0)



    # compute sequence weights for observed sequences
    weights = ccmpred.weighting.weights_simple(alignment, 0.8)

    # compute observed amino acid frequencies
    pseudocounts = PseudoCounts(alignment, weights)
    pseudocounts.calculate_frequencies(
        'uniform_pseudocounts', 1, 1, remove_gaps=False
    )
    single_freq_observed, pairwise_freq_observed = pseudocounts.freqs





    # compute sequence weights for sampled sequences (usually all sampled sequences obtain weight = 1 )
    weights_sampled = ccmpred.weighting.weights_simple(sampled_alignment, 0.8)

    # compute sampled amino acid frequencies
    pseudocounts = PseudoCounts(sampled_alignment, weights_sampled)
    pseudocounts.calculate_frequencies(
        'uniform_pseudocounts', 1, 1, remove_gaps=False
    )
    single_freq_sampled, pairwise_freq_sampled = pseudocounts.freqs

    # degap the frequencies (ignore gap frequencies)
    single_freq_observed = pseudocounts.degap(single_freq_observed, False)
    single_freq_sampled = pseudocounts.degap(single_freq_sampled, False)
    pairwise_freq_observed = pseudocounts.degap(pairwise_freq_observed, False)
    pairwise_freq_sampled = pseudocounts.degap(pairwise_freq_sampled, False)


    # plot
    plot.plot_empirical_vs_model_statistics(
        single_freq_observed, single_freq_sampled,
        pairwise_freq_observed, pairwise_freq_sampled,
        plot_file)



def main():

    args = parse_args()

    if args.plot_types == "cmap":
        print("Write plot for contact map to {0}".format(args.plot_file))

        plot_contact_map(
            args.aln_file, args.aln_format,
            args.braw_file, args.mat_file, args.pdb_file, args.plot_file,
            args.entropy_correction, args.apc,
            args.seqsep, args.contact_threshold
        )

    if args.plot_types == "aa-dist":
        print("Write plot for amino acid distribution in alignment to {0}".format(args.plot_file))

        plot_aminoacid_distribution(
            args.aln_file, args.aln_format,
            args.plot_file
        )

    if args.plot_types == "aln-stats":
        print("Write plot for alignment statistics to {0}".format(args.plot_file))

        plot_alignment_statistics(
            args.aln_file, args.sample_aln_file, args.aln_format,
            args.plot_file
        )



if __name__ == '__main__':
    main()