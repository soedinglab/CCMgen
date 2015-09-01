#!/usr/bin/env python
import numpy as np
import ccmpred.io.alignment
import ccmpred.objfun.cd as cd
import ccmpred.objfun.treecd as treecd
import ccmpred.raw
import ccmpred.trees
import ccmpred.weighting
import Bio.Phylo
import sys


def cb_tree_newick(option, opt_str, value, parser, *args, **kwargs):
    parser.values.tree_source = lambda opt, id0: Bio.Phylo.read(value, "newick")


def cb_tree_binary(option, opt_str, value, parser, *args, **kwargs):
    splits, depth = int(value[0]), float(value[1])
    parser.values.tree_source = lambda opt, id0: ccmpred.trees.create_binary_tree(splits, depth, root_name=id0[0])


def cb_tree_star(option, opt_str, value, parser, *args, **kwargs):
    leaves, depth = int(value[0]), float(value[1])
    parser.values.tree_source = lambda opt, id0: ccmpred.trees.create_star_tree(leaves, depth, root_name=id0[0])


def cb_seq0_file(option, opt_str, value, parser, *args, **kwargs):
    parser.values.seq0_source = lambda opt, raw: ccmpred.io.alignment.read_msa(value, opt.aln_format, return_identifiers=True)


def cb_seq0_mrf(option, opt_str, value, parser, *args, **kwargs):

    nmut = value

    def get_seq0_mrf(opt, raw):

        # generate a poly-A alignment
        seq0 = np.zeros((1, raw.ncol), dtype="uint8")
        id0 = ["root"]

        x = cd.structured_to_linear(raw.x_single, raw.x_pair)

        for _ in range(nmut):
            seq0 = cd.cext.sample_sequences(seq0, x)

        return seq0, id0

    parser.values.seq0_source = get_seq0_mrf


def cb_mutation_rate_neff_aln(option, opt_str, value, parser, *args, **kwargs):
    msa = ccmpred.io.alignment.read_msa(value, parser.values.aln_format)
    neff = np.sum(ccmpred.weighting.weights_simple(msa))

    print("Target Neff is {0}".format(neff))

    parser.values.mutation_rate_neff = neff


def get_options():
    import optparse
    parser = optparse.OptionParser(usage="%prog [options] rawfile outalignment")
    parser.add_option("--aln-format", dest="aln_format", default="fasta", help="Specify format for alignment files [default: \"%default\"]")
    parser.add_option("--mutation-rate", dest="mutation_rate", default=1.7, type=float, help="Specify mutation rate [default: %default]")
    parser.add_option("--mutation-rate-sample", dest="mutation_rate_sample", default=False, action="store_true", help="Sample mutation rates and Neffs")
    parser.add_option("--mutation-rate-neff", dest="mutation_rate_neff", type=float, metavar="NEFF", help="Set mutation rate to generate alignment of given number of effective sequences")
    parser.add_option("--mutation-rate-neff-aln", action="callback", callback=cb_mutation_rate_neff_aln, nargs=1, type=str, metavar="ALNFILE", help="Set mutation rate to generate alignment with same number of effective sequences as given alignment")

    parser.add_option("--tree-newick", dest="tree_source", default=None, metavar="DNDFILE", action="callback", nargs=1, type=str, callback=cb_tree_newick, help="Load tree from newick-formatted file DNDFILE")
    parser.add_option("--tree-binary", metavar="SPLITS DEPTH", action="callback", nargs=2, type=str, callback=cb_tree_binary, help="Generate binary tree with 2^SPLITS sequences and total depth DEPTH")
    parser.add_option("--tree-star", metavar="LEAVES DEPTH", action="callback", nargs=2, type=str, callback=cb_tree_star, help="Generate star tree with LEAVES sequences and total depth DEPTH")

    parser.add_option("--seq0-file", dest="seq0_source", default=None, metavar="ALNFILE", action="callback", nargs=1, type=str, callback=cb_seq0_file, help="Get initial sequenc from ALNFILE")
    parser.add_option("--seq0-mrf", metavar="NMUT", action="callback", nargs=1, type=int, callback=cb_seq0_mrf, help="Sample initial sequence from MRF by mutating NMUT times from a poly-A sequence")

    opt, args = parser.parse_args()

    if not opt.tree_source:
        parser.error("Need one of the --tree-* options!")

    if not opt.seq0_source:
        parser.error("Need one of the --seq0-* options!")

    if len(args) != 2:
        parser.error("Need two positional arguments!")

    return opt, args


def get_child_depth_range(clade):
    level = [(0, clade)]

    mn = float('inf')
    mx = float('-inf')
    while level:
        new_level = []

        for d, parent in level:
            dc = d + parent.branch_length

            if parent.clades:
                for c in parent.clades:
                    new_level.append((dc, c))
            else:
                mn = min(mn, dc)
                mx = max(mx, dc)

        level = new_level

    return mn, mx


def main():
    opt, (rawfile, outalnfile) = get_options()

    raw = ccmpred.raw.parse(rawfile)
    seq0, id0 = opt.seq0_source(opt, raw)

    if raw.ncol != seq0.shape[1]:
        raise Exception("Mismatching number of columns: raw {0}, seq0 {1}".format(raw.ncol, seq0.shape[1]))

    tree = treecd.split_tree(opt.tree_source(opt, id0), id0)
    tree_bfs = [c for c in treecd.bfs_iterator(tree.clade)]

    depth_min, depth_max = get_child_depth_range(tree.clade)

    n_children = np.array([len(c.clades) for c in tree_bfs], dtype='uint64')
    branch_lengths = np.array([c.branch_length for c in tree_bfs], dtype=np.dtype('float64'))

    n_vertices = len(tree_bfs)
    n_leaves = len(tree.get_terminals())

    print("Got tree with {3} leaves, depth_min={0}, depth_max={1}, mutation_rate={2}".format(depth_min, depth_max, opt.mutation_rate, n_leaves))

    x = cd.structured_to_linear(raw.x_single, raw.x_pair)

    if opt.mutation_rate_sample:

        msa_sampled = np.empty((n_leaves, raw.ncol), dtype="uint8")

        def sample_neff(mutation_rate):

            treecd.cext.mutate_along_tree(msa_sampled, n_children, branch_lengths, x, n_vertices, seq0, mutation_rate)
            neff = np.sum(ccmpred.weighting.weights_simple(msa_sampled))

            print(mutation_rate, neff)
            sys.stdout.flush()

        print("x y")
        for _ in range(3):
            for mr in np.arange(0, 4.81, 0.4):
                sample_neff(mr)

    else:

        if opt.mutation_rate_neff:
            # Neff is dependant on the evolutionary distance D by the following
            # relationship:

            # Neff = 1 + (N - 1) / (1 + exp(-a*(D-t)^3 + b*(D-t)))

            # where N is the number of sampled leaves and a, b and t are
            # parameters fitted from a regression.
            a = -1.238
            b = 1.033
            t = 1.547

            # we substitute z = D - t and obtain the polynomial:
            # a*z^3 + b*z + log((N - 1) / (Neff - 1) - 1) = 0
            # solve using numpy.roots
            neffratio = np.log((n_leaves - 1) / (opt.mutation_rate_neff - 1) - 1)
            roots = np.roots([a, 0, b, neffratio])

            # only keep real solutions and resubstitute D = z + t
            mutation_rate = np.real(roots[np.abs(np.imag(roots)) < 1e-5])[0] + t

        else:
            mutation_rate = opt.mutation_rate

        msa_sampled = np.empty((n_leaves, raw.ncol), dtype="uint8")
        msa_sampled = treecd.cext.mutate_along_tree(msa_sampled, n_children, branch_lengths, x, n_vertices, seq0, mutation_rate)

    with open(outalnfile, "w") as f:
        ccmpred.io.alignment.write_msa_psicov(f, msa_sampled)


if __name__ == '__main__':
    main()
