# coding=utf-8
import ccmpred.objfun.cd.cext
import ccmpred.weighting
import ccmpred.trees
import ccmpred.sampling.cext
import numpy as np
import sys
from ccmpred.io.alignment import AMINO_ACIDS
from ccmpred.weighting.cext import count_ids, calculate_weights_simple
import ccmpred.counts
from ccmpred.pseudocounts import PseudoCounts

def gibbs_sample_sequences(x, msa_sampled, gibbs_steps):
    return ccmpred.objfun.cd.cext.gibbs_sample_sequences(msa_sampled, x, gibbs_steps)

def all_parents(tree):
    parents = {}
    for clade in tree.find_clades(order='level'):
        for child in clade:
            parents[child] = clade
    return parents

def mutate_along_phylogeny(tree, seq0, mutation_rate, x):

    ncol = len(seq0)

    #assign ancestor sequence to root
    tree.clade.seq = seq0

    #get all parents
    parents = all_parents(tree)

    #iterate breadth first over tree and mutate sequences
    for clade in tree.find_clades(order="level"):
        if clade.name != "root":
            #print("parent name: {0} parent seq: {1}".format( parents[clade], parents[clade].seq))
            nmut = int(clade.branch_length * mutation_rate * ncol)
            clade.seq =ccmpred.sampling.cext.mutate_sequence(parents[clade].seq, x, nmut, ncol)
            #print("clade name: {0} clade seq:  {1}".format(clade.name, clade.seq))
            #print("---")

    #get sequences of all leave nodes
    msa_sampled = np.array([clade.seq for clade in tree.get_terminals()])

    return msa_sampled

def generate_mcmc_sample(x, ncol, msa, size=10000, burn_in=500, sample_type="original"):

    print("Start sampling {0} sequences according to model starting with {1} sequences using burn-in={2}.".format(
        size, sample_type, burn_in))
    sys.stdout.flush()

    if msa is not None:
        N = msa.shape[0]
    else:
        N = 1000

    # sample at max 1000 sequences per iteration
    sample_size_per_it = np.min([N, 1000])

    ##repeat sampling until 10k sequences are obtained
    repeat = int(np.ceil(size / sample_size_per_it))
    samples = np.empty([repeat * sample_size_per_it, ncol], dtype="uint8")
    for i in range(repeat):

        if sample_type == "aln":

            #random selection of sequences from original MSA
            sample_seq_id = np.random.choice(ncol, sample_size_per_it, replace=False)
            msa_sampled = msa[sample_seq_id]

        elif sample_type == "random":

            #generate random sequences of length L
            msa_sampled = np.ascontiguousarray(
                [np.random.choice(20, ncol, replace=True) for _ in range(sample_size_per_it)], dtype="uint8")

        elif sample_type == "random-gapped":

            #generate random sequences of length L
            msa_sampled = np.ascontiguousarray(
                [np.random.choice(20, ncol, replace=True) for _ in range(sample_size_per_it)], dtype="uint8")

            #find gaps in randomly selected original sequences
            sample_seq_id = np.random.choice(N, sample_size_per_it, replace=False)
            msa_sampled_orig = msa[sample_seq_id]
            gap_indices = np.where(msa_sampled_orig == AMINO_ACIDS.index('-'))

            #assign gap states to random sequences
            msa_sampled[gap_indices] = AMINO_ACIDS.index('-')


        # burn in phase to move away from initial sequences
        msa_sampled = ccmpred.sampling.gibbs_sample_sequences(x, msa_sampled, gibbs_steps=burn_in)

        # add newly sampled sequences
        samples[i * sample_size_per_it: (i + 1) * sample_size_per_it] = msa_sampled
        print("sampled alignment has {0} sequences...".format((i + 1) * sample_size_per_it))
        sys.stdout.flush()

    #compute neff of sampled sequences
    neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

    print("Sampled alignment has Neff {0:.6g}".format(neff))

    return samples, neff

def sample_with_mutation_rate(tree, nseq, seq0, x, mutation_rate):
    """

    Parameters
    ----------
    tree: Tree object
    nseq: int
    seq0: 2dim array
    x:
    mutation_rate: float

    Returns
    -------

    """

    branch_lengths = tree.branch_lengths

    #how many substitutions per sequence will be performed
    nmut = [0]*(len(branch_lengths)-2)
    for i, bl in enumerate(branch_lengths[2:]):
        nmut[i] = bl * mutation_rate * seq0.shape[1]
    print("avg number of amino acid substitutions (parent -> child): {0}".format(
        np.round(np.mean(nmut), decimals=0)))


    # get the average number of amino acid substitution from root --> leave
    if tree.type == "binary" or tree.type == "star":
        number_splits = 1
        if tree.type == "binary":
            number_splits = np.log2(nseq)
        depth_per_clade = 1.0 /np.ceil(number_splits)
        print("avg number of amino acid substitutions (root -> leave): {0}".format(
            np.round(1 / depth_per_clade * np.mean(nmut), decimals=0)))


    # sample sequences according to tree topology
    msa_sampled = mutate_along_phylogeny(tree.tree, seq0[0], mutation_rate, x)

    # randomly choose nseq sequences from sampled msa
    if msa_sampled.shape[0] > nseq:
        msa_sampled = msa_sampled[sorted(np.random.choice(msa_sampled.shape[0], size=nseq, replace=False))]

    # compute neff of sampled sequences
    neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

    print("\nAlignment with {0} sequences was sampled with mutation rate {1} and has Neff {2:.6g}".format(
        nseq, mutation_rate, neff))

    return msa_sampled, neff

def sample_to_neff_increasingly(tree, nseq, target_neff, ncol, x, gibbs_steps):

    branch_lengths = tree.branch_lengths

    print("\nSample alignment of {0} protein sequences with target Neff~{1:.6g}...\n".format(
        nseq, target_neff))

    # keep increasing MR until we are within 1% of target neff
    mutation_rate = 1.0
    neff = -np.inf
    msa_sampled = np.empty((nseq, ncol), dtype="uint8")
    while np.abs(target_neff - neff) > 1e-2 * target_neff:

        # sample a new start sequence
        seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, gibbs_steps)
        print("Ancestor sequence (polyA --> {0} gibbs steps --> seq0) : {1}".format(gibbs_steps, "".join(
            [AMINO_ACIDS[c] for c in seq0[0]])))

        # how many substitutions per sequence will be performed
        nmut = [0] * (len(branch_lengths) - 2)
        for i, bl in enumerate(branch_lengths[2:]):
            nmut[i] = bl * mutation_rate * ncol
        print("avg number of amino acid substitutions (parent -> child): {0}".format(
            np.round(np.mean(nmut), decimals=0)))

        # get the average number of amino acid substitution from root --> leave
        if tree.type == "binary" or tree.type == "star":
            number_splits = 1
            if tree.type == "binary":
                number_splits = np.log2(nseq)
            depth_per_clade = 1.0 / np.ceil(number_splits)
            print("avg number of amino acid substitutions (root -> leave): {0}".format(
                np.round(1 / depth_per_clade * np.mean(nmut), decimals=0)))

        # sample sequences according to tree topology
        msa_sampled = mutate_along_phylogeny(tree.tree, seq0[0], mutation_rate, x)

        # randomly choose nseq sequences from sampled msa
        if msa_sampled.shape[0] > nseq:
            msa_sampled = msa_sampled[sorted(np.random.choice(msa_sampled.shape[0], size=nseq, replace=False))]

        # compute neff of sampled sequences
        neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)
        print("Alignment with {0} sequences was sampled with mutation rate {1:.3g} and has Neff {2:.5g} (ΔNeff [%] = {3:.5g})\n".format(
            nseq, mutation_rate, neff, (target_neff - neff)/target_neff*100))
        sys.stdout.flush()

        # inrease mutation rate
        if target_neff > neff:
            mutation_rate += np.random.random()

        # decrease mutation rate
        if target_neff < neff:
            mutation_rate -= np.random.random()

        #reset mutation rate if it becomes negative
        if mutation_rate < 0 or mutation_rate > 100:
            mutation_rate = 1

    return msa_sampled, neff
