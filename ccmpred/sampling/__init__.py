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


def generate_mcmc_sample(x, msa, size=10000, burn_in=500, sample_type="original"):

    print("Start sampling {0} sequences according to model starting with {1} sequences using burn-in={2}.".format(
        size, sample_type, burn_in))
    sys.stdout.flush()

    N = msa.shape[0]
    L = msa.shape[1]

    # sample at max 1000 sequences per iteration
    sample_size_per_it = np.min([N, 1000])

    ##repeat sampling until 10k sequences are obtained
    repeat = int(np.ceil(size / sample_size_per_it))
    samples = np.empty([repeat * sample_size_per_it, L], dtype="uint8")
    for i in range(repeat):

        if sample_type == "original":

            #random selection of sequences from original MSA
            sample_seq_id = np.random.choice(L, sample_size_per_it, replace=False)
            msa_sampled = msa[sample_seq_id]

        elif sample_type == "random":

            #generate random sequences of length L
            msa_sampled = np.ascontiguousarray(
                [np.random.choice(20, L, replace=True) for _ in range(sample_size_per_it)], dtype="uint8")

        elif sample_type == "random-gapped":

            #generate random sequences of length L
            msa_sampled = np.ascontiguousarray(
                [np.random.choice(20, L, replace=True) for _ in range(sample_size_per_it)], dtype="uint8")

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

def sample_with_mutation_rate(tree, seq0, x, mutation_rate):

    branch_lengths = tree.branch_lengths
    nseq = tree.nseq


    #how many substitutions per sequence will be performed
    nmut = [0]*(len(branch_lengths)-2)
    for i, bl in enumerate(branch_lengths[2:]):
        nmut[i] = bl * mutation_rate * seq0.shape[1]
    print("avg number of amino acid substitutions (parent -> child): {0}".format(
        np.round(np.mean(nmut), decimals=0)))


    #get the average number of amino acid substitution from root --> leave
    if tree.type == "binary" or tree.type == "star":
        number_splits = 1
        if tree.type == "binary":
            number_splits = np.log2(nseq)
        depth_per_clade = 1.0 /np.ceil(number_splits)
        print("avg number of amino acid substitutions (root -> leave): {0}".format(
            np.round(1 / depth_per_clade * np.mean(nmut), decimals=0)))


    #sample sequences according to tree topology
    msa_sampled = mutate_along_phylogeny(tree.tree, seq0[0], mutation_rate, x)

    #usually the case for binary trees
    if msa_sampled.shape[0] > nseq:
        first_half = msa_sampled[:int(np.floor(nseq/2))]
        second_half = msa_sampled[-int(np.ceil(nseq/2)):]
        msa_sampled = np.array(list(first_half) + list(second_half))

    #compute neff of sampled sequences
    neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

    print("\nAlignment was sampled with mutation rate {0} and has Neff {1:.6g}".format(
            mutation_rate, neff))

    return msa_sampled, neff

def sample_to_pair_correlation(tree, target_neff, ncol, x, gibbs_steps, single_freq_observed, pair_freq_observed):
    branch_lengths = tree.branch_lengths
    nseq = tree.nseq

    indices_upper_i, indices_upper_j = np.triu_indices(ncol, k=1)
    single_freq_observed_flat = single_freq_observed.flatten().tolist()
    pair_freq_observed_flat = pair_freq_observed[indices_upper_i, indices_upper_j, :, :].flatten().tolist()
    cov_observed = [pair_freq_observed[i, j, a, b] - (single_freq_observed[i, a] * single_freq_observed[j, b])
                    for i in range(ncol - 1) for j in range(i + 1, ncol) for a in range(20) for b in range(20)]

    print("\nSample sequences to generate alignment with Pearson correlation coefficient > 0.9 "
             "between observed and sampled pair frequencies...\n")

    # sample a new start sequence
    seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, gibbs_steps)
    print("Ancestor sequence (polyA --> {0} gibbs steps --> seq0) : {1}".format(gibbs_steps, "".join(
        [AMINO_ACIDS[c] for c in seq0[0]])))

    mutation_rate = 1.0
    pearson_corr_pair = 0.0
    msa_sampled = np.empty((nseq, ncol), dtype="uint8")
    neff = 0
    while pearson_corr_pair < 0.9:

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

        # usually the case for binary trees
        if msa_sampled.shape[0] > nseq:
            first_half = msa_sampled[:int(np.floor(nseq / 2))]
            second_half = msa_sampled[-int(np.ceil(nseq / 2)):]
            msa_sampled = np.array(list(first_half) + list(second_half))

        # compute neff of sampled sequences
        neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)
        print("Alignment was sampled with mutation rate {0:.3g} and has Neff {1:.5g}\n".format(mutation_rate, neff))

        # compute amino acid frequencies
        weights = calculate_weights_simple(msa_sampled, cutoff=0.8, ignore_gaps=False)
        pseudocounts = PseudoCounts(msa_sampled, weights)
        pseudocounts.calculate_frequencies(
            "uniform_pseudocounts",
            1,
            1,
            remove_gaps=False
        )
        single_freqs_sampled = pseudocounts.degap(pseudocounts.freqs[0], False)
        single_freqs_sampled_flat = single_freqs_sampled.flatten().tolist()

        pair_freqs_sampled = pseudocounts.degap(pseudocounts.freqs[1], False)
        pair_freq_sampled_flat = pair_freqs_sampled[indices_upper_i, indices_upper_j, :, :].flatten().tolist()

        cov_sampled = [pair_freqs_sampled[i, j, a, b] - (single_freqs_sampled[i, a] * single_freqs_sampled[j, b])
                       for i in range(ncol - 1) for j in range(i + 1, ncol) for a in range(20) for b in range(20)]

        pearson_corr_single = np.corrcoef(single_freq_observed_flat, single_freqs_sampled_flat)[0, 1]
        pearson_corr_pair = np.corrcoef(pair_freq_observed_flat, pair_freq_sampled_flat)[0, 1]
        pearson_corr_cov = np.corrcoef(cov_observed, cov_sampled)[0, 1]

        print("Neff difference is {0:.5g}%, correlation with observed single freq: {1:.5g} "
              "and pair freq: {2:.5g} and covariances: {3:.5g}\n".format((target_neff - neff) / target_neff * 100,
                                                                         pearson_corr_single, pearson_corr_pair,
                                                                         pearson_corr_cov))
        sys.stdout.flush()

        #increase mutation rate
        mutation_rate += 0.3

        # prevent mutation rate from becoming too high
        if mutation_rate > 10:
            # sample a new start sequence and begin anew
            seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, gibbs_steps)
            print("Ancestor sequence (polyA --> {0} gibbs steps --> seq0) : {1}".format(gibbs_steps, "".join(
                [AMINO_ACIDS[c] for c in seq0[0]])))
            mutation_rate = 1

    return msa_sampled, neff

def sample_to_neff_increasingly(tree, target_neff, ncol, x, gibbs_steps):

    branch_lengths = tree.branch_lengths
    nseq = tree.nseq


    print("\nSample sequences to generate alignment with target Neff~{0:.6g}...\n".format(
        target_neff))

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

        # usually the case for binary trees
        if msa_sampled.shape[0] > nseq:
            first_half = msa_sampled[:int(np.floor(nseq / 2))]
            second_half = msa_sampled[-int(np.ceil(nseq / 2)):]
            msa_sampled = np.array(list(first_half) + list(second_half))

        # compute neff of sampled sequences
        neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)
        print("Alignment was sampled with mutation rate {0:.3g} and has Neff {1:.5g} (ΔNeff [%] = {2:.5g})\n".format(
            mutation_rate, neff, (target_neff - neff)/target_neff*100))
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

def sample_to_neff(tree, target_neff, ncol, x, gibbs_steps):

    branch_lengths = tree.branch_lengths
    nseq = tree.nseq

    print("\nSample sequences to generate alignment with target Neff~{0:.6g}...\n".format(
        target_neff))

    mr_min = 0.0
    mr_max = 20.0
    mutation_rate = (mr_min + mr_max) / 2

    # keep trying until we are within 1% of target neff
    neff = -np.inf
    msa_sampled = np.empty((nseq, ncol), dtype="uint8")
    while np.abs(neff - target_neff) > 1e-2 * target_neff and mr_min < (0.999 * mr_max):

        #sample a new start sequence
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


        #sample sequences according to tree topology
        msa_sampled = mutate_along_phylogeny(tree.tree, seq0[0], mutation_rate, x)

        # usually the case for binary trees
        if msa_sampled.shape[0] > nseq:
            first_half = msa_sampled[:int(np.floor(nseq / 2))]
            second_half = msa_sampled[-int(np.ceil(nseq / 2)):]
            msa_sampled = np.array(list(first_half) + list(second_half))

        #compute neff of sampled sequences
        neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

        print("Alignment was sampled with mutation rate {0:.3g} (mr_min={1:.3g}, mr_max={2:.3g}) and has Neff {3:.6g}\n".format(
            mutation_rate, mr_min, mr_max, neff))
        sys.stdout.flush()

        if neff < target_neff:
            # neff was too small, increase mutation rate
            mr_min = mutation_rate

        elif neff > target_neff:
            # neff was too big, decrease mutation rate
            mr_max = mutation_rate

        # continue binary search
        mutation_rate = (mr_min + mr_max) / 2

    return msa_sampled, neff