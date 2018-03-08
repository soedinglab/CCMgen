import ccmpred.objfun.cd.cext
import ccmpred.weighting
import ccmpred.trees
import ccmpred.sampling.cext
import numpy as np
import sys
from ccmpred.io.alignment import AMINO_ACIDS

def gibbs_sample_sequences(x, msa_sampled, gibbs_steps):
    return ccmpred.objfun.cd.cext.gibbs_sample_sequences(msa_sampled, x, gibbs_steps)

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

def sample_with_mutation_rate(tree, ncol, x, gibbs_steps, mutation_rate):

    #sample a new start sequence
    seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, gibbs_steps)
    print("Ancestor sequence: {0}".format(seq0[0]))

    branch_lengths = tree.branch_lengths
    n_children = tree.n_children
    n_vertices = tree.n_vertices
    n_leaves = tree.n_leaves
    nseq = tree.nseq

    #sample sequences according to tree topology
    msa_sampled = np.empty((n_leaves, ncol), dtype="uint8")
    msa_sampled = ccmpred.sampling.cext.mutate_along_tree(
        msa_sampled, n_children, branch_lengths, x, n_vertices, seq0, mutation_rate)
    msa_sampled = msa_sampled[:nseq]


    #compute neff of sampled sequences
    neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

    print("Sampled alignment with mutation rate {0} has Neff {1:.6g}".format(
            mutation_rate, neff))

    return msa_sampled, neff

def sample_to_neff(tree, target_neff, ncol, x, gibbs_steps):

    branch_lengths = tree.branch_lengths
    n_children = tree.n_children
    n_vertices = tree.n_vertices
    n_leaves = tree.n_leaves
    nseq = tree.nseq

    print("\nSample sequences to generate alignment with target Neff={0:.6g} similar to original MSA...".format(target_neff))

    mr_min = 0.0
    mr_max = 50.0
    mutation_rate = (mr_min + mr_max) / 2

    # keep trying until we are within 1% of target neff
    neff = -np.inf
    msa_sampled = np.empty((n_leaves, ncol), dtype="uint8")
    while np.abs(neff - target_neff) > 1e-2 * target_neff and mr_min < (0.999 * mr_max):

        #sample a new start sequence
        seq0 = ccmpred.trees.get_seq0_mrf(x, ncol, gibbs_steps)
        print("Ancestor sequence: {0}".format("".join([AMINO_ACIDS[c] for c in seq0[0]])))

        #sample sequences according to tree topology
        msa_sampled = np.empty((n_leaves, ncol), dtype="uint8")
        msa_sampled = ccmpred.sampling.cext.mutate_along_tree(
            msa_sampled, n_children, branch_lengths, x, n_vertices, seq0, mutation_rate)
        msa_sampled = msa_sampled[:nseq]

        #compute neff of sampled sequences
        neff = ccmpred.weighting.get_HHsuite_neff(msa_sampled)

        print("Sampled alignment with mutation rate {0:.3g} (mr_min={1:.3g}, mr_max={2:.3g}) has Neff {3:.6g}".format(
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