import numpy as np
from collections import deque

import ccmpred.raw
import ccmpred.gaps
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd.cext

import ccmpred.pseudocounts
import ccmpred.weighting


class ContrastiveDivergence():

    def __init__(self, msa, freqs, weights, raw, regularization, gibbs_steps=1, persistent=False, min_nseq_factorL=1, pll=False, average_sample_counts=False):


        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        self.x0 = self.structured_to_linear(raw.x_single[:, :20], raw.x_pair)

        self.msa = msa
        self.weights = weights
        self.neff = np.sum(weights)
        self.regularization = regularization

        self.nrow, self.ncol = msa.shape
        self.nsingle = self.ncol * 20
        self.nvar = self.nsingle + self.ncol * self.ncol * 21 * 21


        #perform x steps of sampling (all variables)
        self.gibbs_steps = np.max([gibbs_steps, 1])

        #do not initialise markov chain from input MSA at each iteration
        self.persistent = persistent

        #whether to sample only ONE variable per iteration
        self.pll = pll

        #whether to compute average counts from last X sampled MSA's
        self.average_sample_counts=average_sample_counts
        self.collection_sample_counts_single = deque([])
        self.collection_sample_counts_pair = deque([])

        # get constant alignment counts
        self.freqs_single, self.freqs_pair = freqs
        self.msa_counts_single = self.freqs_single * self.neff
        self.msa_counts_pair = self.freqs_pair * self.neff

        #do not use pseudo counts!
        #self.msa_counts_single, self.msa_counts_pair = ccmpred.counts.both_counts(msa, self.weights)

        # reset gap counts
        self.msa_counts_single[:, 20] = 0
        self.msa_counts_pair[:, :, :, 20] = 0
        self.msa_counts_pair[:, :, 20, :] = 0

        #non_gapped counts
        self.Ni = self.msa_counts_single.sum(1)
        self.Nij = self.msa_counts_pair.sum(3).sum(2)


        #number of sequences used for sampling: multiples of MSA and at least 1xMSA
        self.min_nseq_factorL = np.max([min_nseq_factorL, 1])
        self.n_samples_msa = 1


        # init sample alignment as input MSA
        self.msa_sampled = self.init_sample_alignment(self.min_nseq_factorL)
        self.msa_sampled_weights = ccmpred.weighting.weights_simple(self.msa_sampled)

    def init_sample_alignment(self, min_nseq_factorL):


        # nr of sequences = min_nseq_factorL * L
        self.min_nseq_factorL = np.max([min_nseq_factorL, 1])
        n_sequence_min_nseq_factorL =  self.min_nseq_factorL * self.ncol

        #Use multiples of input MSA: at least 1xMSA
        self.n_samples_msa = int(np.ceil( n_sequence_min_nseq_factorL / float(self.nrow)))

        if self.average_sample_counts:
            return self.msa.copy()
        else:
            seq_id = range(self.nrow) * self.n_samples_msa
            msa_sampled = self.msa[seq_id]

            return msa_sampled.copy()





    def finalize(self, x, meta):
        x_single, x_pair = self.linear_to_structured(x, self.ncol, add_gap_state=True)

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, meta)

    def gibbs_sample_sequences(self, x):
        return ccmpred.objfun.cd.cext.gibbs_sample_sequences(self.msa_sampled,  x, self.gibbs_steps)

    def gibbs_sample_sequences_nogaps(self, x):
        return ccmpred.objfun.cd.cext.gibbs_sample_sequences_nogaps(self.msa_sampled,  x, self.gibbs_steps)

    def sample_position_in_sequences(self, x):
        return ccmpred.objfun.cd.cext.sample_position_in_sequences(self.msa_sampled, x)


    def compute_sample_count_averages(self, sample_counts_single, sample_counts_pair):
        # store sample counts
        self.collection_sample_counts_single.append(sample_counts_single)
        self.collection_sample_counts_pair.append(sample_counts_pair)


        n_samples_msa = float(self.n_sequences / self.nrow)

        # keep only N_SAMPLES_MSA latest counts
        if len(self.collection_sample_counts_single) > n_samples_msa:

            self.collection_sample_counts_single.popleft()
            self.collection_sample_counts_pair.popleft()

        # Sum counts over all stored sample counts
        overall_sampled_counts_single = np.sum(self.collection_sample_counts_single, axis=0)
        overall_sampled_counts_pair = np.sum(self.collection_sample_counts_pair, axis=0)

        return overall_sampled_counts_single, overall_sampled_counts_pair

    def evaluate(self, x):


        #reset the msa for sampling in caes of CD
        if not self.persistent:
            self.msa_sampled = self.init_sample_alignment(self.min_nseq_factorL)

        if self.pll:
            self.msa_sampled = self.sample_position_in_sequences(x)
        else:
            #Gibbs Sampling of sequences (each position of each sequence will be sampled this often: self.gibbs_steps)
            self.msa_sampled = self.gibbs_sample_sequences(x)


        #careful with the weights: sum(sample_counts) should equal sum(msa_counts) !
        sample_counts_single, sample_counts_pair = ccmpred.counts.both_counts(self.msa_sampled, self.msa_sampled_weights)


        #reset gap counts for sampled msa
        sample_counts_single[:, 20] = 0
        sample_counts_pair[:, :, :, 20] = 0
        sample_counts_pair[:, :, 20, :] = 0


        if self.average_sample_counts:
            sample_counts_single, sample_counts_pair = self.compute_sample_count_averages(sample_counts_single, sample_counts_pair)


        # number of non_gapped counts per position(pair)
        Ni_sampled  = sample_counts_single.sum(1) + 1e-10
        Nij_sampled = sample_counts_pair.sum(3).sum(2) + 1e-10

        # normalize counts according to input msa counts
        sampled_freq_single     = sample_counts_single / Ni_sampled[:, np.newaxis]
        sampled_freq_pair       = sample_counts_pair / Nij_sampled[:, :, np.newaxis, np.newaxis]
        sample_counts_single    = sampled_freq_single * self.Ni[:, np.newaxis]
        sample_counts_pair      = sampled_freq_pair * self.Nij[:, :, np.newaxis, np.newaxis]

        #actually compute the gradients
        g_single = sample_counts_single - self.msa_counts_single
        g_pair = sample_counts_pair - self.msa_counts_pair


        #sanity check
        if(np.abs(np.sum(sample_counts_single[0,:20]) - np.sum(self.msa_counts_single[0,:20])) > 1e-5):
            print("Warning: sample aa counts ({0}) do not equal input msa aa counts ({1})!".format(np.sum(sample_counts_single[0,:20]), np.sum(self.msa_counts_single[0,:20])))


        # set gradients for gap states to 0
        g_single[:, 20] = 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        for i in range(self.ncol):
            g_pair[i, i, :, :] = 0


        #gradient for x_single only L x 20
        g = self.structured_to_linear(g_single[:, :20], g_pair)

        #add regularization
        x_single, x_pair = self.linear_to_structured(x, self.ncol)
        _, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair)

        g_reg = self.structured_to_linear(g_single_reg[:, :20], g_pair_reg)
        g += g_reg

        return -1, g

    def __repr__(self):

        str = "{0}{1}contrastive divergence".format(
            "PERSISTENT " if (self.persistent) else "",
            "PLL " if (self.pll) else ""
        )

        str += "\nSampling {0} sequences ({1} x N and {2} x L)  with {3} Gibbs steps.".format(
            (self.n_samples_msa * self.nrow), np.round(self.n_samples_msa, decimals=3),  np.round((self.n_samples_msa * self.nrow) / float(self.ncol), decimals=3), self.gibbs_steps
        )

        return str

    @staticmethod
    def linear_to_structured(x, ncol, add_gap_state=False):
        """Convert linear vector of variables into multidimensional arrays.

        in linear memory, memory order is v[j, a] and w[i, a, j, b] (dimensions Lx20 and Lx21xLx21)
        output will have  memory order of v[j, a] and w[i, j, a, b] (dimensions Lx20 and LxLx21x21)
        """
        nsingle = ncol * 20

        x_single = x[:nsingle].reshape((ncol, 20))
        x_pair = np.transpose(x[nsingle:].reshape((ncol, 21, ncol, 21)), (0, 2, 1, 3))

        if add_gap_state:
            temp = np.zeros((ncol, 21))
            temp[:,:20] = x_single
            x_single = temp

        return x_single, x_pair

    @staticmethod
    def structured_to_linear(x_single, x_pair):
        """Convert structured variables into linear array

        with input arrays of memory order v[j, a] and w[i, j, a, b] (dimensions Lx20 and LxLx21x21)
        output will have  memory order of v[j, a] and w[i, a, j, b] (dimensions Lx20 and Lx21xLx21)
        """

        ncol = x_single.shape[0]
        nsingle = ncol * 20
        nvar = nsingle + ncol * ncol * 21 * 21


        out_x_pair = np.zeros((ncol, 21, ncol, 21), dtype='float64')
        out_x_pair[:, :21, :, :21] = np.transpose(x_pair[:, :, :21, :21], (0, 2, 1, 3))

        x = np.zeros((nvar, ), dtype='float64')

        x[:nsingle] = x_single.reshape(-1)
        x[nsingle:] = out_x_pair.reshape(-1)

        return x
