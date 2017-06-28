import numpy as np
from collections import deque

import ccmpred.raw
import ccmpred.gaps
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd.cext
import ccmpred.parameter_handling
from ccmpred.weighting import SequenceWeights
from ccmpred.pseudocounts import PseudoCounts


class ContrastiveDivergence():

    def __init__(self, ccm, gibbs_steps=1, persistent=False, pll=False, sample_size=0):


        self.msa = ccm.msa
        self.nrow, self.ncol = self.msa.shape
        self.weights = ccm.weights
        self.neff = ccm.neff
        self.regularization = ccm.regularization

        self.pseudocount_type       = ccm.pseudocounts.pseudocount_type
        self.pseudocount_n_single   = ccm.pseudocounts.pseudocount_n_single
        self.pseudocount_n_pair     = ccm.pseudocounts.pseudocount_n_pair


        self.structured_to_linear = lambda x_single, x_pair: \
            ccmpred.parameter_handling.structured_to_linear(x_single,
                                                            x_pair,
                                                            nogapstate=True,
                                                            padding=False)
        self.linear_to_structured = lambda x: \
            ccmpred.parameter_handling.linear_to_structured(x,
                                                            self.ncol,
                                                            nogapstate=True,
                                                            add_gap_state=False,
                                                            padding=False)


        self.x_single = ccm.x_single
        self.x_pair = ccm.x_pair
        self.x = self.structured_to_linear(self.x_single, self.x_pair)



        self.nsingle = self.ncol * 20
        self.npair = self.ncol * self.ncol * 21 * 21
        self.nvar = self.nsingle + self.npair

        #perform x steps of sampling (all variables)
        self.gibbs_steps = np.max([gibbs_steps, 1])

        #do not initialise markov chain from input MSA at each iteration
        self.persistent = persistent

        #whether to sample only ONE variable per iteration
        self.pll = pll

        # get constant alignment counts - INCLUDING PSEUDO COUNTS
        # important for small alignments
        self.freqs_single, self.freqs_pair = ccm.freqs
        self.msa_counts_single = self.freqs_single * self.neff
        self.msa_counts_pair = self.freqs_pair * self.neff

        # reset gap counts
        self.msa_counts_single[:, 20] = 0
        self.msa_counts_pair[:, :, :, 20] = 0
        self.msa_counts_pair[:, :, 20, :] = 0

        # non_gapped counts
        self.Ni = self.msa_counts_single.sum(1)
        self.Nij = self.msa_counts_pair.sum(3).sum(2)


        self.averaging=False
        averaging_size=10
        self.sample_counts_single_prev = deque([], maxlen=averaging_size)
        self.sample_counts_pair_prev = deque([], maxlen=averaging_size)


        if self.persistent:
            self.persistent_msa = self.msa.copy()

        self.sample_size = sample_size
        self.nr_seq_sample = self.nrow
        if (sample_size > 0) and (sample_size * self.ncol) < self.nrow:
             self.nr_seq_sample = sample_size * self.ncol


    def __repr__(self):

        str = "{0}{1}contrastive divergence: ".format(
            "PERSISTENT " if (self.persistent) else "",
            "PLL " if (self.pll) else ""
        )

        str += "#sampled sequences={0} ({1} x N and {2} x L)  Gibbs steps={3} ".format(
            (self.nr_seq_sample),
            np.round(self.nr_seq_sample / float(self.nrow), decimals=3),
            np.round(self.nr_seq_sample / float(self.ncol), decimals=3),
            self.gibbs_steps
        )

        return str

    def init_sample_alignment(self):

        self.sample_seq_id = np.random.choice(self.nrow, self.nr_seq_sample, replace=False)
        msa_sampled = self.msa[self.sample_seq_id]

        #continue sampling
        if self.persistent:
            msa_sampled = self.persistent_msa[self.sample_seq_id]

        return msa_sampled, self.weights[self.sample_seq_id]

    def gibbs_sample_sequences(self, x, gibbs_steps):
        return ccmpred.objfun.cd.cext.gibbs_sample_sequences(self.msa_sampled,  x, gibbs_steps)

    def gibbs_sample_sequences_nogaps(self, x, gibbs_steps):
        return ccmpred.objfun.cd.cext.gibbs_sample_sequences_nogaps(self.msa_sampled,  x, gibbs_steps)

    def sample_position_in_sequences(self, x):
        return ccmpred.objfun.cd.cext.sample_position_in_sequences(self.msa_sampled, x)

    def con_prob(self, x_single, x_pair, seq, pos):
        '''
        log P(x_i = a| v, w, (x1...L\{xi})) prop_to v_i(a) + w_ij(a, x_j)

        :param x_single:    single emissions
        :param x_pair:      pair emissions
        :param seq:         sequence for sampling
        :param pos:         position for sampling
        :return:
        '''


        conditional_prob = [0] * 20

        conditional_prob += x_single[pos, :20]

        for j in range(self.ncol):
            conditional_prob += x_pair[pos, j, :20, seq[j]]
        conditional_prob -= x_pair[pos, pos, :20, seq[pos]]

        #normalize exponentials
        max_log_prob = np.max(conditional_prob)
        conditional_prob = np.exp(conditional_prob - max_log_prob)
        conditional_prob /= np.sum(conditional_prob)

        return conditional_prob

    def gibbs_sampling_slow(self, msa, x, k_steps):
        '''
        Python implementation of Gibbs Sampling just for debugging as
        it is much slower than the C version

        :param msa:
        :param x:
        :param k_steps:
        :return:
        '''

        x_single, x_pair = self.linear_to_structured(x, self.ncol)

        for n in range(msa.shape[0]):

            #get all non gapped positions in sequence
            positions = np.where(msa[n] != 20)[0]

            #permutation vector
            #np.random.shuffle(positions)

            #choose random positions with replacement
            positions = np.random.choice(positions, size=k_steps * len(positions), replace=True)

            #sample every non-gapped position of sequence
            for pos in positions:
                conditonal_prob = self.con_prob(x_single, x_pair, msa[n], pos)
                msa[n, pos] = np.random.choice(range(20), p=conditonal_prob) #gap==20

        return msa

    def get_pairwise_freq_from_sample(self, x, gibbs_steps):

        #so that initial sample alignment is NOT a minibatch
        self.nr_seq_minibatch = 0

        #initialise sequences for sampling from input MSA
        self.msa_sampled, self.msa_sampled_weights = self.init_sample_alignment()

        #do gibbs sampling
        self.msa_sampled = self.gibbs_sample_sequences(x, gibbs_steps)

        #counts from sample
        sample_counts_pair = ccmpred.counts.pair_counts(self.msa_sampled, self.msa_sampled_weights)

        # reset gap counts
        sample_counts_pair[:, :, :, 20] = 0
        sample_counts_pair[:, :, 20, :] = 0

        Nij_sampled = sample_counts_pair.sum(3).sum(2) + 1e-10
        sampled_freq_pair       = sample_counts_pair / Nij_sampled[:, :, np.newaxis, np.newaxis]

        return sampled_freq_pair

    def finalize(self, x):
        return ccmpred.parameter_handling.linear_to_structured(x,
                                                            self.ncol,
                                                            clip=False,
                                                            nogapstate=True,
                                                            add_gap_state=True,
                                                            padding=False)

    def evaluate(self, x):


        #setup sequences for sampling
        self.msa_sampled, self.msa_sampled_weights = self.init_sample_alignment()

        #Gibbs Sampling of sequences (each position of each sequence will be sampled this often: self.gibbs_steps)
        if self.pll:
            self.msa_sampled = self.sample_position_in_sequences(x)
        else:
            self.msa_sampled = self.gibbs_sample_sequences(x, self.gibbs_steps)

        #save the markov chain
        if  self.persistent:
            self.persistent_msa[self.sample_seq_id] = self.msa_sampled


        # compute amino acid frequencies from sample
        # add pseudocounts for stability
        pseudocounts = PseudoCounts(self.msa_sampled, self.msa_sampled_weights)
        sampled_freq_single, sampled_freq_pair = pseudocounts.calculate_frequencies(
                self.pseudocount_type,
                self.pseudocount_n_single,
                self.pseudocount_n_pair,
                remove_gaps=False)

        sampled_freq_single = pseudocounts.degap(sampled_freq_single, True)
        sampled_freq_pair   = pseudocounts.degap(sampled_freq_pair, True)

        #compute counts and scale them accordingly to size of input MSA
        sample_counts_single    = sampled_freq_single * self.Ni[:, np.newaxis]
        sample_counts_pair      = sampled_freq_pair * self.Nij[:, :, np.newaxis, np.newaxis]


        #actually compute the gradients
        g_single = sample_counts_single - self.msa_counts_single
        g_pair = sample_counts_pair - self.msa_counts_pair


        #sanity check
        if(np.abs(np.sum(sample_counts_single[1,:20]) - np.sum(self.msa_counts_single[1,:20])) > 1e-5):
            print("Warning: sample aa counts ({0}) do not equal input msa aa counts ({1})!".format(np.sum(sample_counts_single[1,:20]), np.sum(self.msa_counts_single[1,:20])))


        # set gradients for gap states to 0
        g_single[:, 20] = 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        for i in range(self.ncol):
            g_pair[i, i, :, :] = 0


        #compute regularization
        x_single, x_pair = self.linear_to_structured(x)                     #x_single has dim L x 20
        _, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair) #g_single_reg has dim L x 20


        #gradient for x_single only L x 20
        g = self.structured_to_linear(g_single[:, :20], g_pair)
        g_reg = self.structured_to_linear(g_single_reg[:, :20], g_pair_reg)

        return -1, g, g_reg

    def get_parameters(self):
        parameters = {}
        parameters['gibbs_steps'] = self.gibbs_steps
        parameters['persistent'] = self.persistent
        parameters['pll']       = self.pll
        parameters['averaging'] = self.averaging
        parameters['sample_size'] = self.sample_size
        parameters['nr_seq_sample'] = self.nr_seq_sample


        return parameters