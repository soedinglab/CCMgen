import numpy as np

import ccmpred.raw
import ccmpred.gaps
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd.cext

import ccmpred.pseudocounts
import ccmpred.weighting

class ContrastiveDivergence(ccmpred.objfun.ObjectiveFunction):

    def __init__(self, msa, freqs, weights, regularization, n_samples,  gibbs_steps, persistent):
        super(ContrastiveDivergence, self).__init__()


        self.msa = msa
        self.weights = weights
        self.regularization = regularization

        self.nrow, self.ncol = msa.shape
        self.nsingle = self.ncol * 20
        self.nvar = self.nsingle + self.ncol * self.ncol * 21 * 21
        self.n_samples = n_samples
        self.gibbs_steps = gibbs_steps
        self.persistent = persistent

        # get constant alignment counts
        self.neff = np.sum(weights)
        freqs_single, freqs_pair = freqs
        self.msa_counts_single = freqs_single * self.neff
        self.msa_counts_pair = freqs_pair * self.neff
        # self.msa_counts_single, self.msa_counts_pair = ccmpred.counts.both_counts(msa, None)

        # reset gap counts
        self.msa_counts_single[:, 20] = 0
        self.msa_counts_pair[:, :, :, 20] = 0
        self.msa_counts_pair[:, :, 20, :] = 0

        #non_gapped counts
        self.Ni = self.msa_counts_single.sum(1)
        self.Nij = self.msa_counts_pair.sum(3).sum(2)

        # init sample alignment
        self.init_sample_alignment()

        self.linear_to_structured = lambda x: linear_to_structured(x, self.ncol)
        self.structured_to_linear = structured_to_linear

    def init_sample_alignment(self):

        if self.n_samples == 0:
            self.msa_sampled = self.msa.copy()
            self.weights_msa_sampled = self.weights
        else:

            #pick random sequences from original alignment
            random_sequence_ids =  np.random.randint(0, self.nrow, self.n_samples)

            self.msa_sampled = self.msa[random_sequence_ids]
            self.weights_msa_sampled = self.weights[random_sequence_ids]


    @classmethod
    def init_from_raw(cls, msa, freqs, weights, raw, regularization, gibbs_steps=1, persistent=False, n_samples=0):

        #n_samples = msa.shape[0]

        res = cls(msa, freqs, weights, regularization, n_samples, gibbs_steps, persistent)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        # raw.x_single is of shape L x 20
        # raw.x_pair   is of shape L x L x 21 x 21
        x = structured_to_linear(raw.x_single, raw.x_pair)
        return x, res

    def finalize(self, x, meta):
        x_single, x_pair = linear_to_structured(x, self.ncol, clip=True)

        return ccmpred.raw.CCMRaw(self.ncol, x_single, x_pair, meta)

    def gibbs_sample_sequences(self, x):

        if self.persistent:
            return ccmpred.objfun.cd.cext.gibbs_sample_sequences(self.msa_sampled,  x, self.gibbs_steps)
        else:
            #for CD start from the input data
            self.init_sample_alignment()
            return ccmpred.objfun.cd.cext.gibbs_sample_sequences(self.msa_sampled,  x, self.gibbs_steps)

    def sample_sequences(self, x):
        #for PERSISTENT CD continue the markov chain
        return ccmpred.objfun.cd.cext.sample_sequences(self.msa_sampled, x)

    def evaluate(self, x):

        self.msa_sampled = self.gibbs_sample_sequences(x)

        #careful with the weights: sum(sample_counts) should equal sum(msa_counts) !
        sample_counts_single, sample_counts_pair = ccmpred.counts.both_counts(self.msa_sampled, self.weights_msa_sampled)

        # reset gap counts for sampled msa
        sample_counts_single[:, 20] = 0
        sample_counts_pair[:, :, :, 20] = 0
        sample_counts_pair[:, :, 20, :] = 0

        #non_gapped counts per position / pair
        Ni_sampled = sample_counts_single.sum(1)
        Nij_sampled = sample_counts_pair.sum(3).sum(2)


        #normalize counts: divide single counts of sampled msa by sum of non_gapped REAL counts
        normalization_single_counts = self.Ni / (Ni_sampled + 1e-10)
        normalization_pair_counts   = self.Nij / (Nij_sampled + 1e-10)

        sample_counts_single *= normalization_single_counts[:, np.newaxis]
        sample_counts_pair   *= normalization_pair_counts[:, :,  np.newaxis, np.newaxis]


        g_single = sample_counts_single - self.msa_counts_single
        g_pair = sample_counts_pair - self.msa_counts_pair

        #sanity check
        # if(np.abs(np.sum(sample_counts_single[0,:20]) - np.sum(self.msa_counts_single[0,:20])) > 1e-10):
        #     print("Warning: sample aa counts ({0}) do not equal input msa aa counts ({1})!".format(np.sum(sample_counts_single[0,:20]), np.sum(self.msa_counts_single[0,:20])))

        x_single, x_pair = linear_to_structured(x, self.ncol)
        _, g_single_reg, g_pair_reg = self.regularization(x_single, x_pair)

        g_single[:, :20] += g_single_reg   #g_single_reg is of dim Lx20 as x_single is of dim Lx20
        g_pair += g_pair_reg

        #no need to set g_single[:, 20]  to 0  as we return only g_single[:, :20]
        # set gradients for gap states to 0
        g_pair[:, :, :, 20] = 0
        g_pair[:, :, 20, :] = 0

        for i in range(self.ncol):
            g_pair[i, i, :, :] = 0

        #gradient for x_single only L x 20
        g = structured_to_linear(g_single[:, :20], g_pair)
        return -1, g

    def __repr__(self):
        return "{0} contrastive divergence using {1} Gibbs sampling steps for sampling {2} sequences ".format(
            "PERSISTENT" if (self.persistent) else " ",
            self.gibbs_steps,
            self.n_samples if self.n_samples > 0 else self.nrow
        )

def linear_to_structured(x, ncol, clip=False):
    """Convert linear vector of variables into multidimensional arrays.

    in linear memory, memory order is v[j, a] and w[i, a, j, b] (dimensions Lx20 and Lx21xLx21)
    output will have  memory order of v[j, a] and w[i, j, a, b] (dimensions Lx20 and LxLx21x21)
    """

    nsingle = ncol * 20

    x_single = x[:nsingle].reshape((ncol, 20))
    x_pair = np.transpose(x[nsingle:].reshape((ncol, 21, ncol, 21)), (0, 2, 1, 3))

    if clip:
        x_pair = x_pair[:, :, :21, :21]

    return x_single, x_pair


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
