__version__ = '1.0.0'

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV-"

import datetime
import os
import sys
import numpy as np
import io as io
import gaps
import counts
from weighting import SequenceWeights
from pseudocounts import PseudoCounts
import local_methods
import centering
import initialise_potentials
import raw
import model_probabilities
import parameter_handling
import sanity_check
import regularization

class CCMpred():
    """
    CCMpred is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods. From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model with 21 states for amino acids and gaps. The L2 norms of the pairwise coupling potentials will be written to the output matfile.
    """

    def __init__(self, alignment_file, mat_file):

        self.alignment_file = alignment_file
        self.mat_file       = mat_file
        self.protein        = os.path.basename(self.alignment_file).split(".")[0]

        self.msa            = None
        self.gapped_positions = None
        self.max_gap_ratio = None
        self.N              = None
        self.L              = None
        self.neff           = None
        self.diversity      = None

        #sequence weighting
        self.weighting = None
        self.weighting_type = None
        self.weights   = None


        #counts and frequencies
        self.counts = None
        self.freqs  = None
        self.pseudocounts =  None

        #regularization and initialisation
        self.regularization = None
        self.reg_type = None
        self.reg_scaling = None
        self.single_potential_init = None
        self.pair_potential_init = None

        #Compute alternative scores (omes or mi)
        self.alternative_score = None

        #variables
        self.x_single = None
        self.x_pair = None

        #obj function
        self.f = None

        #optimization algorithm
        self.alg = None

        #results of optimization
        self.fx = None
        self.x = None
        self.algret = None
        self.mat = None
        self.meta = {}

        #compute apc
        self.apc=False

        #write files
        self.cd_alnfile = None
        self.outrawfile = None
        self.outmsgpackfile = None
        self.outmodelprobmsgpackfile = None

    def __repr__(self):

        def walk_dict(d, depth=24, repr_str=""):
            for k,v in d.iteritems():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, list) or isinstance(item, dict):
                            repr_str += walk_dict(item, depth, "")
                        else:
                            repr_str += "{0:>{1}}: {2:>8}\n".format(k,depth,v)
                elif isinstance(v, dict):
                    repr_str += "\n{0:>{1}}\n".format(k, depth)
                    repr_str += walk_dict(v,depth, "")
                else:
                    repr_str += "{0:>{1}}: {2:>8}\n".format(k,depth,v)
            return repr_str


        repr_str = walk_dict(self.create_meta_data())
        return repr_str

    def create_meta_data(self):

        meta={}

        meta['version'] = __version__
        meta['method'] = 'ccmpred-py'

        meta['workflow'] = []
        meta['workflow'].append({})
        meta['workflow'][0]['timestamp'] = str(datetime.datetime.now())


        meta['workflow'][0]['msafile'] = {}
        meta['workflow'][0]['msafile']['neff'] = self.neff
        meta['workflow'][0]['msafile']['nrow'] = self.N
        meta['workflow'][0]['msafile']['ncol'] = self.L
        meta['workflow'][0]['msafile']['file'] = self.alignment_file
        meta['workflow'][0]['msafile']['max_gap_ratio'] = self.max_gap_ratio

        meta['workflow'][0]['pseudocounts'] = {}
        meta['workflow'][0]['pseudocounts']['pseudocount_type'] = self.pseudocounts.pseudocount_type
        meta['workflow'][0]['pseudocounts']['pseudocount_n_single'] = self.pseudocounts.pseudocount_n_single
        meta['workflow'][0]['pseudocounts']['pseudocount_n_pair'] = self.pseudocounts.pseudocount_n_pair
        meta['workflow'][0]['pseudocounts']['remove_gaps'] = self.pseudocounts.remove_gaps
        meta['workflow'][0]['pseudocounts']['pseudocount_ratio_single'] = self.pseudocounts.pseudocount_ratio_single
        meta['workflow'][0]['pseudocounts']['pseudocount_ratio_pair'] = self.pseudocounts.pseudocount_ratio_pair


        meta['workflow'][0]['weighting'] = {}
        meta['workflow'][0]['weighting']['type'] = self.weighting_type
        meta['workflow'][0]['weighting']['ignore_gaps'] = self.weighting.ignore_gaps
        meta['workflow'][0]['weighting']['cutoff'] = self.weighting.cutoff


        meta['workflow'][0]['parameters'] = {}
        meta['workflow'][0]['parameters']['apc'] = self.apc


        meta['workflow'][0]['results'] = {}
        meta['workflow'][0]['results']['matfile'] = self.mat_file


        if self.alternative_score is not None:
            meta['workflow'][0]['alternative_score'] = self.alternative_score
            meta['workflow'][0]['results']['opt_code'] = 0

        meta['workflow'][0]['initialisation'] = {}
        meta['workflow'][0]['initialisation']['single_potential_init'] = self.single_potential_init
        meta['workflow'][0]['initialisation']['pair_potential_init'] = self.pair_potential_init

        meta['workflow'][0]['regularization'] = {}
        meta['workflow'][0]['regularization']['regularization_type'] =     self.reg_type
        meta['workflow'][0]['regularization']['regularization_scaling'] =  self.reg_scaling
        if self.regularization is not None:
            meta['workflow'][0]['regularization']['lambda_single'] = self.regularization.lambda_single
            meta['workflow'][0]['regularization']['lambda_pair'] = self.regularization.lambda_pair
            meta['workflow'][0]['regularization']['lambda_pair_factor'] = self.regularization.lambda_pair_factor

        if self.alg is not None:
            meta['workflow'][0]['optimization'] = {}
            meta['workflow'][0]['optimization']['method'] = self.alg.__class__.__name__
            meta['workflow'][0]['optimization'].update(self.alg.get_parameters())

            meta['workflow'][0]['progress'] = {}
            meta['workflow'][0]['progress']['plotfile'] = self.alg.progress.plotfile

            meta['workflow'][0]['results']['num_iterations'] = len(self.alg.progress.optimization_log)
            meta['workflow'][0]['results']['opt_message'] = self.algret['message']
            meta['workflow'][0]['results']['opt_code'] = self.algret['code']
            meta['workflow'][0]['results']['fx_final'] = self.fx

        if self.f is not None:
            meta['workflow'][0]['obj_function'] = {}
            meta['workflow'][0]['obj_function']['name'] = self.f.__class__.__name__
            meta['workflow'][0]['obj_function'].update(self.f.get_parameters())


        if self.cd_alnfile:
            meta['workflow'][0]['results']['cd_alignmentfile'] = self.cd_alnfile
        if self.outrawfile:
            meta['workflow'][0]['results']['rawfile'] = self.outrawfile
        if self.outmsgpackfile:
            meta['workflow'][0]['results']['msgpackfile'] = self.outmsgpackfile
        if self.outmodelprobmsgpackfile:
            meta['workflow'][0]['results']['modelprobmsgpackfile'] = self.outmodelprobmsgpackfile

        return meta

    def read_alignment(self, aln_format="psicov", max_gap_ratio=100):
        self.msa = io.read_msa(self.alignment_file, aln_format)
        self.msa, self.gapped_positions = gaps.remove_gapped_positions(self.msa, max_gap_ratio)
        self.max_gap_ratio = max_gap_ratio

        self.N = self.msa.shape[0]
        self.L = self.msa.shape[1]

        print("{0} is of length L={1}. Alignemnt has {2} sequences.".format(
            self.protein, self.L, self.N))

    def compute_sequence_weights(self, weighting_type, ignore_gaps=False, cutoff=0.8):

        self.weighting = SequenceWeights(ignore_gaps, cutoff)
        self.weights = getattr(self.weighting, weighting_type)(self.msa)
        self.weighting_type = weighting_type

        self.neff   = np.sum(self.weights)
        self.diversity = np.sqrt(self.neff)/self.L

        print("Number of effective sequences after {0} reweighting (id-threshold={1}, ignore_gaps={2}): {3:g}. Neff(HHsuite-like)={4}".format(
            self.weighting_type, self.weighting.cutoff, self.weighting.ignore_gaps, self.neff, self.weighting.get_HHsuite_neff(self.msa)))
        print("Alignment has diversity={0}.".format(np.round(self.diversity, decimals=3)))

    def compute_frequencies(self, pseudocount_type, pseudocount_n_single=1,  pseudocount_n_pair=1, dev_center_v=False):

        self.pseudocounts = PseudoCounts(self.msa, self.weights)
        self.counts = self.pseudocounts.counts

        if dev_center_v:
            self.freqs = self.pseudocounts.calculate_frequencies_dev_center_v()
        else:
            self.freqs = self.pseudocounts.calculate_frequencies(
                pseudocount_type,
                pseudocount_n_single,
                pseudocount_n_pair,
                remove_gaps=False)


        print("Calculating AA Frequencies with {0} percent pseudocounts ({1} {2} {3})".format(
            np.round(self.pseudocounts.pseudocount_ratio_single, decimals=5),
            self.pseudocounts.pseudocount_type,
            self.pseudocounts.pseudocount_n_single,
            self.pseudocounts.pseudocount_n_pair)
        )

    def compute_omes(self, omes_fodoraldrich=False):

        if omes_fodoraldrich:
            self.alternative_score = "omes_fodoraldrich"
            print("Will compute Observed Minus Expected Squared (OMES) Covariance score (acc to Fodor & Aldrich).")
            self.mat = local_methods.compute_omes_freq(self.counts, self.freqs, True)
        else:
            self.alternative_score = "omes"
            print("Will compute Observed Minus Expected Squared (OMES) Covariance score (acc to Kass & Horovitz).")
            self.mat = local_methods.compute_omes_freq(self.counts, self.freqs, False)

    def compute_mutual_info(self, mi_normalized=False, mi_pseudocounts=False):

        if mi_pseudocounts:
            self.alternative_score = "mi_pseudocounts"
            print("Will compute mutual information score with pseudocounts")
            self.mat = local_methods.compute_mi_pseudocounts(self.freqs)
        elif mi_normalized:
            self.alternative_score = "mi_normalized"
            print("Will compute normalized mutual information score")
            self.mat = local_methods.compute_mi(self.counts, normalized=True)
        else:
            self.alternative_score = "mi"
            print("Will compute mutual information score")
            self.mat = local_methods.compute_mi(self.counts)

    def specify_regularization(self, lambda_single, lambda_pair_factor, reg_type="center-v", scaling="L", dev_center_v=False ):

        self.reg_type = reg_type
        self.reg_scaling = scaling

        if dev_center_v or reg_type == "center-v":
            prior_v_mu   = centering.center_v(self.freqs)
        else:
            prior_v_mu   = centering.center_zero(self.freqs)

        REG_L2_SCALING = {
            "L" : self.L-1,
            "diversity" :   self.diversity,
            "1": 1
        }
        scaling = REG_L2_SCALING[scaling]
        self.regularization = regularization.L2(lambda_single, lambda_pair_factor, scaling, prior_v_mu)

    def intialise_potentials(self, initrawfile=None, vanilla_ccmpred=False):


        if initrawfile:
            if not os.path.exists(initrawfile, ):
                print("Init file {0} does not exist! Exit".format(initrawfile))
                sys.exit(0)

            raw_potentials = raw.parse_msgpack(initrawfile)
            self.x_single, self.x_pair = raw_potentials.x_single, raw_potentials.x_pair

            self.single_potential_init  = initrawfile
            self.pair_potential_init    = initrawfile

        else:
            v = self.regularization.center_x_single
            self.single_potential_init  = self.reg_type
            self.pair_potential_init    = "zero"

            if vanilla_ccmpred:
                freqs_for_init = self.pseudocounts.calculate_frequencies_vanilla(self.msa)
                v = centering.center_vanilla(freqs_for_init)
                self.single_potential_init = "ccmpred-vanilla"
                # besides initialisation and regularization, there seems to be another difference in gradient calculation between CCMpred vanilla and CCMpred-dev-center-v
                # furthermore initialisation does NOT assure sum_a(v_ia) == 1

            # default initialisation of parameters
            self.x_single, self.x_pair = initialise_potentials.init(self.L, v)

    def minimize(self, obj_fun, alg, plotfile=None):


        #initialize objective function
        self.f = obj_fun

        #initialise optimizer
        self.alg = alg


        print("\n Will optimize {0} {1} variables wrt {2} and {3}".format(self.f.x.size, self.f.x.dtype, self.f, self.regularization))
        print("Optimizer: {0}".format(self.alg))

        if plotfile:
            print("The optimization log file will be written to {0}".format(plotfile))


        self.fx, self.x, self.algret    = self.alg.minimize(self.f, self.f.x)
        self.x_single, self.x_pair      = self.f.finalize(self.x)

        condition = "Finished" if self.algret['code'] >= 0 else "Exited"
        print("\n{0} with code {code} -- {message}\n".format(condition, **self.algret))


        if self.max_gap_ratio < 100:
            self.x_single, self.x_pair = gaps.backinsert_gapped_positions(self.x_single, self.x_pair, self.gapped_positions)
            self.L = self.x_single.shape[0]

        self.mat = io.contactmatrix.frobenius_score(self.x_pair)

        # #Refine with persistent CD
        # refine=False
        # if refine:
        #     if opt.alpha0 == 0:
        #         alg.alpha0 = 1e-3 * (np.log(protein['Neff']) / protein['L'])
        #     if opt.decay_rate == 0:
        #         alg.decay_rate = 1e-6 / (np.log(protein['Neff']) / protein['L'])
        #     opt.cd_persistent=True
        #     opt.minibatch_size=0
        #     f = OBJ_FUNC[opt.objfun](opt, msa, freqs, weights, raw_init, regularization)
        #     fx, x, algret = alg.minimize(f, x)

    def recenter_potentials(self):

        #perform checks on potentials: do v_i and w_ij sum to 0?
        check_x_single  = sanity_check.check_single_potentials(self.x_single, verbose=1, epsilon=1e-2)
        check_x_pair  = sanity_check.check_pair_potentials(self.x_pair, verbose=1, epsilon=1e-2)

        #enforce sum(wij)=0 and sum(v_i)=0
        if not check_x_single or not check_x_pair:
            print("Enforce sum(v_i)=0 and sum(w_ij)=0 by centering potentials at zero.")
            self.x_single, self.x_pair = sanity_check.centering_potentials(self.x_single, self.x_pair)

    def write_sampled_alignment(self, cd_alnfile):

        self.cd_alnfile = cd_alnfile
        print("\nWriting sampled alignment to {0}".format(cd_alnfile))
        msa_sampled = self.f.msa_sampled

        with open(cd_alnfile, "w") as f:
            io.alignment.write_msa_psicov(f, msa_sampled)

    def write_mat(self, apc=False):

        if apc:
            self.apc=True
            self.mat = io.contactmatrix.apc(self.mat)

        self.meta = self.create_meta_data()

        io.contactmatrix.write_matrix(self.mat_file, self.mat, self.meta)

    def write_raw(self, outrawfile):

        self.outrawfile = outrawfile
        raw_out = raw.CCMRaw(self.L, self.x_single[:, :20], self.x_pair[:, :, :21, :21], self.meta)
        print("\nWriting raw-formatted potentials to {0}".format(outrawfile))
        raw.write_oldraw(outrawfile, raw_out)

    def write_binary_raw(self, outmsgpackfile ):

        self.outmsgpackfile = outmsgpackfile
        raw_out = raw.CCMRaw(self.L, self.x_single[:, :20], self.x_pair[:, :, :21, :21], self.meta)
        print("\nWriting msgpack-formatted potentials to {0}".format(outmsgpackfile))
        raw.write_msgpack(outmsgpackfile, raw_out)

    def write_binary_modelprobs(self, outmodelprobmsgpackfile):

        self.outmodelprobmsgpackfile = outmodelprobmsgpackfile
        print("\nWriting msgpack-formatted model probabilties to {0}".format(outmodelprobmsgpackfile))

        # if opt.max_gap_ratio < 100:
        #     msa = ccmpred.io.alignment.read_msa(opt.alnfile, opt.aln_format)
        #     freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, opt.pseudocounts[0], pseudocount_n_single=opt.pseudocounts[1], pseudocount_n_pair=opt.pseudocount_pair_count)
        # if self.opt.dev_center_v:
        #     freqs = ccmpred.pseudocounts.calculate_frequencies(msa, weights, ccmpred.pseudocounts.constant_pseudocounts, pseudocount_n_single=1, pseudocount_n_pair=1, remove_gaps=True)

        model_probabilities.write_msgpack(outmodelprobmsgpackfile, self.x_pair, self.neff, self.freqs, self.regularization.lambda_pair)

