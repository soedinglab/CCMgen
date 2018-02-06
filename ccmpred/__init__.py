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
import raw
import parameter_handling
import sanity_check
from regularization import L2

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
        self.max_gap_ratio  = 100
        self.min_coverage   = 0
        self.N              = None
        self.L              = None
        self.neff           = None
        self.diversity      = None

        #sequence weighting
        self.weighting = None
        self.weighting_type = None
        self.weights   = None


        #counts and frequencies in class ccmpred.pseudocounts.PseudoCounts
        self.pseudocounts =  None

        #regularization and initialisation
        #class ccmpred.regularization.L2
        self.regularization = None
        self.reg_type = None
        self.reg_scaling = None
        self.single_potential_init = None
        self.pair_potential_init = None

        #variables
        self.x_single = None
        self.x_pair = None
        self.g_x_single = None
        self.g_x_pair = None

        #obj function
        self.f = None

        #optimization algorithm
        self.alg = None

        #results of optimization
        self.fx = None
        self.x = None
        self.algret = None
        self.mats = {}
        self.meta = {}


        #write files
        self.cd_alnfile = None
        self.out_binary_raw_file = None

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

    def create_meta_data(self, mat_name=None):

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
        meta['workflow'][0]['msafile']['min_coverage'] = self.min_coverage

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

        if mat_name is not None:
            meta['workflow'][0]['contact_map'] = {}
            meta['workflow'][0]['contact_map']['correction'] = self.mats[mat_name]['correction']
            if 'scaling_factor_eta' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['scaling_factor_eta'] = self.mats[mat_name]['scaling_factor_eta']
            if 'nr_states' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['nr_states'] = self.mats[mat_name]['nr_states']
            if 'log' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['nr_states'] = self.mats[mat_name]['log']
            meta['workflow'][0]['contact_map']['score'] = self.mats[mat_name]['score']
            meta['workflow'][0]['contact_map']['matfile'] = self.mats[mat_name]['mat_file']


        meta['workflow'][0]['results'] = {}
        meta['workflow'][0]['results']['opt_code'] = 0 #default

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

            meta['workflow'][0]['results']['opt_message'] = self.algret['message']
            meta['workflow'][0]['results']['opt_code'] = self.algret['code']
            meta['workflow'][0]['results']['num_iterations'] = self.algret['num_iterations']
            meta['workflow'][0]['results']['fx_final'] = self.fx

        if self.f is not None:
            meta['workflow'][0]['obj_function'] = {}
            meta['workflow'][0]['obj_function']['name'] = self.f.__class__.__name__
            meta['workflow'][0]['obj_function'].update(self.f.get_parameters())


        if self.cd_alnfile:
            meta['workflow'][0]['results']['cd_alignmentfile'] = self.cd_alnfile
        if self.out_binary_raw_file:
            meta['workflow'][0]['results']['out_binary_raw_file'] = self.out_binary_raw_file


        return meta

    def read_alignment(self, aln_format="psicov", max_gap_ratio=100, min_coverage=0):
        self.msa = io.read_msa(self.alignment_file, aln_format)

        if min_coverage > 0:
            self.msa = gaps.remove_gapped_sequences(self.msa, min_coverage)
            self.min_coverage=min_coverage

        if max_gap_ratio < 100:
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

    def compute_frequencies(self, pseudocount_type, pseudocount_n_single=1,  pseudocount_n_pair=1):

        self.pseudocounts = PseudoCounts(self.msa, self.weights)

        self.pseudocounts.calculate_frequencies(
            pseudocount_type,
            pseudocount_n_single,
            pseudocount_n_pair,
            remove_gaps=False
        )

        print("Calculating AA Frequencies with {0} percent pseudocounts ({1} {2} {3})".format(
            np.round(self.pseudocounts.pseudocount_ratio_single, decimals=5),
            self.pseudocounts.pseudocount_type,
            self.pseudocounts.pseudocount_n_single,
            self.pseudocounts.pseudocount_n_pair)
        )

    def compute_omes(self, omes_fodoraldrich=False):
        """
        Compute OMES score (chi-square statistic) according to Kass and Horovitz 2002
        :param omes_fodoraldrich: modified version according to Fodor & Aldrich 2004
        :return:
        """

        mat_path, mat_name = os.path.split(self.mat_file)

        if omes_fodoraldrich:
            print("Will compute Observed Minus Expected Squared (OMES) Covariance score (acc to Fodor & Aldrich).")
            self.mats["omes_fodoraldrich"] = {
                'mat': local_methods.compute_omes_freq(self.pseudocounts.counts, self.pseudocounts.freqs, True),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".omes_fa." + mat_name.split(".")[-1],
                'score': "omes_fodoraldrich",
                'correction': "no"
            }
        else:
            print("Will compute Observed Minus Expected Squared (OMES) Covariance score (acc to Kass & Horovitz).")
            self.mats["omes"] = {
                'mat': local_methods.compute_omes_freq(self.pseudocounts.counts, self.pseudocounts.freqs, False),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".omes." + mat_name.split(".")[-1],
                'score': "omes",
                'correction': "no"
            }

    def compute_mutual_info(self, mi_normalized=False, mi_pseudocounts=False):
        """
        Compute mutual information and variations thereof

        :param mi_normalized: compute the normalized mutual information
        :param mi_pseudocounts: compute mutual information using pseudo counts
        :return:
        """

        mat_path, mat_name = os.path.split(self.mat_file)

        if mi_pseudocounts:
            print("\nComputing mutual information score with pseudocounts")
            self.mats["mutual information (pseudo counts)"] = {
                'mat' : local_methods.compute_mi_pseudocounts(self.pseudocounts.freqs),
                'mat_file':  mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".mi_pc." + mat_name.split(".")[-1],
                'score': "mutual information with pseudo counts",
                'correction': "no"
            }
        if mi_normalized:
            print("\nComputing normalized mutual information score")
            self.mats["normalized mutual information"] = {
                'mat': local_methods.compute_mi(self.pseudocounts.counts, normalized=True),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".nmi." + mat_name.split(".")[-1],
                'score': "normalized mutual information",
                'correction': "no"
            }

        print("\nComputing mutual information score")
        self.mats["mutual information"] = {
            'mat': local_methods.compute_mi(self.pseudocounts.counts),
            'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".mi." + mat_name.split(".")[-1],
            'score': "mutual information",
            'correction': "no"
        }

    def specify_regularization(self, lambda_single, lambda_pair_factor, reg_type="v-center", scaling="L"):
        """

        use L2 regularization for single potentials and pair potentials

        :param lambda_single: regularization coefficient for single potentials
        :param lambda_pair_factor: regularization factor for pair potentials (scaled by reg-type)
        :param reg_type: defines location of regularizer for single potentials
        :param scaling: multiplier to define regularization coefficient for pair potentials
        :return:
        """

        #save setting for meta data
        self.reg_type = reg_type
        self.reg_scaling = scaling

        if reg_type == "v-center":
            prior_v_mu = centering.center_v(self.pseudocounts.freqs)
        else:
            prior_v_mu = centering.center_zero(self.pseudocounts.freqs)

        if scaling == "L":
            multiplier = self.L-1
        else:
            multiplier = 1

        self.regularization = L2(lambda_single, lambda_pair_factor, multiplier, prior_v_mu)
        print(self.regularization)

    def intialise_potentials(self, initrawfile=None):

        if initrawfile:
            if not os.path.exists(initrawfile):
                print("Init file {0} does not exist! Exit".format(initrawfile))
                sys.exit(0)

            try:
                raw_potentials = raw.parse_msgpack(initrawfile)
            except:
                print("Unexpected error whil reading binary raw file {0}: {1}".format(initrawfile, sys.exc_info()[0]))
                sys.exit(0)

            self.x_single, self.x_pair = raw_potentials.x_single, raw_potentials.x_pair

            #save setting for meta data
            self.single_potential_init  = initrawfile
            self.pair_potential_init    = initrawfile

        else:
            # default initialisation of parameters:
            # initialise single potentials from regularization prior
            self.x_single = self.regularization.center_x_single

            # initialise pair potnetials at zero
            self.x_pair = np.zeros((self.L, self.L, 21, 21))


            # save settting for meta data
            self.single_potential_init  = self.reg_type
            self.pair_potential_init    = "zero"

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

        ###experimental - only for pll with cg
        #g_x = self.alg.get_gradient_x()
        #self.g_x_single, self.g_x_pair = self.f.finalize(g_x)


        condition = "Finished" if self.algret['code'] >= 0 else "Exited"
        print("\n{0} with code {code} -- {message}\n".format(condition, **self.algret))



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
        """
        Enforce Gauge:
            - 0 = sum_a,b w_ij(a,b)
            - 0 = sum_a v_i(a)

        :return:
        """

        #perform checks on potentials: do v_i and w_ij sum to 0?
        check_x_single = sanity_check.check_single_potentials(self.x_single, verbose=1, epsilon=1e-2)
        check_x_pair = sanity_check.check_pair_potentials(self.x_pair, verbose=1, epsilon=1e-2)

        #enforce sum(wij)=0 and sum(v_i)=0
        if not check_x_single or not check_x_pair:
            print("Enforce sum(v_i)=0 and sum(w_ij)=0 by centering potentials at zero.")
            self.x_single, self.x_pair = sanity_check.centering_potentials(self.x_single, self.x_pair)

    def compute_contact_matrix(self, recenter_potentials=False, frob=True):
        """
        Compute contact scores and save in dictionary self.mats

        :param recenter_potentials: Ensure sum(v_i)=0 and sum(w_ij)=0
        :param frob:    compute frobenius norm of couplings
        :return:
        """

        if recenter_potentials:
            self.recenter_potentials()

        mat_path, mat_name = os.path.split(self.mat_file)

        if frob:
            self.mats["frobenius"] = {
                'mat':    io.contactmatrix.frobenius_score(self.x_pair),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".frobenius." + mat_name.split(".")[-1],
                'score': "frobenius",
                'correction': "no correction"
            }

    def compute_correction(self, apc=False, entropy_correction=False, joint_entropy=False, sergeys_jec=False):
        """
        Compute bias correction for raw contact maps

        :param apc:     apply average product correction
        :param entropy_correction: apply entropy correction
        :param joint_entropy: use joint entropy instead of geometric mean of marginal entropies
        :param sergeys_jec: scale couplings with a joint entropy correction
        :return:
        """

        mat_path, mat_name = os.path.split(self.mat_file)

        #iterate over all raw contact matrices
        for score_mat in self.mats.keys():

            mat_dict = self.mats[score_mat]
            score = mat_dict['score']
            score_matrix = mat_dict['mat']

            if apc:
                self.mats[score_mat + "-apc"]={
                    'mat': io.contactmatrix.apc(score_matrix),
                    'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + "." + score + ".apc." + mat_name.split(".")[-1],
                    'score': score,
                    'correction': "apc"
                    }



            if entropy_correction and score == "frobenius":
                nr_states = 21
                log = np.log2


                # use amino acid frequencies including gap states and with pseudo-counts
                single_freq = self.pseudocounts.freqs[0]


                scaling_factor_eta, mat_corrected = io.contactmatrix.compute_local_correction(
                    single_freq, self.x_pair, self.neff, self.regularization.lambda_pair,
                    squared=False, entropy=True, nr_states=nr_states, log=log
                )

                self.mats[score_mat+"-ec."+ str(nr_states) + "." + str(log.__name__)] = {
                    'mat': mat_corrected,
                    'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + "." + score +
                                ".ec." + str(nr_states) + "." + str(log.__name__) + "." +
                                mat_name.split(".")[-1],
                    'score': score,
                    'correction': "entropy_correction",
                    'scaling_factor_eta': scaling_factor_eta,
                    'nr_states': nr_states,
                    'log': log.__name__
                }


            if joint_entropy and score == "frobenius":
                nr_states = 21
                log = np.log2

                # use amino acid frequencies including gap states and with pseudo-counts
                pair_freq = self.pseudocounts.freqs[1]

                scaling_factor_eta, mat_corrected = io.contactmatrix.compute_joint_entropy_correction(
                    pair_freq, self.neff, self.regularization.lambda_pair, self.x_pair,
                    nr_states = nr_states, log=log
                )

                self.mats[score_mat + "-jec."+ str(nr_states) + "." + str(log.__name__)] = {
                    'mat': mat_corrected,
                    'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + "." + score +
                                ".jec." + str(nr_states) + "." + str(log.__name__) + "." +
                                mat_name.split(".")[-1],
                    'score': score,
                    'correction': "joint_entropy_correction",
                    'scaling_factor_eta': scaling_factor_eta,
                    'nr_states': nr_states,
                    'log': log.__name__
                }

            if sergeys_jec and score == "frobenius":
                nr_states = 21
                log = np.log2

                # use amino acid frequencies including gap states and with pseudo-counts
                pair_freq = self.pseudocounts.freqs[1]

                mat_corrected = io.contactmatrix.compute_corrected_mat_sergey_style(
                    pair_freq, self.x_pair, nr_states = nr_states, log=log)

                self.mats[score_mat + "-sjec."+ str(nr_states) + "." + str(log.__name__)] = {
                    'mat': mat_corrected,
                    'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + "." + score +
                                ".sjec." + str(nr_states) + "." + str(log.__name__) + "." +
                                mat_name.split(".")[-1],
                    'score': score,
                    'correction': "sergeys joint entropy correction",
                    'nr_states': nr_states,
                    'log': log.__name__
                }

    def write_sampled_alignment(self, cd_alnfile):

        self.cd_alnfile = cd_alnfile
        print("\nWriting sampled alignment to {0}".format(cd_alnfile))
        msa_sampled = self.f.msa_sampled

        with open(cd_alnfile, "w") as f:
            io.alignment.write_msa_psicov(f, msa_sampled)

    def write_matrix(self):
        """
        Write (corrected) contact maps to text file including meta data
        :return:
        """

        for mat_name, mat_dict in self.mats.iteritems():

            mat = mat_dict['mat']
            if self.max_gap_ratio < 100:
                mat = gaps.backinsert_gapped_positions_mat(mat, self.gapped_positions)
                self.L = mat.shape[0]

            meta = self.create_meta_data(mat_name)
            io.contactmatrix.write_matrix(mat_dict['mat_file'], mat, meta)

    def write_binary_raw(self, out_binary_raw_file):
        """
        Write single and pair potentials including meta data to msgpack-formatted binary raw file

        :param out_binary_raw_file: path to out file
        :return:
        """

        if self.max_gap_ratio < 100:
            self.x_single, self.x_pair = gaps.backinsert_gapped_positions(
                self.x_single, self.x_pair, self.gapped_positions)
            self.L = self.x_single.shape[0]

        self.out_binary_raw_file = out_binary_raw_file
        meta = self.create_meta_data()

        raw_out = raw.CCMRaw(self.L, self.x_single[:, :20], self.x_pair[:, :, :21, :21], meta)
        print("\nWriting msgpack-formatted potentials to {0}".format(out_binary_raw_file))
        raw.write_msgpack(out_binary_raw_file, raw_out)

