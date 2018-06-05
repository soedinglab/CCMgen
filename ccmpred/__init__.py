__version__ = '1.0.0'

import datetime
import os
import sys
import numpy as np
import io
import ccmpred.gaps
import ccmpred.counts
from ccmpred.pseudocounts import PseudoCounts
import ccmpred.locmeth
import ccmpred.centering
import ccmpred.raw
import ccmpred.parameter_handling
import ccmpred.sanity_check
from ccmpred.regularization import L2
import ccmpred.plotting
import ccmpred.sampling
import ccmpred.weighting
import ccmpred.monitor.progress as pr
import ccmpred.objfun.pll as pll
import ccmpred.objfun.cd as cd
import ccmpred.algorithm.gradient_descent as gd
import ccmpred.algorithm.lbfgs as lbfgs


class CCMpred():
    """
    CCMpred is a fast python implementation of the maximum pseudo-likelihood class of contact prediction methods.
    From an alignment given as alnfile, it will maximize the likelihood of the pseudo-likelihood of a Potts model
    with 21 states for amino acids and gaps.

    The L2 norms of the pairwise coupling potentials will be written to the output matfile.

    """

    def __init__(self):

        self.alignment_file = None
        self.mat_file       = None
        self.pdb_file       = None
        self.init_raw_file  = None
        self.protein        = None

        self.contact_threshold = 8
        self.non_contact_indices = None

        self.msa            = None
        self.gapped_positions = None
        self.max_gap_pos    = 100
        self.max_gap_seq    = 100
        self.N              = None
        self.L              = None
        self.neff           = None
        self.neff_entropy = None
        self.diversity      = None

        #sequence weighting
        self.weighting_type = None
        self.weights   = None
        self.wt_cutoff  = None


        #counts and frequencies in class ccmpred.pseudocounts.PseudoCounts
        self.pseudocounts =  None

        #regularization and parameter initialisation
        self.regularization = None
        self.reg_type = None
        self.reg_scaling = None
        self.single_prior = None
        self.single_potential_init = None
        self.pair_potential_init = None

        #variables
        self.x_single = None
        self.x_pair = None
        self.g_x_single = None
        self.g_x_pair = None

        #objective function (pLL, CD, PCD)
        self.f = None

        #optimization algorithm (CG, LBFGS, GD, ADAM)
        self.alg = None

        #optimization progress logger
        self.progress = None

        #results of optimization
        self.fx = None
        self.x = None
        self.algret = None
        self.mats = {}
        self.meta = {}

        #write files
        self.sample_alnfile = None
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
        meta['method'] = 'CCMpredPy'

        meta['workflow'] = []
        meta['workflow'].append({})
        meta['workflow'][0]['timestamp'] = str(datetime.datetime.now())


        meta['workflow'][0]['msafile'] = {}
        meta['workflow'][0]['msafile']['neff'] = self.neff
        meta['workflow'][0]['msafile']['nrow'] = self.N
        meta['workflow'][0]['msafile']['ncol'] = self.L
        meta['workflow'][0]['msafile']['file'] = self.alignment_file
        meta['workflow'][0]['msafile']['max_gap_pos'] = self.max_gap_pos
        meta['workflow'][0]['msafile']['max_gap_seq'] = self.max_gap_seq

        meta['workflow'][0]['pseudocounts'] = {}
        meta['workflow'][0]['pseudocounts']['pseudocount_type'] = self.pseudocounts.pseudocount_type
        meta['workflow'][0]['pseudocounts']['pseudocount_n_single'] = self.pseudocounts.pseudocount_n_single
        meta['workflow'][0]['pseudocounts']['pseudocount_n_pair'] = self.pseudocounts.pseudocount_n_pair
        meta['workflow'][0]['pseudocounts']['remove_gaps'] = self.pseudocounts.remove_gaps
        meta['workflow'][0]['pseudocounts']['pseudocount_ratio_single'] = self.pseudocounts.pseudocount_ratio_single
        meta['workflow'][0]['pseudocounts']['pseudocount_ratio_pair'] = self.pseudocounts.pseudocount_ratio_pair


        meta['workflow'][0]['weighting'] = {}
        meta['workflow'][0]['weighting']['type'] = self.weighting_type
        meta['workflow'][0]['weighting']['cutoff'] = self.wt_cutoff

        if mat_name is not None:
            meta['workflow'][0]['contact_map'] = {}
            meta['workflow'][0]['contact_map']['correction'] = self.mats[mat_name]['correction']
            meta['workflow'][0]['contact_map']['score'] = self.mats[mat_name]['score']
            meta['workflow'][0]['contact_map']['matfile'] = self.mats[mat_name]['mat_file']
            if 'scaling_factor' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['scaling_factor'] = self.mats[mat_name]['scaling_factor']
            if 'nr_states' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['nr_states'] = self.mats[mat_name]['nr_states']
            if 'log' in self.mats[mat_name].keys():
                meta['workflow'][0]['contact_map']['log'] = self.mats[mat_name]['log']



        meta['workflow'][0]['results'] = {}
        meta['workflow'][0]['results']['opt_code'] = 0 #default

        meta['workflow'][0]['initialisation'] = {}
        meta['workflow'][0]['initialisation']['single_potential_init'] = self.single_potential_init
        meta['workflow'][0]['initialisation']['pair_potential_init'] = self.pair_potential_init

        meta['workflow'][0]['regularization'] = {}
        meta['workflow'][0]['regularization']['regularization_type'] =     self.reg_type
        meta['workflow'][0]['regularization']['regularization_scaling'] =  self.reg_scaling
        meta['workflow'][0]['regularization']['single_prior'] =  self.single_prior

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
            meta['workflow'][0]['results']['runtime'] = self.algret['runtime']
            meta['workflow'][0]['results']['fx_final'] = self.fx

        if self.f is not None:

            meta['workflow'][0]['obj_function'] = {}
            meta['workflow'][0]['obj_function']['name'] = self.f.__class__.__name__
            meta['workflow'][0]['obj_function'].update(self.f.get_parameters())

        if self.sample_alnfile:
            meta['workflow'][0]['results']['sample_alignment_file'] = self.sample_alnfile
        if self.out_binary_raw_file:
            meta['workflow'][0]['results']['out_binary_raw_file'] = self.out_binary_raw_file


        return meta

    def set_alignment_file(self, alnfile=None):

        if alnfile is not None:
            if os.path.exists(alnfile):
                self.alignment_file = alnfile
            else:
                print("Alignment file {0} does not exist!".format(alnfile))
                sys.exit(0)

            self.protein = os.path.basename(alnfile).split(".")[0]

    def set_matfile(self, matfile=None):

        if matfile is not None:
            self.mat_file = matfile

    def set_pdb_file(self, pdbfile=None):

        if pdbfile is not None:
            if os.path.exists(pdbfile):
                self.pdb_file = pdbfile
            else:
                print("PDB file {0} does not exist!".format(pdbfile))
                sys.exit(0)

    def set_initraw_file(self, initrawfile=None):

        if initrawfile is not None:
            if os.path.exists(initrawfile):
                self.init_raw_file = initrawfile
            else:
                print("Binary raw coupling file {0} does not exist!".format(initrawfile))
                sys.exit(0)

    def read_alignment(self, aln_format="psicov", max_gap_pos=100, max_gap_seq=100):
        self.msa = io.read_msa(self.alignment_file, aln_format)

        if max_gap_seq < 100:
            self.msa = gaps.remove_gapped_sequences(self.msa, max_gap_seq)
            self.max_gap_seq=max_gap_seq

        if max_gap_pos < 100:
            self.msa, self.gapped_positions = gaps.remove_gapped_positions(self.msa, max_gap_pos)
            self.max_gap_pos = max_gap_pos

        self.N = self.msa.shape[0]
        self.L = self.msa.shape[1]
        self.diversity = np.sqrt(self.N)/self.L
        self.neff_entropy = ccmpred.weighting.get_HHsuite_neff(self.msa)

        print("{0} is of length L={1} and there are {2} sequences in the alignment.".format(
            self.protein, self.L, self.N))
        print("Alignment has diversity [sqrt(N)/L]={0} and Neff(HHsuite-like)={1}.".format(np.round(self.diversity, decimals=3), np.round(self.neff_entropy, decimals=3)))

    def read_pdb(self, contact_threshold=8):
        self.contact_threshold = contact_threshold

        if self.max_gap_pos < 100:
            L = self.L + len(self.gapped_positions)
            distance_map = io.distance_map(self.pdb_file, L=L)

            indices = [i for i in range(L) if i not in self.gapped_positions]
            distance_map = distance_map[indices, :]
            distance_map = distance_map[:, indices]
        else:
            distance_map = io.distance_map(self.pdb_file, L=self.L)

        self.non_contact_indices = np.where(distance_map > contact_threshold)

    def compute_sequence_weights(self, weighting_type, cutoff=0.8):

        self.weights = ccmpred.weighting.WEIGHTING_TYPE[weighting_type](self.msa, cutoff)

        self.weighting_type = weighting_type
        self.wt_cutoff = cutoff
        self.neff   = np.sum(self.weights)

        print("Number of effective sequences after {0} reweighting (id-threshold={1}): {2:g}.".format(
            weighting_type, cutoff, self.neff))

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
                'mat': locmeth.compute_omes_freq(self.pseudocounts.counts, self.pseudocounts.freqs, True),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".omes_fa." + mat_name.split(".")[-1],
                'score': "omes_fodoraldrich",
                'correction': "no"
            }
        else:
            print("Will compute Observed Minus Expected Squared (OMES) Covariance score (acc to Kass & Horovitz).")
            self.mats["omes"] = {
                'mat': locmeth.compute_omes_freq(self.pseudocounts.counts, self.pseudocounts.freqs, False),
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
                'mat' : locmeth.compute_mi_pseudocounts(self.pseudocounts.freqs),
                'mat_file':  mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".mi_pc." + mat_name.split(".")[-1],
                'score': "mutual information with pseudo counts",
                'correction': "no"
            }
        if mi_normalized:
            print("\nComputing normalized mutual information score")
            self.mats["normalized mutual information"] = {
                'mat': locmeth.compute_mi(self.pseudocounts.counts, normalized=True),
                'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".nmi." + mat_name.split(".")[-1],
                'score': "normalized mutual information",
                'correction': "no"
            }

        print("\nComputing mutual information score")
        self.mats["mutual information"] = {
            'mat': locmeth.compute_mi(self.pseudocounts.counts),
            'mat_file': mat_path + "/" + ".".join(mat_name.split(".")[:-1]) + ".mi." + mat_name.split(".")[-1],
            'score': "mutual information",
            'correction': "no"
        }

    def specify_regularization(self, lambda_single, lambda_pair_factor,
                               reg_type="L2", scaling="L", single_prior="v-center"):
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
        self.single_prior = scaling

        if single_prior == "v-center":
            prior_v_mu = centering.center_v(self.pseudocounts.freqs)
        else:
            prior_v_mu = centering.center_zero(self.pseudocounts.freqs)

        if scaling == "L":
            multiplier = self.L-1
        else:
            multiplier = 1

        if reg_type == "L2":
            self.regularization = L2(lambda_single, lambda_pair_factor, multiplier, prior_v_mu)
        else:
            self.regularization = L1(lambda_single, lambda_pair_factor, multiplier, prior_v_mu)

        print(self.regularization)

    def intialise_potentials(self):

        if self.init_raw_file is not None:

            try:
                raw_potentials = raw.parse_msgpack(self.init_raw_file)
            except:
                print("Unexpected error whil reading binary raw file {0}: {1}".format(self.init_raw_file, sys.exc_info()[0]))
                sys.exit(0)

            print("\nSuccessfully loaded model parameters from {0}.".format(self.init_raw_file))
            self.x_single, self.x_pair = raw_potentials.x_single, raw_potentials.x_pair

            #in case positions with many gaps should be removed
            if self.gapped_positions is not None:
                indices = [i for i in range(raw_potentials.ncol) if i not in self.gapped_positions]
                self.x_single = self.x_single[indices, :]
                self.x_pair = self.x_pair[indices, :, :, :]
                self.x_pair = self.x_pair[:, indices, :, :]
                print("Removed parameters for positions with >{0}% gaps.".format(self.max_gap_pos))

            #save setting for meta data
            self.single_potential_init  = self.init_raw_file
            self.pair_potential_init    = self.init_raw_file

        else:
            # default initialisation of parameters:
            # initialise single potentials from regularization prior
            self.x_single = self.regularization.center_x_single

            # initialise pair potnetials at zero
            self.x_pair = np.zeros((self.L, self.L, 21, 21))


            # save settting for meta data
            self.single_potential_init  = self.reg_type
            self.pair_potential_init    = "zero"

    def initiate_logging(self, plot_file=None):
        # setup progress logging
        self.progress = pr.Progress()

        if plot_file is not None:

            plot_title = "L={0} N={1} Neff={2} Diversity={3}<br>".format(
                self.L, self.N, np.round(self.neff, decimals=3), np.round(self.diversity, decimals=3))
            self.progress.set_plot_title(plot_title)

            if plot_file.split(".")[-1] != "html":
                plot_file += ".html"

            print("Plot with optimization statistics will be written to {0}".format(plot_file))
            self.progress.set_plot_file(plot_file)

    def minimize(self, opt, plotfile=None):

        OBJ_FUNC = {
            "pll": lambda opt: pll.PseudoLikelihood(
                self.msa, self.weights, self.regularization, self.pseudocounts, self.x_single, self.x_pair),
            "cd": lambda opt: cd.ContrastiveDivergence(
                self.msa, self.weights, self.regularization, self.pseudocounts, self.x_single, self.x_pair,
                gibbs_steps=opt.cd_gibbs_steps,
                nr_seq_sample=opt.nr_seq_sample,
                persistent=opt.cd_persistent
            )
        }

        ALGORITHMS = {
            "pll": lambda opt: lbfgs.LBFGS(
                self.progress, maxit=opt.maxit, ftol=opt.ftol, max_linesearch=opt.max_linesearch, maxcor=opt.max_cor,
                non_contact_indices=self.non_contact_indices
            ),
            "cd": lambda opt: gd.gradientDescent(
                self.progress, self.neff, maxit=opt.maxit, alpha0=opt.alpha0, decay=opt.decay, decay_start=opt.decay_start,
                decay_rate=opt.decay_rate, decay_type=opt.decay_type, epsilon=opt.epsilon,
                convergence_prev=opt.convergence_prev, early_stopping=opt.early_stopping,
                non_contact_indices=self.non_contact_indices,
            )
        }

        #initialize objective function
        self.f = OBJ_FUNC[opt.objfun](opt)

        #initialise optimizer
        self.alg = ALGORITHMS[opt.objfun](opt)

        print("\nWill optimize {0} {1} variables wrt {2} \nand {3}".format(
            self.f.x.size, self.f.x.dtype, self.f, self.regularization))
        print("Optimizer: {0}".format(self.alg))

        if plotfile:
            print("The optimization log file will be written to {0}".format(plotfile))

        start=datetime.datetime.now()
        self.fx, self.x, self.algret    = self.alg.minimize(self.f, self.f.x)
        self.algret['runtime']=(datetime.datetime.now() - start).total_seconds() / 60

        self.x_single, self.x_pair      = self.f.finalize(self.x)

        condition = "Finished" if self.algret['code'] >= 0 else "Exited"
        print("\n{0} with code {code} -- {message}\n".format(condition, **self.algret))

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

        if frob:

            print("\nCompute contact map using frobenius norm of couplings.\n")

            self.mats["frobenius"] = {
                'mat':    io.contactmatrix.frobenius_score(self.x_pair),
                'mat_file': self.mat_file,
                'score': "frobenius",
                'correction': "no correction"
            }

    def compute_correction(self, apc_file=None, entropy_correction_file=None):

        """
        Compute bias correction for raw contact maps

        :param apc:     apply average product correction
        :param entropy_correction: apply entropy correction
        :param joint_entropy: use joint entropy instead of geometric mean of marginal entropies
        :param sergeys_jec: scale couplings with a joint entropy correction
        :return:
        """


        #iterate over all raw contact matrices
        for score_mat in list(self.mats.keys()):

            mat_dict = self.mats[score_mat]
            score = mat_dict['score']
            score_matrix = mat_dict['mat']

            if apc_file is not None:
                self.mats[score_mat + "-apc"]={
                    'mat': io.contactmatrix.apc(score_matrix),
                    'mat_file': apc_file,
                    'score': score,
                    'correction': "apc"
                    }

            if entropy_correction_file is not None and score == "frobenius":
                nr_states = 20
                log = np.log2

                # use amino acid frequencies including gap states and with pseudo-counts
                single_freq = self.pseudocounts.freqs[0]

                scaling_factor, mat_corrected = io.contactmatrix.compute_local_correction(
                    single_freq, self.x_pair, self.neff, self.regularization.lambda_pair,
                    squared=False, entropy=True, nr_states=nr_states, log=log
                )

                self.mats[score_mat+"-ec."+ str(nr_states) + "." + str(log.__name__)] = {
                    'mat': mat_corrected,
                    'mat_file': entropy_correction_file,
                    'score': score,
                    'correction': "entropy_correction",
                    'scaling_factor': scaling_factor,
                    'nr_states': nr_states,
                    'log': log.__name__
                }

    def write_matrix(self):
        """
        Write (corrected) contact maps to text file including meta data
        :return:
        """

        print("\nWriting contact matrices to: ")
        for mat_name, mat_dict in self.mats.items():

            mat = mat_dict['mat']
            if self.max_gap_pos < 100:
                mat = gaps.backinsert_gapped_positions_mat(mat, self.gapped_positions)
                self.L = mat.shape[0]

            meta = self.create_meta_data(mat_name)
            io.contactmatrix.write_matrix(mat_dict['mat_file'], mat, meta)
            print("\t" + mat_dict['mat_file'])

    def write_binary_raw(self, out_binary_raw_file):
        """
        Write single and pair potentials including meta data to msgpack-formatted binary raw file

        :param out_binary_raw_file: path to out file
        :return:
        """

        if self.max_gap_pos < 100:
            self.x_single, self.x_pair = gaps.backinsert_gapped_positions(
                self.x_single, self.x_pair, self.gapped_positions)
            self.L = self.x_single.shape[0]

        self.out_binary_raw_file = out_binary_raw_file
        meta = self.create_meta_data()

        raw_out = raw.CCMRaw(self.L, self.x_single[:, :20], self.x_pair[:, :, :21, :21], meta)
        print("\nWriting msgpack-formatted potentials to {0}".format(out_binary_raw_file))
        raw.write_msgpack(out_binary_raw_file, raw_out)

