# CCMpredPy

This is the Python implementation of the [CCMpred](https://github.com/soedinglab/CCMpred) predictor of residue-residue contacts.


## Installing

Clone this repository to your own computer using `git clone`. Then, you can compile the C extensions by running:

	python setup.py build_ext --inplace
  
Alternatively, you can install ccmpredpy as a package directly from this repository using pip by running:

	pip install git+https://bitbucket.org/svorberg/ccmpred-new@master

## Example Usage via Command Line

Print available command line options:

	python ccmpred.py -h

Per default (`--ofn-pll`) CCMpredPy maximizes the pseudo-likelihood to obtain couplings. Results differ slightly from the C implementation of [CCMpred](https://github.com/soedinglab/CCMpred) due to the following modifications:
- single potential regularization prior is centered at ML estimate of single potentials v*
- single potentials are initialized at v*
- regularization strength lambda_v = 10 in order to achieve comparable results to C implementation of [CCMpred](https://github.com/soedinglab/CCMpred)
- slight modification in the conjugate gradient optimizer compared to [libconjugrad](https://bitbucket.org/soedinglab/libconjugrad.git) used in [CCMpred](https://github.com/soedinglab/CCMpred)

This command will print the optimization progress to stdout and produce the file ./example/1mkcA00.frobenius.mat:

	python ccmpred.py ./example/1mkcA00.aln ./example/1mkcA00.mat

The opimization progress can be visualized as an interactive plotly graph by additionaly specifying the `--plot_opt_progress flag`. The html file containing the graph is updated during optimization and will be written to ./example/1mkcA00.opt_progress.html.

	python ccmpred.py --plot_opt_progress ./example/1mkcA00.aln ./example/1mkcA00.mat

Bias correction can be switched on by using the flags `--apc` and `--entropy-correction`. Using these two additional flags will generate three contact map files: `./example/1mkcA00.frobnenius.mat`, `./example/1mkcA00.frobnenius.apc.mat` and `./example/1mkcA00.frobnenius.ec.mat`:

	python ccmpred.py --apc --entropy-correction ./example/1mkcA00.aln ./example/1mkcA00.mat

Contact maps can be visualized using the script `plot_contact_map.py`. By specifying a PDB file (numbering of amino acids starting at 1!), the distance matrix is plotted in the lower right triangle. By specifying an alignment file, the percentage of gaps and the entropy are plotted as subplot.

	python plot_contact_map.py --mat-file ./example/1mkcA00.frobenius.mat --alignment-file ./example/1mkcA00.aln --pdb-file ./example/1mkcA00.pdb --plot-out ./example/ --seq-sep 4 --contact-threshold 8 --apc

## License

GNU Affero General Public License, Version 3.0
