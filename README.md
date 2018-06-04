# CCMgen and CCMpredPy

CCMgen is a Python toolkit for sampling protein-like sequences from a second-order Markov Randon Field model. By using second-order interactions, sampled protein sequences are more realistic than what can be sampled from e.g. a model using only a PSSM representation.

CCMgen is accompanied by CCMpredPy, a fast Python implementation of an evolutionary coupling method for learning a Markov Randon Field Model from the multiple sequence alignment of a protein family by either state-of-the-art pseudo-likelihood maximization (less accurate) or persistent contrastive divergence (recommended for use with CCMgen).
The coupling potentials encoded by the learned Markov Random Field model can be used with CCMgen to generate new sequences. 

## License

CCMgen and CCMpredPy are released under the GNU Affero GPL License, version 3.0 or later.

## Requirements

CCMgen requires Python 3.6 or later and the following Python packages installed on your system:

  * NumPy (`pip install numpy`)
  * SciPy (`pip install scipy`)
  * BioPython (`pip install biopython`)
  * MsgPack (`pip install msgpack-python`)
  * six (`pip install six`)

or, in one command:

```bash
pip install numpy scipy biopython msgpack-python six
```

## Downloading

### Release Versions
Please check out the [GitHub releases page for CCMgen](TODO TODO TODO) to download a stable CCMpred release. After you're done downloading and extracting, please follow the [installation instructions below](#user-content-installation-1)

### Development Versions from Git

To clone CCMgen directly from git, please use the following command line:

```bash
git clone https://github.com/soedinglab/ccmgen.git
```

## Installation

There are some C libraries to speed up crucial parts of the calculations. To compile all C libraries for your system, from the main directory, please run:

```bash
python setup.py build_ext --inplace
```
  
Alternatively, you can install CCMgen and CCMpredPy directly from the repository with pip by running:

```bash
pip install git+https://github.com/susannvorberg/CCmpredPy@master
```
and keep updated with 

```bash
	pip install git+https://github.com/susannvorberg/CCmpredPy@master --upgrade
```
	
Note: When installing on osx, make sure to use an appropriate gcc compiler and not clang, e.g. by setting `export CC=/usr/local/Cellar/gcc/X.X.X/bin/gcc-X` if gcc was installed via brew.

## Next Steps
Now you're ready to use CCMgen and CCMpredPy! You can have a look at the [getting started guide](https://github.com/soedinglab/CCMgen/wiki/getting-started) to learn how to use both tools.
