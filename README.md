# CCMgen and CCMpredPy
[![Build Status](https://dev.azure.com/christianroth0419/christianroth/_apis/build/status/soedinglab.CCMgen?branchName=master)](https://dev.azure.com/christianroth0419/christianroth/_build/latest?definitionId=4&branchName=master) [![Generic badge](https://img.shields.io/badge/DOI-10.1371/journal.pcbi.1006526-blue.svg)](https://doi.org/10.1371/journal.pcbi.1006526)

This repository provides a Python toolkit for learning second-order Markov Random Field (MRF) models from multiple sequence alignments of a protein families and using these models for generating realistic synthetic protein sequences. 

CCMpredPy is a fast implementation of an evolutionary coupling method for learning a Markov Randon Field (MRF) Model for a protein family. The parameters of the MRF can either be ingerred by pseudo-likelihood maximization or with persistent contrastive divergence.
While state-of-the-art pseudo-likelihood models have consistenly been found to work best for the purpose of predicting residue-residue contacts, models learned with persistent contrastive divergence are much more accurate in their fine statistics and are recommended for the use with CCMgen to generate realistic sequence samples.

CCMgen is a tool for sampling protein-like sequences from a second-order Markov Randon Field (MRF) model, such as it can be learned with CCMpredPy. The residues of generated sequences will obey the selection pressures described by the MRF with pairwise statistical couplings between residue positions. Furthermore, CCMgen provides full control over the generation of the synthetic alignment by allowing to specify the evolutionary times and phylogeny along which the sequences are sampled.

## Citation
Vorberg S, Seemayer S, Söding J. Synthetic protein alignments by CCMgen quantify noise in residue-residue contact prediction. PLoS computational biology. 2018 Nov 5;14(11):e1006526.

## License

CCMgen and CCMpredPy are released under the [GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/) license.

## Dependencies

- CCMgen/CCMpredPy was developed and tested with Python 3.6
- There are some C libraries to speed up crucial parts of the calculations that need to be compiled with a C compiler.
  Note: When installing on osx, make sure to use an appropriate gcc compiler and not clang, e.g. by setting `export CC=/usr/local/Cellar/gcc/X.X.X/bin/gcc-X` if gcc was installed via brew.

The following Python packages are required

  * NumPy 
  * SciPy
  * BioPython 
  * MsgPack 
  * six 
  * plotly 
  * colorlover 

## Download

### Release Versions
Please check out the [GitHub releases page for CCMgen](https://github.com/soedinglab/CCMgen/releases/tag/v1.0.0-alpha) to download a stable CCMgen/CCMpredPy release. After you're done downloading and extracting, please follow the installation instructions below.

### Development Versions from Git

To clone the latest development version of CCMgen/CCMpredPy, please use the following command line:

```bash
git clone https://github.com/soedinglab/ccmgen.git
```

## Installation

### From cloned/downloaded repository

CCMgen/CCmpredPy can be installed from the main directory into your local Python environment via `pip`:

```bash
pip install .
```

### Directly from Github Repository
  
Alternatively, you can install the latest development version of CCMgen/CCMpredPy with `pip` directly from this repository:

```bash
pip install git+https://github.com/soedinglab/ccmgen@master
```
and keep updated with:

```bash
pip install git+https://github.com/soedinglab/ccmgen@master --upgrade
```
## Uninstall

The CCMgen/CCmpredPy toolkit can be uninstalled with:

```bash
pip uninstall ccmgen
```



## Next Steps
Now you're ready to use CCMgen and CCMpredPy! You can have a look at the [getting started guide](https://github.com/soedinglab/CCMgen/wiki/Getting-Started-with-CCMgen-and-CCMpredPy) to learn how to use both tools.
