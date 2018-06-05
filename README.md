# CCMgen and CCMpredPy

CCMgen is a Python toolkit for sampling protein-like sequences from a second-order Markov Randon Field model. By using second-order interactions, sampled protein sequences are more realistic than what can be sampled from e.g. a model using only a PSSM representation.

CCMgen is accompanied by CCMpredPy, a fast Python implementation of an evolutionary coupling method for learning a Markov Randon Field Model from the multiple sequence alignment of a protein family by either state-of-the-art pseudo-likelihood maximization (less accurate) or persistent contrastive divergence (recommended for use with CCMgen).
The coupling potentials encoded by the learned Markov Random Field model can be used with CCMgen to generate new sequences. 

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

and can be installed manually with pip by running:

```bash
pip install numpy scipy biopython msgpack-python six plotly colorlover
```

## Download

### Release Versions
Please check out the [GitHub releases page for CCMgen](TODO TODO TODO) to download a stable CCMgen/CCMpredPy release. After you're done downloading and extracting, please follow the installation instructions below.

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


## Next Steps
Now you're ready to use CCMgen and CCMpredPy! You can have a look at the [getting started guide](https://github.com/soedinglab/CCMgen/wiki/Getting-Started-with-CCMgen-and-CCMpredPy) to learn how to use both tools.
