#import numpy.distutils.intelccompiler
from setuptools import setup, Extension, find_packages

def ext(name, sources=[], include_dirs=[], library_dirs=[], libraries=[], extra_compile_args=['-g', '-fopenmp', '-std=c99'], extra_link_args=['-g', '-fopenmp']):
    return Extension(name, include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries, sources=sources, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

print(find_packages())

setup(
    name="CCMpredPy",
    version="1.0.0",
    description="Residue-residue contact prediction from correlated mutations predicted quickly and precisely",
    license="AGPLv3",
    author="Stefan Seemayer, Susann Vorberg",
    author_email="Susann.Vorberg@gmail.com",
    url="https://github.com/susannvorberg/CCmpredPy",
    packages=find_packages(),
    install_requires=['msgpack-python', 'numpy', 'plotly', 'scipy', 'pandas', 'biopython', 'colorlover'],
    ext_modules=[
        ext(
            'ccmpred.objfun.pll.cext.libpll',
            sources=['ccmpred/objfun/pll/cext/pll.c']
        ),
        ext(
            'ccmpred.objfun.cd.cext.libcd',
            sources=[
                'ccmpred/objfun/cd/cext/cd.c',
                'ccmpred/objfun/cd/cext/cdutil.c'
            ]
        ),
        ext(
            'ccmpred.counts.libmsacounts',
            sources=['ccmpred/counts/msacounts.c']
        ),        
        ext(
            'ccmpred.gaps.cext.libgaps',
            sources=['ccmpred/gaps/cext/gaps.c'],
            extra_compile_args=['-g','-std=c99'],
            extra_link_args=['-g'],
        ),
        ext(
            'ccmpred.weighting.cext.libweighting',
            sources=['ccmpred/weighting/cext/weighting.c']
        )
    ],
    scripts=['run_ccmpred.py', 'replace_gaps.py', 'plot_contact_map.py']
)
