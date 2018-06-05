from setuptools import setup, Extension, find_packages

def ext(name, sources=[], include_dirs=[], library_dirs=[], libraries=[], extra_compile_args=['-g', '-fopenmp', '-std=c99'], extra_link_args=['-g', '-fopenmp']):
    return Extension(name, include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries, sources=sources, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(
    name="ccmgen",
    version="1.0.0",
    description="Residue-residue contact prediction from correlated mutations predicted quickly and precisely",
    license="AGPLv3",
    author="Susann Vorberg, Stefan Seemayer, Johannes Soeding",
    author_email="Susann.Vorberg@gmail.com",
    url="https://github.com/soedinglab/ccmgen",
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
        ),
        ext(
            'ccmpred.sampling.cext.libtreecd',
            include_dirs=['ccmpred/objfun/cd/cext'],
            sources=[
                'ccmpred/objfun/cd/cext/cd.c',
                'ccmpred/objfun/cd/cext/cdutil.c',
                'ccmpred/sampling/cext/treecd.c',
            ]
        ),
    ],
    entry_points={
        'console_scripts': [
            'ccmpred=ccmpred.scripts.run_ccmpred:main',
            'ccmgen=ccmpred.scripts.run_ccmgen:main',
            'ccm_replace_gaps=ccmpred.scripts.replace_gaps:main',
            'ccm_plot=ccmpred.scripts.plot_ccmpred:main',
            'ccm_convert_aln=ccmpred.scripts.convert:main'
        ]
    }
)
