import numpy.distutils.intelccompiler
from setuptools import setup, Extension, find_packages

setup(
    name="CCMpred",
    version="1.0",
    description="Contact Prediction",
    packages=find_packages(),
    ext_modules=[
        Extension(
            'ccmpred.objfun.pll.cext.libpll',
            include_dirs=[],
            library_dirs=[],
            libraries=[],
            sources=['ccmpred/objfun/pll/cext/pll.c'],
            extra_compile_args=['-g -fopenmp -std=c99'],
            extra_link_args=['-g -fopenmp'],
        ),
        Extension(
            'ccmpred.objfun.cd.cext.libcd',
            include_dirs=[],
            library_dirs=[],
            libraries=[],
            sources=[
                'ccmpred/objfun/cd/cext/cd.c',
                'ccmpred/objfun/cd/cext/cdutil.c'
            ],
            extra_compile_args=['-g -fopenmp -std=c99'],
            extra_link_args=['-g -fopenmp'],
        ),
        Extension(
            'ccmpred.objfun.treecd.cext.libtreecd',
            include_dirs=['ccmpred/objfun/cd/cext'],
            library_dirs=[],
            libraries=[],
            sources=[
                'ccmpred/objfun/cd/cext/cd.c',
                'ccmpred/objfun/cd/cext/cdutil.c',
                'ccmpred/objfun/treecd/cext/treecd.c',
            ],
            extra_compile_args=['-g -fopenmp -std=c99'],
            extra_link_args=['-g -fopenmp'],
        ),
        Extension(
            'ccmpred.counts.libmsacounts',
            include_dirs=[],
            library_dirs=[],
            libraries=[],
            sources=['ccmpred/counts/msacounts.c'],
            extra_compile_args=['-g -fopenmp -std=c99'],
            extra_link_args=['-g -fopenmp'],
        )
    ],
    scripts=['ccmpred.py']
)
