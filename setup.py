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
            sources=['ccmpred/objfun/pll/cext/pll.c']
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
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
        Extension(
            'ccmpred.counts.libmsacounts',
            include_dirs=[],
            library_dirs=[],
            libraries=[],
            sources=['ccmpred/counts/msacounts.c']
        )
    ],
    scripts=['ccmpred.py']
)
