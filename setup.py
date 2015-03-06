from setuptools import setup, Extension, find_packages


setup(
    name="CCMpred",
    version="1.0",
    description="Contact Prediction",
    packages=find_packages(),
    ext_modules=[
        Extension('ccmpred.objfun.pll.cext.libpll', include_dirs=[], library_dirs=[], libraries=[], sources=['ccmpred/objfun/pll/cext/pll.c']),
        Extension('ccmpred.counts.libmsacounts', include_dirs=[], library_dirs=[], libraries=[], sources=['ccmpred/counts/msacounts.c'])
    ],
    scripts=['ccmpred.py']
)
