import platform
import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


import numpy as np


__version__ = '0.0.0dev'

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

# this probably won't work for everyone. Works for me, though!
# they'll need gcc 7 installed. Unfortunately, I don't have any idea how
# to detect local c compilers. :/
if 'darwin' in platform.system().lower():
    os.environ["CC"] = "gcc-7"
    os.environ["CXX"] = "gcc-7"

# build cython with `python setup.py build_ext --inplace`

cython_extensions = [
    Extension(
        "enspara.info_theory.libinfo",
        ["enspara/info_theory/libinfo.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        )
    ]

setup(
    name='enspara',
    version=__version__,
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    classifiers=CLASSIFIERS,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(cython_extensions),
)
