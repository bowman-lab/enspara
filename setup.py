import platform
import os
import subprocess
import sys

from distutils.core import setup
from distutils.extension import Extension

__version__ = '0.1'

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
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

# protect agaist absent cython/numpy
try:
    import numpy
    import Cython
    if Cython.__version__ < '0.19':
        raise ImportError
    from Cython.Build import cythonize
except ImportError:
    print('-' * 80, file=sys.stderr)
    print('\n'.join([
        'Error: building mdtraj requires numpy and cython>=0.19',
        'Try running the command ``pip install numpy cython`` or'
        '``conda install numpy cython``.',

        'or see http://docs.scipy.org/doc/numpy/user/install.html and'
        'http://cython.org/ for more information.']))

import numpy as np

# this probably won't work for everyone. Works for me, though!
# they'll need gcc 7 installed. Unfortunately, I don't have any idea how
# to detect local c compilers. :/
if 'darwin' in platform.system().lower():
    use_openmp = False
else:
    use_openmp = True

extra_compile_args = ['-Wno-unreachable-code']
extra_link_args = []

if use_openmp:
    extra_compile_args += '-fopenmp'
    extra_link_args = '-fopenmp'

# build cython with `python setup.py build_ext --inplace`

cython_extensions = [
    Extension(
        "enspara.info_theory.libinfo",
        ["enspara/info_theory/libinfo.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ), Extension(
        "enspara.geometry.libdist",
        ["enspara/geometry/libdist.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )]

setup(
    name='enspara',
    packages=['enspara'],
    version=__version__,
    url="https://github.com/bowman-lab/enspara",
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    classifiers=CLASSIFIERS,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(cython_extensions),
    python_requires='>=3.5,<3.7',
    install_requires=[
        'Cython>=0.24',
        'tables>=3.2',
        'mdtraj>=1.7,<1.9',
        'mpi4py',
        'scikit-learn>=0.17.1',
        'scipy>=0.17'
    ],
    extras_require={
        'dev': [
            'nose',
        ]
    },
    zip_safe=False
)
