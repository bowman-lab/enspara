from distutils.core import setup
from Cython.Build import cythonize

import numpy as np


__version__ = '0.0.0dev'

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# build cython with `python setup.py build_ext --inplace`

cython_extensions = [
    "enspara/info_theory/libinfo.pyx"
]

setup(
    name='enspara',
    version=__version__,
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    classifiers=CLASSIFIERS.splitlines(),
    include_dirs=[np.get_include()],
    ext_modules=cythonize(cython_extensions),
)
