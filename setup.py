from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

# build cython with `python setup.py build_ext --inplace`

setup(
  name='Statistical Trajectory Analysis and Guidance',
  ext_modules=cythonize("enspara/geometry/_euclidean_distance.pyx"),
  include_dirs=[np.get_include()]
)
