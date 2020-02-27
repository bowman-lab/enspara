import platform
import sys

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
import distutils.ccompiler
__version__ = '0.1.0'

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
    import numpy as np
    import Cython
    if Cython.__version__ < '0.19':
        raise ImportError
    from Cython.Build import cythonize
except ImportError:
    sys.stderr.write('-' * 80)
    sys.stderr.write('\n'.join([
        'Error: building enspara requires numpy and cython>=0.19',
        'Try running the command ``pip install numpy cython`` or'
        '``conda install numpy cython``.',

        'or see http://docs.scipy.org/doc/numpy/user/install.html and'
        'http://cython.org/ for more information.']))

use_openmp = False
def use_openmp():
    use_openmp = True
    #install_requires.append('mpi4py>=2.0.0')

install_requires = [
    'Cython>=0.24',
    'numpy>=1.13',
    'tables>=3.2',
    'matplotlib>=1.5.1',
    'mdtraj>=1.7',
    'psutil>=5.2.2',
    'pandas',
    'scikit-learn>=0.21.0',
    'scipy>=0.17'
]

# this code checks for OS. If OS is OSx then it checks for GCC as default compiler
#if GCC is the default compiler adds -fopenmp to linker and compiler args.
if 'darwin' in platform.system().lower():
    if 'gcc' in  distutils.ccompiler.get_default_compiler():
        use_openmp()
    else:
        use_openmp = False
else:
    use_openmp()

extra_compile_args = ['-Wno-unreachable-code']
extra_link_args = []

if use_openmp:
    extra_compile_args += ['-fopenmp']
    extra_link_args = ['-fopenmp']

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
    ), Extension(
        "enspara.msm.libmsm",
        ["enspara/msm/libmsm.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )]

setup(
    name='enspara',
    packages=find_packages(exclude=["tests"],),
    version=__version__,
    project_urls={
        'Documentation': 'https://enspara.readthedocs.io',
        'Source': 'https://github.com/bowman-lab/enspara',
        'Tracker': 'https://github.com/bowman-lab/enspara/issues',
    },
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    classifiers=CLASSIFIERS,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(cython_extensions),
    python_requires='>=3.5,<3.8',  # cython is broken for 3.7
    entry_points={'console_scripts': ['enspara = enspara.apps.main:main']},
    setup_requires=['Cython>=0.24', 'numpy>=1.13'],
    install_requires=install_requires,
    package_data={'': ['articles.json']},
    include_package_data=True,
    extras_require={
        'dev': [
            'nose',
        ],
        'docs': [
            'Sphinx>=1.6.4',
            'sphinx-rtd-theme>=0.2.4',
            'sphinxcontrib-websupport>=1.0.1',
            'numpydoc>=0.7.0',
        ]
    },
    zip_safe=False
)
