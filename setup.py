from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import platform
import numpy as np
import sys
import distutils.ccompiler

extra_compile_args = ['-Wno-unreachable-code']
extra_link_args = []
use_openmp = False

if 'darwin' in platform.system().lower():
    if 'gcc' in distutils.ccompiler.get_default_compiler():
        use_openmp = True
else:
    use_openmp = True

if use_openmp:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

extensions = [
    Extension(
        "enspara.info_theory.libinfo",
        ["enspara/info_theory/libinfo.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "enspara.geometry.libdist",
        ["enspara/geometry/libdist.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "enspara.msm.libmsm",
        ["enspara/msm/libmsm.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="enspara",
    version="0.2.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        "Cython>=0.24",
        "numpy>=1.20",
        "tables>=3.2",
        "matplotlib>=1.5.1",
        "mdtraj>=1.7",
        "psutil>=5.2.2",
        "pandas",
        "scikit-learn>=0.23.2",
        "scipy>=0.17",
        "pyyaml"
    ],
    entry_points={
        'console_scripts': ['enspara = enspara.apps.main:main']
    },
)
