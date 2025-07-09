from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import distutils.ccompiler

# this code checks for OS. If OS is OSx then it checks for GCC as default compiler
#if GCC is the default compiler adds -fopenmp to linker and compiler args.
use_openmp = (not 'darwin' in platform.system().lower()) or ('gcc' in distutils.ccompiler.get_default_compiler())

extra_compile_args = [ '-Wno-unreachable-code' ]
extra_link_args = []
define_macros = [
    # Target numpy ABIs >=2.0
    ('NPY_NO_DEPRECATED_API', 'NPY_2_0_API_VERSION')
]

if use_openmp:
    extra_compile_args += [ '-fopenmp' ]
    extra_link_args += [ '-fopenmp' ]

ext_modules = cythonize(
    [
        Extension(
            "enspara.info_theory.libinfo",
            [ "enspara/info_theory/libinfo.pyx" ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
        ),
        Extension(
            "enspara.geometry.libdist",
            [ "enspara/geometry/libdist.pyx" ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
        ),
        Extension(
            "enspara.msm.libmsm",
            [ "enspara/msm/libmsm.pyx" ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
        )
    ],
    compiler_directives={ "language_level" : "3" }
)

setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
