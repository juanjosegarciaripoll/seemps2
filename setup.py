from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import glob
import numpy as np
import sys

# The main interface is through Pybind11Extension.
# * You can add cxx_std = 11 / 14 / 17, and then build_ext can be removed.
# * You can set include_pybind11 = false to add the include directory yourself,
# say from a submodule.
#
# Note:
# Sort input source files if you glob sources to ensure bit - for - bit
# reproducible builds(https: // github.com/pybind/python_example/pull/53)
from pybind11.setup_helpers import Pybind11Extension

if sys.platform == "linux":
    # We assume GCC or other compilers with compatible command line
    extra_compile_args = ["-O3", "-ffast-math", "-Wno-sign-compare"]
else:
    # We assume Microsoft Visual C / C++ compiler
    extra_compile_args = ["/Ox", "/fp:fast"]
extra_dependencies = [
    s.replace("\\", "/") for s in glob.glob("src/**/*.h", recursive=True)
] + [s.replace("\\", "/") for s in glob.glob("src/**/*.cc", recursive=True)]
pybind11_modules = [
    Pybind11Extension(
        "seemps.state.core",
        [
            "src/seemps/state/tools.cc",
            "src/seemps/state/blas.cc",
            "src/seemps/state/svd.cc",
            "src/seemps/state/tensors.cc",
            "src/seemps/state/mps.cc",
            "src/seemps/state/mpssum.cc",
            "src/seemps/state/canonical.cc",
            "src/seemps/state/mps_algebra.cc",
            "src/seemps/state/schmidt.cc",
            "src/seemps/state/contractions.cc",
            "src/seemps/state/strategy.cc",
            "src/seemps/state/environments.cc",
            "src/seemps/state/core.cc",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        depends=extra_dependencies,
        cxx_std=17,
    ),
]

setup(ext_modules=pybind11_modules)
