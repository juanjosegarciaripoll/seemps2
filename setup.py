import warnings
from setuptools import setup, Extension  # type: ignore
import glob
import numpy as np
import sys
import os
from Cython.Build import cythonize  # type: ignore
import Cython.Compiler.Options  # type: ignore

# Note:
# Sort input source files if you glob sources to ensure bit - for - bit
# reproducible builds (https://github.com/pybind/python_example/pull/53)
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile

# This flag controls whether we build the library with bounds checks and
# other safety measures. Useful when testing where a code breaks down;
# but bad for production performance
debug_library = "SEEMPS_DEBUG" in os.environ
extra_compile_args = []
extra_link_args: list[str] = []

# See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# for a deeper explanation of the choices here
# Cython.Compiler.Options.docstring = False
Cython.Compiler.Options.error_on_uninitialized = True
if not debug_library:
    directives = {
        "language_level": "3",  # We assume Python 3 code
        "boundscheck": False,  # Do not check array access
        "wraparound": False,  # a[-1] does not work
        "always_allow_keywords": False,  # Faster calling conventions
        "cdivision": True,  # No exception on zero denominator
        "initializedcheck": False,  # We take care of initializing cdef classes and memory views
        "overflowcheck": False,
        "binding": False,
    }
    if sys.platform in ["linux", "darwin"]:
        # We assume GCC or other compilers with compatible command line
        extra_compile_args = ["-O3", "-ffast-math", "-Wno-unused-function"]
    else:
        # We assume Microsoft Visual C/C++ compiler
        # /we4239 removes a non-conformant behavior, whereby a function
        # argument non-const lvalue reference type can be assigned a temp.
        extra_compile_args = ["/Ox", "/fp:fast", "/we4239"]
else:
    directives = {
        "language_level": "3",  # We assume Python 3 code
        "always_allow_keywords": False,  # Faster calling conventions
        "boundscheck": True,
        "initializedcheck": True,
        "wraparound": False,
    }

# All Cython files with unix pathnames
cython_files = [s.replace("\\", "/") for s in glob.glob("src/**/*.pyx", recursive=True)]
include_files = [
    s.replace("\\", "/") for s in glob.glob("src/**/*.pxi", recursive=True)
]
extension_names = [".".join(f[4:-4].split("/")) for f in cython_files]
print("Extension names: ", extension_names)
print("Cython files:    ", cython_files)
extensions = [
    Extension(
        name,
        [file],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
        depends=include_files,
        language="c++",
    )
    for name, file in zip(extension_names, cython_files)
]

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# Enable ASAN (AddressSanitizer) for debugging memory issues
if "SEEMPS_ASAN" in os.environ:
    if sys.platform in ["linux", "darwin"]:
        asan_flags = ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"]
        extra_compile_args.extend(asan_flags)
        extra_link_args.extend(["-fsanitize=address"])
    else:
        warnings.warn("SEEMPS_ASAN not supported on Windows")

pybind11_modules = [
    Pybind11Extension(
        "seemps.cython.pybind",
        [
            "src/seemps/cython/pybind/schmidt.cc",
            "src/seemps/cython/pybind/blas.cc",
            "src/seemps/cython/pybind/svd.cc",
            "src/seemps/cython/pybind/tensors.cc",
            "src/seemps/cython/pybind/contractions.cc",
            "src/seemps/cython/pybind/strategy.cc",
            "src/seemps/cython/pybind/environments.cc",
            "src/seemps/cython/pybind/core.cc",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        cxx_std=17,
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives=directives) + pybind11_modules
)
