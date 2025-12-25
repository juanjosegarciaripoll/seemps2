from setuptools import setup, Extension  # type: ignore
import glob
import numpy as np
import sys

# This flag controls whether we build the library with bounds checks and
# other safety measures. Useful when testing where a code breaks down;
# but bad for production performance
debug_library = False
extra_compile_args = []
from Cython.Build import cythonize  # type: ignore
import Cython.Compiler.Options  # type: ignore

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
    if sys.platform == "linux":
        # We assume GCC or other compilers with compatible command line
        extra_compile_args = ["-O3", "-ffast-math"]
    elif sys.platform == "darwin":  # <-- just change this part
        extra_compile_args = ["-O3", "-ffast-math"]  # <-- macOS  use gcc/clang style.
    else:
        # We assume Microsoft Visual C/C++ compiler
        extra_compile_args = ["/Ox", "/fp:fast"]
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
        include_dirs=[np.get_include()],
        depends=include_files,
    )
    for name, file in zip(extension_names, cython_files)
]
setup(ext_modules=cythonize(extensions, compiler_directives=directives))
