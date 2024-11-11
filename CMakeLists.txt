# Set the minimum CMake version and policies for highest tested version
cmake_minimum_required(VERSION 3.15...3.27)

# Set up the project and ensure there is a working C++ compiler
project(seemps LANGUAGES CXX)

# Whether to use ASAN
option(USE_ASAN "Use Address Sanitizer during build" FALSE)
option(ASAN_TYPE "Address Sanitizer type" "address")
include(CheckCXXCompilerFlag)
if (USE_ASAN)
  set(ASAN_TYPE "$ENV{SANITIZE}")
  if (NOT ASAN_TYPE STREQUAL "address")
	message(FATAL_ERROR "Unknown value of SANITIZE environment variable: ${ASAN_TYPE}")
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message("Looking for -fsanitize=${ASAN_TYPE}")
    set(CMAKE_REQUIRED_FLAGS "-Werror -fsanitize=${ASAN_TYPE}")
    check_cxx_compiler_flag(-fsanitize=${ASAN_TYPE} HAVE_FLAG_SANITIZER)
    if (HAVE_FLAG_SANITIZER)
        message("Adding -fsanitize=${ASAN_TYPE}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${ASAN_TYPE} -fno-omit-frame-pointer")
        set(DCMAKE_C_FLAGS "${DCMAKE_C_FLAGS} -fsanitize=${ASAN_TYPE} -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=${ASAN_TYPE}")
        set(DCMAKE_MODULE_LINKER_FLAGS "${DCMAKE_MODULE_LINKER_FLAGS} -fsanitize=${ASAN_TYPE}")
    else ()
        message("-fsanitize=${flag} unavailable")
    endif ()
  endif ()
endif ()

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif ()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module NumPy
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)


# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  core

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  # STABLE_ABI

  # Source code goes here
  src/seemps/state/blas.cc
  src/seemps/state/canonical.cc
  src/seemps/state/contractions.cc
  src/seemps/state/core.cc
  src/seemps/state/environments.cc
  src/seemps/state/mps_algebra.cc
  src/seemps/state/mps.cc
  src/seemps/state/mpssum.cc
  src/seemps/state/schmidt.cc
  src/seemps/state/strategy.cc
  src/seemps/state/svd.cc
  src/seemps/state/tensors.cc
  src/seemps/state/tools.cc
)


# We need to add the Numpy headers to this project
target_include_directories(core PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Silence Numpy warnings
target_compile_definitions(core PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# Install directive for scikit-build-core
install(TARGETS core LIBRARY DESTINATION seemps/state)
