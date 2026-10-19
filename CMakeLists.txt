cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The kernels do not depend on torch: Python passes raw data pointers, so the
# wheel works with any torch version.
nanobind_add_module(_C STABLE_ABI NB_STATIC
  csrc/bindings.cpp
  csrc/selective_scan.cpp
)
target_include_directories(_C PRIVATE csrc)
target_link_libraries(_C PRIVATE Threads::Threads)

# Wheels must run on any CPU of the target arch, so no -march=native here.
if(MSVC)
  target_compile_options(_C PRIVATE /O2 /fp:precise)
else()
  target_compile_options(_C PRIVATE -O3)
endif()

install(TARGETS _C LIBRARY DESTINATION mamba_cpu)
