# Copyright (c) 2023-2025 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(
  ArmCompute
  VERSION 0.0.0
  DESCRIPTION
  "The Compute Library is a collection of low-level machine learning functions \
   optimized for Arm® Cortex®-A, Arm® Neoverse™ CPU and Arm® Mali™ GPU \
   architectures."
  LANGUAGES C CXX ASM
)

# * Component build options.
option(ARM_COMPUTE_BUILD_EXAMPLES "Build example programs." OFF)
option(ARM_COMPUTE_BUILD_SHARED_LIB "Build a shared library." ON)
option(ARM_COMPUTE_BUILD_TESTING "Build tests." OFF)

# * Feature build options.
option(ARM_COMPUTE_ENABLE_ASSERTS "Enable asserts." OFF)
option(ARM_COMPUTE_ENABLE_CPPTHREADS "Enable C++11 threads backend." OFF)
option(ARM_COMPUTE_ENABLE_LOGGING "Enable logging." OFF)
option(ARM_COMPUTE_ENABLE_OPENMP "Enable OpenMP backend." ON)
option(ARM_COMPUTE_ENABLE_WERROR "Enable fatal warnings." OFF)

# * Debugging options.
option(ARM_COMPUTE_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(ARM_COMPUTE_ENABLE_SANITIZERS "Enable sanitizers." OFF)
option(ARM_COMPUTE_USE_LIBCXX "Use libcxx instead of the default stdlib." OFF)

# * Set architecture.
set(ARM_COMPUTE_ARCH armv8-a CACHE STRING "Architecture (march) for core library.")
set(ARM_COMPUTE_CORE_FP16_ARCH armv8.2-a+fp16 CACHE STRING "Architecture (march) for core library that require fp16 support.")
set(ARM_COMPUTE_SVE_ARCH armv8.2-a+sve+fp16+dotprod CACHE STRING "Architecture (march) for sve library.")
set(ARM_COMPUTE_SVE2_ARCH armv8.6-a+sve2+fp16+dotprod CACHE STRING "Architecture (march) for sve2 library.")

# * Set variables.
set(ARM_COMPUTE_C_STANDARD 99 CACHE STRING "C Standard to use for the library.")
set(ARM_COMPUTE_CXX_STANDARD 14 CACHE STRING "CXX Standard to use for the library.")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/configurations.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/compilers/setup.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/version.cmake)

if(ARM_COMPUTE_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Set lib build type accordingly
if(ARM_COMPUTE_BUILD_SHARED_LIB)
  message(STATUS "Will build the Arm Compute Library with shared libraries.")
  set(ARM_COMPUTE_LIB_BUILD_TYPE SHARED)
else()
  message(STATUS "Will build the Arm Compute Library with static libraries.")
  set(ARM_COMPUTE_LIB_BUILD_TYPE STATIC)
endif()

set(
  ARM_COMPUTE_PUBLIC_INCLUDE
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(
  ARM_COMPUTE_COMMON_INCLUDE
  ${ARM_COMPUTE_PUBLIC_INCLUDE}
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src/cpu/kernels/assembly
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/arm_gemm
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/assembly
)

set(
  ARM_COMPUTE_GRAPH_INCLUDE
  ${ARM_COMPUTE_COMMON_INCLUDE}
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/convolution/common
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/arm_conv/depthwise
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/convolution/winograd
)

set(
  ARM_COMPUTE_INCLUDE
  ${ARM_COMPUTE_GRAPH_INCLUDE}
  ${CMAKE_CURRENT_LIST_DIR}/third_party/kleidiai
)

set(
  ARM_COMPUTE_SVE_COMMON_INCLUDE
  ${ARM_COMPUTE_COMMON_INCLUDE}
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/arm_conv
  ${CMAKE_CURRENT_LIST_DIR}/src/core/cpu/kernels/assembly
  ${CMAKE_CURRENT_LIST_DIR}/src/cpu/kernels/assembly
  ${CMAKE_CURRENT_LIST_DIR}/src/core/NEON/kernels/arm_gemm/merges
)

add_library(arm_compute_sve OBJECT)
set_target_properties(
  arm_compute_sve
  PROPERTIES
  COMPILE_OPTIONS "${ARM_COMPUTE_SVE_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
  COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
  INCLUDE_DIRECTORIES "${ARM_COMPUTE_SVE_COMMON_INCLUDE}"
  LINK_LIBRARIES "${ARM_COMPUTE_LINK_LIBS}"
)

add_library(arm_compute_sve2 OBJECT)
set_target_properties(
  arm_compute_sve2
  PROPERTIES
  COMPILE_OPTIONS "${ARM_COMPUTE_SVE2_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
  COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
  INCLUDE_DIRECTORIES "${ARM_COMPUTE_SVE_COMMON_INCLUDE}"
  LINK_LIBRARIES "${ARM_COMPUTE_LINK_LIBS}"
)

add_library(arm_compute_core OBJECT)
set_target_properties(
  arm_compute_core
  PROPERTIES
  COMPILE_OPTIONS "${ARM_COMPUTE_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
  COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
  INCLUDE_DIRECTORIES "${ARM_COMPUTE_INCLUDE}"
  LINK_LIBRARIES "${ARM_COMPUTE_LINK_LIBS}"
)

add_library(arm_compute_core_fp16 OBJECT)
set_target_properties(
  arm_compute_core_fp16
  PROPERTIES
  COMPILE_OPTIONS "${ARM_COMPUTE_CORE_FP16_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
  COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
  INCLUDE_DIRECTORIES "${ARM_COMPUTE_INCLUDE}"
  LINK_LIBRARIES "${ARM_COMPUTE_LINK_LIBS}"
)

add_library(arm_compute_graph ${ARM_COMPUTE_LIB_BUILD_TYPE})
set_target_properties(
    arm_compute_graph
    PROPERTIES
        COMPILE_OPTIONS "${ARM_COMPUTE_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
        COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
        INCLUDE_DIRECTORIES "${ARM_COMPUTE_GRAPH_INCLUDE}"
        LINK_LIBRARIES "${ARM_COMPUTE_LINK_LIBS}$<IF:$<PLATFORM_ID:Darwin>,;arm_compute,>"
)

add_library(
  arm_compute
  ${ARM_COMPUTE_LIB_BUILD_TYPE}
  $<TARGET_OBJECTS:arm_compute_core>
  $<TARGET_OBJECTS:arm_compute_core_fp16>
  $<TARGET_OBJECTS:arm_compute_sve>
  $<TARGET_OBJECTS:arm_compute_sve2>
)

# Linking to arm_compute[_graph] should automatically bring includes and dependent libs with it
foreach(TARGET IN ITEMS arm_compute arm_compute_graph)
  target_link_libraries(${TARGET} PUBLIC ${ARM_COMPUTE_LINK_LIBS})
  target_include_directories(${TARGET} PUBLIC ${ARM_COMPUTE_PUBLIC_INCLUDE})
  set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

list(APPEND ARM_COMPUTE_TARGETS arm_compute arm_compute_graph arm_compute_core arm_compute_core_fp16 arm_compute_sve arm_compute_sve2)

# Library target sources.
add_subdirectory(src)

if(ARM_COMPUTE_BUILD_TESTING)
  add_library(arm_compute_validation_framework OBJECT)
  add_subdirectory(tests)
  add_executable(arm_compute_validation $<TARGET_OBJECTS:arm_compute_validation_framework>)
  add_subdirectory(tests/validation)
  add_executable(arm_compute_benchmark $<TARGET_OBJECTS:arm_compute_validation_framework>)
  add_subdirectory(tests/benchmark)

  set_target_properties(
    arm_compute_benchmark arm_compute_validation_framework arm_compute_validation
    PROPERTIES
    COMPILE_OPTIONS "${ARM_COMPUTE_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
    INCLUDE_DIRECTORIES "${ARM_COMPUTE_PUBLIC_INCLUDE};${ARM_COMPUTE_COMMON_INCLUDE}"
    COMPILE_DEFINITIONS "${ARM_COMPUTE_DEFINES}"
    LINK_LIBRARIES "arm_compute;arm_compute_graph"
  )
  list(APPEND ARM_COMPUTE_TARGETS arm_compute_validation arm_compute_benchmark)
endif(ARM_COMPUTE_BUILD_TESTING)

# Examples Binaries
if(ARM_COMPUTE_BUILD_EXAMPLES)
  add_subdirectory(examples)

  # Graph Examples
  foreach(test_name ${EXAMPLE_GRAPH_NAMES})
    add_executable(
      ${test_name}
      "examples/${test_name}.cpp"
      utils/Utils.cpp
      utils/GraphUtils.cpp
      utils/CommonGraphOptions.cpp
    )
  endforeach()

  # NEON Examples
  foreach(test_name ${EXAMPLE_NEON_NAMES})
    add_executable(
      ${test_name}
      "examples/${test_name}.cpp"
      utils/Utils.cpp
    )
  endforeach()

  # Set common properties
  set_target_properties(
    ${EXAMPLE_GRAPH_NAMES} ${EXAMPLE_NEON_NAMES}
    PROPERTIES
    COMPILE_OPTIONS "${ARM_COMPUTE_ARCH};${ARM_COMPUTE_COMMON_CCXX_FLAGS}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples"
  )

  # Now set the linking properties which differ between the NEON and Graph examples
  set_target_properties(${EXAMPLE_GRAPH_NAMES} PROPERTIES LINK_LIBRARIES "arm_compute;arm_compute_graph")
  set_target_properties(${EXAMPLE_NEON_NAMES} PROPERTIES LINK_LIBRARIES arm_compute)
endif(ARM_COMPUTE_BUILD_EXAMPLES)

set_target_properties(
  ${ARM_COMPUTE_TARGETS} ${EXAMPLE_NEON_NAMES} ${EXAMPLE_GRAPH_NAMES}
  PROPERTIES
  COMPILE_WARNING_AS_ERROR ${ARM_COMPUTE_ENABLE_WERROR}
  C_STANDARD ${ARM_COMPUTE_C_STANDARD}
  C_STANDARD_REQUIRED ON
  C_CLANG_TIDY "${ARM_COMPUTE_CLANG_TIDY}"
  C_EXTENSIONS OFF
  C_VISIBILITY_PRESET hidden
  CXX_CLANG_TIDY "${ARM_COMPUTE_CLANG_TIDY}"
  CXX_EXTENSIONS OFF
  CXX_SCAN_FOR_MODULES OFF
  CXX_STANDARD ${ARM_COMPUTE_CXX_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET default
  LINK_OPTIONS "${ARM_COMPUTE_LINKER_FLAGS}"
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN ON
)

# Install libraries
include(GNUInstallDirs)
install(
  TARGETS
  ${ARM_COMPUTE_TARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
