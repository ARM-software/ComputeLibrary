# Copyright (c) 2023 Arm Limited.
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

# ---------------------------------------------------------------------
# Project ArmCompute

list(APPEND CMAKE_MESSAGE_CONTEXT ArmCompute)
project(
  ArmCompute
  VERSION 33.0.0
  DESCRIPTION
    "The Arm Compute Library is a collection of low-level machine learning functions optimized for Arm® Cortex®-A CPU and Arm® Mali™ GPU architectures"
  LANGUAGES C CXX ASM)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.cmake)

# Require at least gcc/g++ 11) CMAKE_CXX_COMPILER_VERSION OR
if(CMAKE_C_COMPILER_VERSION VERSION_LESS 10.2 OR CMAKE_CXX_COMPILER_VERSION
                                                 VERSION_LESS 10.2)
  message(
    FATAL_ERROR "gcc and g++ version => 10.2 is required for building project!")
endif()

# ---------------------------------------------------------------------
# Configuration

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -gdwarf-2 -DARM_COMPUTE_ASSERTS_ENABLED")
# Default to Release Build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose build type, available options are: Debug, Release, RelWithDebInfo"
        FORCE)
endif()

# ---------------------------------------------------------------------
# Information

message(STATUS "Arm Compute Library ${PROJECT_VERSION}")

message(VERBOSE "-----------------------------------------------------")
message(VERBOSE "Build information:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(VERBOSE "Host system: ${CMAKE_SYSTEM_NAME}")
message(VERBOSE "Host processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(VERBOSE "Build path: ${CMAKE_CURRENT_BINARY_DIR}")
message(VERBOSE "Enable OpenCL acceleration: ${ENABLE_OPENCL}")
message(VERBOSE "Enable CPU acceleration: ${ENABLE_NEON}")
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(VERBOSE "-----------------------------------------------------")

# ---------------------------------------------------------------------
# Compile options and features

set(COMMON_CXX_FLAGS
    -Wall
    -DARCH_ARM
    -Wextra
    -Wdisabled-optimization
    -Wformat=2
    -Winit-self
    -Wstrict-overflow=2
    -Wswitch-default
    -Woverloaded-virtual
    -Wformat-security
    -Wctor-dtor-privacy
    -Wsign-promo
    -Weffc++
    -Wno-overlength-strings
    -Wno-ignored-attributes
    -Wlogical-op
    -Wnoexcept
    -Wstrict-null-sentinel
    -Wno-misleading-indentation
    -O3)

# Disable note popups on compiler ABI changes
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wno-psabi")
endif()

# Compile with -Werror if ARM_COMPUTE_WERROR set
if(ARM_COMPUTE_WERROR)
  add_compile_options("-Werror")
endif()

# Compile with -fno-exceptions flag and define ARM_COMPUTE_EXCEPTIONS_DISABLED
# if ARM_COMPUTE_EXCEPTIONS not set
if(NOT ARM_COMPUTE_EXCEPTIONS)
  add_compile_options("-fno-exceptions")
  add_definitions(-DARM_COMPUTE_EXCEPTIONS_DISABLED)
endif()

# Link OpenMP libraries if ARM_COMPUTE_OPENMP set
if(ARM_COMPUTE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
    add_definitions(-DARM_COMPUTE_OPENMP_SCHEDULER)
  else()
    message(FATAL_ERROR "OPENMP was set but no OpenMP library was found!")
  endif()
endif()

# ---------------------------------------------------------------------
# SVE Library

add_library(arm_compute_sve "")
target_compile_options(arm_compute_sve
                       PRIVATE "-march=armv8.2-a+sve+fp16+dotprod")
target_compile_definitions(arm_compute_sve PRIVATE ARM_COMPUTE_ENABLE_BF16)
target_compile_definitions(arm_compute_sve PRIVATE ENABLE_SVE)
target_compile_definitions(arm_compute_sve PRIVATE ARM_COMPUTE_ENABLE_SVE)
target_include_directories(
  arm_compute_sve
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC src
         src/core/NEON/kernels/arm_conv
         src/core/NEON/kernels/arm_gemm
         src/core/NEON/kernels/assembly
         src/core/cpu/kernels/assembly
         src/cpu/kernels/assembly
         src/core/NEON/kernels/arm_gemm/merges)

# ---------------------------------------------------------------------
# SVE2 Library

add_library(arm_compute_sve2 "")
target_compile_options(arm_compute_sve2
                       PRIVATE "-march=armv8.6-a+sve2+fp16+dotprod")
target_compile_definitions(arm_compute_sve2 PRIVATE ARM_COMPUTE_ENABLE_SVE2)
target_compile_definitions(arm_compute_sve2 PRIVATE ARM_COMPUTE_ENABLE_BF16)
target_compile_definitions(arm_compute_sve2 PRIVATE ENABLE_SVE)
target_compile_definitions(arm_compute_sve2 PRIVATE ARM_COMPUTE_ENABLE_SVE)
target_include_directories(
  arm_compute_sve2
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC src
         src/core/NEON/kernels/arm_conv
         src/core/NEON/kernels/arm_gemm
         src/core/NEON/kernels/assembly
         src/core/cpu/kernels/assembly
         src/cpu/kernels/assembly
         src/core/NEON/kernels/arm_gemm/merges)

# ---------------------------------------------------------------------
# Core Library

add_library(arm_compute "")
target_compile_options(arm_compute PRIVATE "-march=${ARM_COMPUTE_ARCH}")
target_compile_definitions(arm_compute PRIVATE ARM_COMPUTE_ENABLE_BF16)
target_compile_definitions(arm_compute PRIVATE ENABLE_SVE)
target_compile_definitions(arm_compute PRIVATE ARM_COMPUTE_ENABLE_SVE)
target_include_directories(
  arm_compute
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE src
          src/cpu/kernels/assembly
          src/core/NEON/kernels/arm_gemm
          src/core/NEON/kernels/assembly
          src/core/NEON/kernels/convolution/common
          src/core/NEON/kernels/arm_conv/depthwise
          src/core/NEON/kernels/convolution/winograd)
target_compile_options(arm_compute PUBLIC ${COMMON_CXX_FLAGS})

add_library(ArmCompute::Core ALIAS arm_compute)
target_link_libraries(
  arm_compute PUBLIC arm_compute_sve arm_compute_sve2)

# ---------------------------------------------------------------------
# Graph Library

add_library(arm_compute_graph "")
target_compile_options(arm_compute_graph PRIVATE "-march=${ARM_COMPUTE_ARCH}")
target_compile_definitions(arm_compute_graph PRIVATE ENABLE_SVE)
target_compile_definitions(arm_compute_graph PRIVATE ARM_COMPUTE_ENABLE_SVE)
# add_subdirectory(src/graph)

target_include_directories(
  arm_compute_graph
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE src
          src/cpu/kernels/assembly
          src/core/NEON/kernels/arm_gemm
          src/core/NEON/kernels/assembly
          src/core/NEON/kernels/convolution/common
          src/core/NEON/kernels/arm_conv/depthwise
          src/core/NEON/kernels/convolution/winograd)
target_compile_options(arm_compute_graph PUBLIC ${COMMON_CXX_FLAGS})

add_library(ArmCompute::Graph ALIAS arm_compute_graph)

# ---------------------------------------------------------------------
# Library Target Sources
add_subdirectory(src)

if(ARM_COMPUTE_BUILD_TESTING)
  # ---------------------------------------------------------------------
  # Validation Framework Library
  add_library(arm_compute_validation_framework "")
  # target_compile_options(arm_compute_validation_framework PRIVATE
  # "-march=armv8.2-a")
  target_compile_options(arm_compute_validation_framework
                        PRIVATE "-march=${ARM_COMPUTE_ARCH}")

  add_subdirectory(tests)
  target_include_directories(
    arm_compute_validation_framework
    PUBLIC $<INSTALL_INTERFACE:include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(arm_compute_validation_framework
                        PUBLIC ${COMMON_CXX_FLAGS})
                        target_link_libraries(
                          arm_compute_validation_framework
      PUBLIC arm_compute arm_compute_graph)

  # ---------------------------------------------------------------------
  # Validation Binary

  add_executable(arm_compute_validation "")
  target_compile_options(arm_compute_validation PRIVATE "-march=${ARM_COMPUTE_ARCH}")
  if(ARM_COMPUTE_ENABLE_BF16_VALIDATION)
    target_compile_definitions(arm_compute_validation PRIVATE ARM_COMPUTE_ENABLE_BF16)
  endif()
  if(ARM_COMPUTE_ENABLE_SVE_VALIDATION)
    target_compile_definitions(arm_compute_validation PRIVATE ENABLE_SVE)
    target_compile_definitions(arm_compute_validation PRIVATE ARM_COMPUTE_ENABLE_SVE)
  endif()
  add_subdirectory(tests/validation)
  target_compile_options(arm_compute_validation PUBLIC ${COMMON_CXX_FLAGS})
  set_target_properties(
    arm_compute_validation PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                      "${CMAKE_BINARY_DIR}/validation")
  target_link_libraries(
    arm_compute_validation
    PUBLIC arm_compute arm_compute_graph arm_compute_validation_framework
           arm_compute_sve)
  target_link_directories(arm_compute_validation PUBLIC tests)

  # ---------------------------------------------------------------------
  # Benchmark Binary

  add_executable(arm_compute_benchmark)
  target_compile_options(arm_compute_benchmark PRIVATE "-march=${ARM_COMPUTE_ARCH}")

  add_subdirectory(tests/benchmark)
  target_compile_options(arm_compute_benchmark PUBLIC ${COMMON_CXX_FLAGS})
  set_target_properties(
    arm_compute_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     "${CMAKE_BINARY_DIR}/validation")
  target_link_libraries(
    arm_compute_benchmark PUBLIC arm_compute arm_compute_graph
                                 arm_compute_validation_framework)

endif() # ARM_COMPUTE_BUILD_TESTING
# ---------------------------------------------------------------------
# Examples Binaries

if(ARM_COMPUTE_BUILD_EXAMPLES)

  add_subdirectory(examples)

  # Graph Examples
  foreach(test_name ${EXAMPLE_GRAPH_NAMES})
    add_executable(
      ${test_name} "examples/${test_name}.cpp" utils/Utils.cpp
                   utils/GraphUtils.cpp utils/CommonGraphOptions.cpp)
    target_compile_options(${test_name} PRIVATE "-march=${ARM_COMPUTE_ARCH}")
    set_target_properties(
      ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              "${CMAKE_BINARY_DIR}/examples")
    target_link_libraries(${test_name} PUBLIC arm_compute
                                              arm_compute_graph arm_compute_sve)
  endforeach()

  # NEON Examples
  foreach(test_name ${EXAMPLE_NEON_NAMES})
    add_executable(${test_name} "examples/${test_name}.cpp" utils/Utils.cpp)
    target_compile_options(${test_name} PRIVATE "-march=${ARM_COMPUTE_ARCH}")
    set_target_properties(
      ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              "${CMAKE_BINARY_DIR}/examples")
    target_link_libraries(${test_name} PUBLIC arm_compute)
  endforeach()

endif() # ARM_COMPUTE_BUILD_EXAMPLES
