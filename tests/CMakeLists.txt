# Copyright (c) 2017 ARM Limited.
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
cmake_minimum_required (VERSION 3.1)
project (arm_compute_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(boost_unit_test_framework STATIC IMPORTED)
set_target_properties(boost_unit_test_framework PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../3rdparty/linux/armv7a/libboost_unit_test_framework.a"
)

add_library(boost_program_options STATIC IMPORTED)
set_target_properties(boost_program_options PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../3rdparty/linux/armv7a/libboost_program_options.a"
)

add_library(arm_compute SHARED IMPORTED)
set_target_properties(arm_compute PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../build/libarm_compute.so"
)

include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/..")
include_directories("${CMAKE_SOURCE_DIR}/../3rdparty/include/")

# AssetsLibrary
set(tensor_library_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/RawTensor.h
    ${CMAKE_SOURCE_DIR}/RawTensor.cpp
    ${CMAKE_SOURCE_DIR}/TensorCache.h
    ${CMAKE_SOURCE_DIR}/AssetsLibrary.h
    ${CMAKE_SOURCE_DIR}/AssetsLibrary.cpp
)

add_library(tensor_library OBJECT
    ${tensor_library_SOURCE_FILES}
)

set(arm_compute_test_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/BorderModeDataset.h
    ${CMAKE_SOURCE_DIR}/ConvertPolicyDataset.h
    ${CMAKE_SOURCE_DIR}/Globals.h
    ${CMAKE_SOURCE_DIR}/IAccessor.h
    ${CMAKE_SOURCE_DIR}/ImageDatasets.h
    ${CMAKE_SOURCE_DIR}/InterpolationPolicyDataset.h
    ${CMAKE_SOURCE_DIR}/NormalizationTypeDataset.h
    ${CMAKE_SOURCE_DIR}/ProgramOptions.h
    ${CMAKE_SOURCE_DIR}/ProgramOptions.cpp
    ${CMAKE_SOURCE_DIR}/RoundingPolicyDataset.h
    ${CMAKE_SOURCE_DIR}/ShapeDatasets.h
    ${CMAKE_SOURCE_DIR}/TypePrinter.h
    ${CMAKE_SOURCE_DIR}/TypeReader.h
    ${CMAKE_SOURCE_DIR}/UserConfiguration.h
    ${CMAKE_SOURCE_DIR}/UserConfiguration.cpp
    ${CMAKE_SOURCE_DIR}/Utils.h
    ${CMAKE_SOURCE_DIR}/boost_wrapper.h
)

add_library(arm_compute_test OBJECT
    ${arm_compute_test_SOURCE_FILES}
)

add_subdirectory(validation)
add_subdirectory(benchmark)
