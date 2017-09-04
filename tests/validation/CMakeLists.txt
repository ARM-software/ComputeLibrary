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

add_library(openvx SHARED IMPORTED)
set_target_properties(openvx PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../3rdparty/linux/armv7a/libopenvx.so"
)

add_library(vxu SHARED IMPORTED)
set_target_properties(vxu PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../3rdparty/linux/armv7a/libvxu.so"
)

add_library(OpenCL SHARED IMPORTED)
set_target_properties(OpenCL PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../build/opencl-1.2-stubs/libOpenCL.so"
    IMPORTED_NO_SONAME 1
)

add_definitions(-DBOOST)

set(ARM_COMPUTE_TARGETS_TO_VALIDATE "all" CACHE STRING "Semicolon-separated list of targets to include in validation.")

set(ARM_COMPUTE_ALL_TARGETS
    NEON
    CL
    UNIT
    VX
)

if(ARM_COMPUTE_TARGETS_TO_VALIDATE STREQUAL "all")
    set(ARM_COMPUTE_TARGETS_TO_VALIDATE ${ARM_COMPUTE_ALL_TARGETS})
endif()

list(REMOVE_DUPLICATES ARM_COMPUTE_TARGETS_TO_VALIDATE)

foreach(TARGET ${ARM_COMPUTE_TARGETS_TO_VALIDATE})
    list(FIND ARM_COMPUTE_ALL_TARGETS ${TARGET} idx)

    if(${idx} LESS 0)
        message(FATAL_ERROR "The target '${TARGET}' does not exist. It should be one of\n${ARM_COMPUTE_ALL_TARGETS}")
    else()
        add_subdirectory(${TARGET})
    endif()
endforeach()

set(arm_compute_test_validation_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Datasets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Reference.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Reference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ReferenceCPP.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ReferenceCPP.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ValidationProgramOptions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ValidationUserConfiguration.h
)

add_library(arm_compute_test_validation OBJECT
    ${arm_compute_test_validation_SOURCE_FILES}
)

add_executable(arm_compute_validation
    $<TARGET_OBJECTS:arm_compute_test_validation>
    ${arm_compute_test_validation_TARGET_OBJECTS}
    $<TARGET_OBJECTS:tensor_library>
    $<TARGET_OBJECTS:arm_compute_test>
)

target_link_libraries(arm_compute_validation
    boost_unit_test_framework
    boost_program_options
    arm_compute
    ${arm_compute_test_validation_TARGET_LIBRARIES}
)
