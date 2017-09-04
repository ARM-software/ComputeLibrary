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

add_library(benchmark STATIC IMPORTED)
set_target_properties(benchmark PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../3rdparty/linux/armv7a/libbenchmark.a"
)

add_library(OpenCL SHARED IMPORTED)
set_target_properties(OpenCL PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../build/opencl-1.2-stubs/libOpenCL.so"
    IMPORTED_NO_SONAME 1
)

option(ENABLE_PMU_COUNTER "Compile with PMU counter support")

set(ARM_COMPUTE_TARGETS_TO_MEASURE "all" CACHE STRING "Semicolon-separated list of targets to include in validation.")

set(ARM_COMPUTE_ALL_TARGETS
    NEON
    CL
)

if(ARM_COMPUTE_TARGETS_TO_MEASURE STREQUAL "all")
    set(ARM_COMPUTE_TARGETS_TO_MEASURE ${ARM_COMPUTE_ALL_TARGETS})
endif()

list(REMOVE_DUPLICATES ARM_COMPUTE_TARGETS_TO_MEASURE)

foreach(TARGET ${ARM_COMPUTE_TARGETS_TO_MEASURE})
    list(FIND ARM_COMPUTE_ALL_TARGETS ${TARGET} idx)

    if(${idx} LESS 0)
        message(FATAL_ERROR "The target '${TARGET}' does not exist. It should be one of\n${ARM_COMPUTE_ALL_TARGETS}")
    else()
        add_subdirectory(${TARGET})
    endif()
endforeach()

set(arm_compute_test_benchmark_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Datasets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Instrument.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Profiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceProgramOptions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceProgramOptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceUserConfiguration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceUserConfiguration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WallClockTimer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/WallClockTimer.cpp
)

if(${ENABLE_PMU_COUNTER})
    list(APPEND arm_compute_test_benchmark_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/PMUCounter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PMUCounter.cpp
    )
endif()

add_library(arm_compute_test_benchmark OBJECT
    ${arm_compute_test_benchmark_SOURCE_FILES}
)

add_definitions(${arm_compute_test_benchmark_TARGET_DEFINITIONS})
include_directories(${arm_compute_test_benchmark_TARGET_INCLUDES})

add_executable(arm_compute_benchmark
    $<TARGET_OBJECTS:arm_compute_test_benchmark>
    ${arm_compute_test_benchmark_TARGET_OBJECTS}
    $<TARGET_OBJECTS:tensor_library>
    $<TARGET_OBJECTS:arm_compute_test>
)

target_link_libraries(arm_compute_benchmark
    benchmark
    boost_program_options
    arm_compute
    ${arm_compute_test_benchmark_TARGET_LIBRARIES}
)
