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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

#---------------------------------------------------------------------
# Compute Kernel Writer Project

project(ComputeKernelWriter
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

message(STATUS "${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_VERSION}")

#---------------------------------------------------------------------
# Options

option(CKW_ENABLE_OPENCL "Enable OpenCL code generation" OFF)
option(CKW_ENABLE_ASSERTS "Enable assertions. Always enabled in Debug builds" OFF)
option(CKW_BUILD_TESTING "Build the Compute Kernel Writer validation test suite" OFF)
option(CKW_BUILD_PROTOTYPE "Build the prototype implementation of kernel writer." OFF)
option(CKW_CCACHE "Use compiler cache for faster recompilation" OFF)

#---------------------------------------------------------------------
# Build configuration

get_property(CKW_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Allow only Release or Debug builds
if(NOT CKW_IS_MULTI_CONFIG) # Single-config generators
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Options: Release (default) or Debug" FORCE)
    endif()
else() # Multi-config generators
    list(REMOVE_ITEM CMAKE_CONFIGURATION_TYPES RelWithDebInfo MinSizeRel)
endif()

# Simplistic CCache setup
if(CKW_CCACHE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
        set(CMAKE_C_COMPILER_LAUNCHER ${CACHE_FOUND})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CACHE_FOUND})
    endif()
endif()

#---------------------------------------------------------------------
# Library targets

set(CKW_CXX_FLAGS
    -Wall
    -Werror
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
    -pedantic
)
set(GNU_WARNINGS
    -Wlogical-op
    -Wstrict-null-sentinel
)
set(CKW_ASSERTS_OPTS
    -fstack-protector-strong
)

add_library(ckw)
target_compile_options(ckw
    PUBLIC
    ${CKW_CXX_FLAGS}
    "$<$<CXX_COMPILER_ID:GNU>:${GNU_WARNINGS}>"
    "$<$<CONFIG:Debug>:${CKW_ASSERTS_OPTS}>"
    "$<$<BOOL:${CKW_ENABLE_ASSERTS}>:${CKW_ASSERTS_OPTS}>"
    # Set CMAKE_CXX_FLAGS last so user can overwrite options
    ${CMAKE_CXX_FLAGS}
    PRIVATE
    # Always optimize for binary size
    $<$<CONFIG:Release>:-Os>
)

target_compile_definitions(ckw PUBLIC
    $<$<CONFIG:Debug>:COMPUTE_KERNEL_WRITER_DEBUG_ENABLED>
    $<$<CONFIG:Debug>:COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED>
    $<$<BOOL:${CKW_ENABLE_ASSERTS}>:COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED>
    $<$<BOOL:${CKW_ENABLE_OPENCL}>:COMPUTE_KERNEL_WRITER_OPENCL_ENABLED>
)

target_sources(ckw PRIVATE
    src/types/ConstantData.cpp
    src/types/DataTypeHelpers.cpp
    src/Error.cpp
    src/Helpers.cpp
    src/ITile.cpp
    src/Kernel.cpp
    src/KernelArgument.cpp
    src/KernelWriter.cpp
    src/Tensor3dMapper.cpp
    src/TensorInfo.cpp
    src/TensorOperand.cpp
    src/TensorSampler.cpp
    src/TensorUtils.cpp
    src/TileInfo.cpp
    src/TileOperand.cpp
    src/TileView.cpp
)

if(CKW_ENABLE_OPENCL)
    target_sources(ckw PRIVATE
        src/cl/CLTensorArgument.cpp
        src/cl/CLTensorComponent.cpp
        src/cl/CLHelpers.cpp
        src/cl/CLTile.cpp
        src/cl/CLKernelWriter.cpp
        src/cl/helpers/CLMemoryOpBufferHelper.cpp
        src/cl/helpers/CLMemoryOpImage2dHelper.cpp
    )
endif()

target_include_directories(ckw
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

#---------------------------------------------------------------------
# Validation tests

if(CKW_BUILD_TESTING)
    add_executable(ckw_validation
        validation/Validation.cpp
    )

    target_link_libraries(ckw_validation PRIVATE ckw)
    target_include_directories(ckw_validation
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}
    )
endif()

#---------------------------------------------------------------------
# Prototype

if(CKW_BUILD_PROTOTYPE)
    add_subdirectory(prototype)
endif()

#---------------------------------------------------------------------
# Installing

install(TARGETS ckw
    CONFIGURATIONS Release
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ckw
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
