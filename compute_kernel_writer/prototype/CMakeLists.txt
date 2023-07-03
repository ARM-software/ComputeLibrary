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
# Prototype

add_library(ckw_prototype
    src/TileInfo.cpp
    src/TensorInfo.cpp
    src/Kernel.cpp
    src/KernelWriter.cpp
    src/OperandBase.cpp
    src/TileOperand.cpp
    src/TensorOperand.cpp
    src/TensorTileSampler.cpp
)

target_compile_options(ckw_prototype
    PUBLIC
    ${CKW_CXX_FLAGS}
    "$<$<CXX_COMPILER_ID:GNU>:${GNU_WARNINGS}>"
    "$<$<CONFIG:Debug>:${CKW_ASSERTS_OPTS}>"
    "$<$<BOOL:${CKW_ASSERTS}>:${CKW_ASSERTS_OPTS}>"
    ${CMAKE_CXX_FLAGS}
)

target_compile_definitions(ckw_prototype PUBLIC
    $<$<CONFIG:Debug>:COMPUTE_KERNEL_WRITER_DEBUG_ENABLED>
    $<$<CONFIG:Debug>:COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED>
    $<$<BOOL:${CKW_ASSERTS}>:COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED>
    $<$<BOOL:${CKW_ENABLE_OPENCL}>:COMPUTE_KERNEL_WRITER_OPENCL_ENABLED>
)

target_include_directories(ckw_prototype
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

#---------------------------------------------------------------------
# Examples

add_library(ckw_prototype_examples_common
    examples/common/ExampleKernelWriter.cpp
    examples/common/ExampleScopedKernelWriter.cpp
    examples/common/ExampleComponentArgument.cpp
)

target_link_libraries(ckw_prototype_examples_common PUBLIC ckw_prototype)

add_executable(ckw_prototype_examples_add_exp_store examples/add_exp_store.cpp)
target_link_libraries(ckw_prototype_examples_add_exp_store PUBLIC ckw_prototype_examples_common)
