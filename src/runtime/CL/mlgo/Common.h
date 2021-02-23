/*
 * Copyright (c) 2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef SRC_RUNTIME_CL_MLGO_COMMON_H
#define SRC_RUNTIME_CL_MLGO_COMMON_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTypes.h"

namespace arm_compute
{
namespace mlgo
{
/** Types of Heuristic (tree) */
enum class HeuristicType
{
    GEMM_Type,                     /**< About the type of gemm */
    GEMM_Config_Native,            /**< About the gemm config for native kernel */
    GEMM_Config_Reshaped_Only_RHS, /**< About the gemm config for reshaped only rhs kernel */
    GEMM_Config_Reshaped           /**< About the gemm config for reshaped kernel */
};

using GEMMType = CLGEMMKernelType;

/** GEMM Configuration for Native kernel */
struct GEMMConfigNative
{
    unsigned int m0{ 1 }; /**< Number of rows processed by the matrix multiplication */
    unsigned int n0{ 1 }; /**< Number of columns processed by the matrix multiplication */
    unsigned int k0{ 1 }; /**< Number of partial accumulations performed by the matrix multiplication */
};

/** GEMM Configuration for Reshaped Only RHS kernel */
struct GEMMConfigReshapedOnlyRHS
{
    unsigned int m0{ 1 };                  /**< Number of rows processed by the matrix multiplication */
    unsigned int n0{ 1 };                  /**< Number of columns processed by the matrix multiplication */
    unsigned int k0{ 1 };                  /**< Number of partial accumulations performed by the matrix multiplication */
    unsigned int h0{ 1 };                  /**< Number of horizontal blocks of size (k0xn0) stored on the same output row */
    bool         interleave_rhs{ false };  /**< True if the h0 (k0xn0) blocks have to be interleaved in the output row */
    bool         transpose_rhs{ false };   /**< True if the (k0xn0) block has to be transposed before been stored */
    bool         export_cl_image{ false }; /**< True if the reshaped rhs has to be exported to cl_image. n0 must be equal to 4 */
};

/** GEMM Configuration for Reshaped kernel */
struct GEMMConfigReshaped
{
    unsigned int m0{ 1 };                  /**< Number of rows processed by the matrix multiplication */
    unsigned int n0{ 1 };                  /**< Number of columns processed by the matrix multiplication */
    unsigned int k0{ 1 };                  /**< Number of partial accumulations performed by the matrix multiplication */
    unsigned int v0{ 1 };                  /**< Number of vertical blocks of size (m0xk0) stored on the same output row */
    unsigned int h0{ 1 };                  /**< Number of horizontal blocks of size (k0xn0) stored on the same output row */
    bool         interleave_lhs{ false };  /**< True if the v0 (m0xk0) blocks have to be interleaved in the output row */
    bool         interleave_rhs{ false };  /**< True if the h0 (k0xn0) blocks have to be interleaved in the output row */
    bool         transpose_rhs{ false };   /**< True if the (k0xn0) block has to be transposed before been stored */
    bool         export_cl_image{ false }; /**< True if the reshaped rhs has to be exported to cl_image. n0 must be equal to 4 */
};

} // namespace mlgo
} // namespace arm_compute
#endif // SRC_RUNTIME_CL_MLGO_COMMON_H