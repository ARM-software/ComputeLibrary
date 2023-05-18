/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEHELPERS
#define SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEHELPERS

#include "arm_compute/core/Types.h"

namespace arm_compute
{
// Forward declaration
struct MatMulKernelInfo;

namespace cl_matmul
{
using MatMulNativeConfigsMatrix = std::vector<std::vector<int32_t>>;

/** This function accepts two MatMulKernelInfo objects where only the first can be with cl_image2d support enabled.
 *  The aim of this function is to check whether the first MatMulKernelInfo object is valid. If not, the function will
 *  return the second MatMulKernelInfo object. Otherwise, the first one.
 *
 * @param[in] info0            MatMulKernelInfo with cl_image2d support
 * @param[in] info1            MatMulKernelInfo to fall-back if cl_image2d cannot be used
 * @param[in] m                Number of rows (M) of the LHS matrix
 * @param[in] n                Number of columns (N) in the RHS matrix not reshaped
 * @param[in] k                Number of rows (K) in the RHS matrix not reshaped
 * @param[in] b                Batch size
 * @param[in] data_type        Data type
 * @param[in] rhs_lock_padding Flag used to know whether the RHS paddings are locked
 *
 * @return @ref MatMulKernelInfo
 */
MatMulKernelInfo select_info(const MatMulKernelInfo &info0,
                             const MatMulKernelInfo &info1,
                             unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type, bool rhs_lock_padding);

/** Find the preferred configurations for the MatMul Native kernel using the MatMulNativeConfigsMatrix provided by the user
 *
 * @param[in] configs List of best configurations for a limited number of MatMul shapes
 * @param[in] adj_lhs Adjoint LHS flag value
 * @param[in] adj_rhs Adjoint RHS flag value
 * @param[in] m       Number of rows (M) of the LHS matrix
 * @param[in] n       Number of columns (N) in the RHS matrix not reshaped
 * @param[in] k       Number of rows (K) in the RHS matrix not reshaped
 * @param[in] b       Batch size
 *
 * @return @ref MatMulKernelInfo
 */
MatMulKernelInfo find_info(const MatMulNativeConfigsMatrix &configs, bool adj_lhs, bool adj_rhs, unsigned int m, unsigned int n, unsigned int k, unsigned int b);
} // namespace cl_matmul
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEHELPERS */
