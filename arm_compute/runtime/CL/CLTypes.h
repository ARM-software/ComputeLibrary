/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_RUNTIME_CLTYPES_H
#define ARM_COMPUTE_RUNTIME_CLTYPES_H

namespace arm_compute
{
/** OpenCL GEMM kernel types */
enum class CLGEMMKernelType
{
    /** Native GEMM kernel with fixed block size.
     * @note Temporary variant to keep compatibility with the old implementation.
     * @note This variant will be deprecated in favor of a new and configurable NATIVE variant
     */
    NATIVE_V1,
    /** Native GEMM kernel with configurable block size.*/
    NATIVE,
    /** Reshaped GEMM kernel where both lhs and rhs matrices are reshaped. Fixed block size fixed.
     * @note Temporary variant to keep compatibility with the old implementation.
     * @note This variant will be deprecated in favor of RESHAPED
     */
    RESHAPED_V1,
    /** Reshaped GEMM kernel where both lhs and rhs matrices are reshaped. Configurable reshape and block size */
    RESHAPED,
    /** Reshaped GEMM kernel where only the rhs matrix is reshaped. Configurable reshape and block size */
    RESHAPED_ONLY_RHS
};

/** OpenCL GEMM kernel selection parameters. These information are retrieved to select the GEMM kernel on OpenCL */
struct CLGEMMKernelSelectionParams
{
    unsigned int m{ 0 };                         /**< Number of rows for the lhs matrix. Lhs matrix NOT transposed */
    unsigned int n{ 0 };                         /**< Number of columns for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int k{ 0 };                         /**< Number of rows for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int b{ 0 };                         /**< Batch size */
    bool         is_rhs_constant{ false };       /**< True if the content of the rhs matrix is constant */
    DataType     data_type{ DataType::UNKNOWN }; /**< Data type */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_CLTYPES_H */
