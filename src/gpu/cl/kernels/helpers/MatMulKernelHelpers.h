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

#ifndef ACL_SRC_GPU_CL_KERNELS_HELPERS_MATMULKERNELHELPERS_H
#define ACL_SRC_GPU_CL_KERNELS_HELPERS_MATMULKERNELHELPERS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Validate the input shapes of Matmul operation
 *
 * @param[in] lhs_shape          Lhs tensor shape
 * @param[in] rhs_shape          Rhs tensor shape
 * @param[in] matmul_kernel_info Matmul kernel info
 *
 * @return true if the shapes and matmul kernel info matches
 */
Status validate_matmul_input_shapes(const TensorShape      &lhs_shape,
                                    const TensorShape      &rhs_shape,
                                    const MatMulKernelInfo &matmul_kernel_info);

/** Validate and configure window for Matmul MMUL kernels
 *
 * @param[in] lhs                Lhs tensor info
 * @param[in] rhs                Rhs tensor info
 * @param[in] dst                Dst tensor info
 * @param[in] matmul_kernel_info Matmul kernel info
 * @param[in] mmul_m0            Number of rows in the MMUL block
 * @param[in] mmul_n0            Number of columns in the MMUL block
 *
 * @return a pair of Status and Window object
 */
std::pair<Status, Window> validate_and_configure_window_for_mmul_kernels(const ITensorInfo      *lhs,
                                                                         const ITensorInfo      *rhs,
                                                                         const ITensorInfo      *dst,
                                                                         const MatMulKernelInfo &matmul_kernel_info,
                                                                         int                     mmul_m0,
                                                                         int                     mmul_n0);
} // namespace kernels
} // namespace opencl
} // namespace arm_compute

#endif // ACL_SRC_GPU_CL_KERNELS_HELPERS_MATMULKERNELHELPERS_H
