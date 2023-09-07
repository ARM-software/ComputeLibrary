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

#include "src/gpu/cl/kernels/helpers/MatMulKernelHelpers.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/math/Math.h"

#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
Status validate_matmul_input_shapes(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const MatMulKernelInfo &matmul_kernel_info)
{
    const size_t lhs_k = matmul_kernel_info.adj_lhs ? lhs_shape.y() : lhs_shape.x();
    const size_t rhs_k = matmul_kernel_info.adj_rhs ? rhs_shape.x() : rhs_shape.y();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_k != rhs_k, "K dimension in Lhs and Rhs matrices must match.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_shape.total_size() == 0, "Lhs tensor can't be empty");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_shape.total_size() == 0, "Rhs tensor can't be empty");

    constexpr size_t batch_dim_start = 2;
    for(size_t i = batch_dim_start; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_shape[i] != rhs_shape[i], "Batch dimension broadcasting is not supported");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_for_mmul_kernels(const ITensorInfo *lhs,
                                                                         const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulKernelInfo &matmul_kernel_info,
                                                                         int mmul_m0, int mmul_n0)
{
    ARM_COMPUTE_UNUSED(lhs, rhs);

    const Window win = calculate_max_window(*dst, Steps(1, 1));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window collapsed = win.collapse(win, Window::DimZ);

    // Reconfigure window size, one arm_matrix_multiply call needs 16 threads to finish.
    Window::Dimension x_dimension = collapsed.x();
    Window::Dimension y_dimension = collapsed.y();

    const int m = dst->dimension(1);
    const int n = dst->dimension(0);

    const int m0 = std::min(matmul_kernel_info.m0, m);
    const int n0 = adjust_vec_size(matmul_kernel_info.n0, n);

    // Make M and N multiple of M0 and N0 respectively
    const unsigned int ceil_to_multiple_n_n0 = ceil_to_multiple(n, n0);
    const unsigned int ceil_to_multiple_m_m0 = ceil_to_multiple(m, m0);

    // Divide M and N by M0 and N0 respectively
    const unsigned int n_div_n0 = ceil_to_multiple_n_n0 / n0;
    const unsigned int m_div_m0 = ceil_to_multiple_m_m0 / m0;

    // Make n_div_n0 and m_div_m0 multiple of mmul_n0 and mmul_m0 respectively
    const unsigned int ceil_to_multiple_n_div_n0_mmul_n0 = ceil_to_multiple(n_div_n0, mmul_n0);
    const unsigned int ceil_to_multiple_m_div_m0_mmul_m0 = ceil_to_multiple(m_div_m0, mmul_m0);

    // Ensure x_dimension is multiple of MMUL block size (mmul_m0 * mmul_n0)
    x_dimension.set_end(ceil_to_multiple_n_div_n0_mmul_n0 * mmul_m0);
    y_dimension.set_end(ceil_to_multiple_m_div_m0_mmul_m0 / mmul_m0);

    collapsed.set(Window::DimX, x_dimension);
    collapsed.set(Window::DimY, y_dimension);

    return std::make_pair(Status{}, collapsed);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
