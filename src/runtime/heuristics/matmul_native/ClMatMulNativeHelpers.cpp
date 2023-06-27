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
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeHelpers.h"

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"

#include <limits>
#include <utility>

namespace arm_compute
{
namespace cl_matmul
{
MatMulKernelInfo select_info(const MatMulKernelInfo &info0,
                             const MatMulKernelInfo &info1,
                             unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type, bool rhs_lock_padding)
{
    ARM_COMPUTE_ERROR_ON_MSG(info1.export_rhs_to_cl_image == true, "The fallback MatMul configuration cannot have export_to_cl_image = true");
    ARM_COMPUTE_ERROR_ON_MSG(info0.adj_lhs != info1.adj_lhs, "The MatMul configurations must have the same adj_lhs value");
    ARM_COMPUTE_ERROR_ON_MSG(info0.adj_rhs != info1.adj_rhs, "The MatMul configurations must have the same adj_rhs value");

    const bool adj_lhs = info0.adj_lhs;
    const bool adj_rhs = info0.adj_rhs;

    TensorInfo lhs_info = !adj_lhs ? TensorInfo(TensorShape(k, m, b), 1, data_type) : TensorInfo(TensorShape(m, k, b), 1, data_type);
    TensorInfo rhs_info = !adj_rhs ? TensorInfo(TensorShape(n, k, b), 1, data_type) : TensorInfo(TensorShape(k, n, b), 1, data_type);
    TensorInfo dst_info;

    if(rhs_lock_padding == false)
    {
        if(bool(opencl::kernels::ClMatMulNativeKernel::validate(&lhs_info, &rhs_info, nullptr, &dst_info, info0)))
        {
            return info0;
        }
        else
        {
            return info1;
        }
    }
    else
    {
        return info1;
    }
}

MatMulKernelInfo find_info(const MatMulNativeConfigsMatrix &configs, bool adj_lhs, bool adj_rhs, unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    size_t min_acc = std::numeric_limits<size_t>::max();
    size_t min_idx = 0;

    ARM_COMPUTE_ERROR_ON(configs.size() == 0);
    const size_t num_rows = configs.size();
    const size_t num_cols = configs[0].size();

    ARM_COMPUTE_ERROR_ON_MSG(num_cols != 8U, "The entry should have 8 integer values representing: M, N, K, B, M0, N0. K0, IMG_RHS");
    ARM_COMPUTE_UNUSED(num_cols);

    // Find nearest GeMM workload
    // Note: the workload does not depend on the K dimension
    for(size_t y = 0; y < num_rows; ++y)
    {
        size_t mc0 = static_cast<size_t>(configs[y][0]);
        size_t nc0 = static_cast<size_t>(configs[y][1]);
        size_t kc0 = static_cast<size_t>(configs[y][2]);
        size_t bc0 = static_cast<size_t>(configs[y][3]);

        size_t acc = 0;
        acc += (m - mc0) * (m - mc0);
        acc += (n - nc0) * (n - nc0);
        acc += (k - kc0) * (k - kc0);
        acc += (b - bc0) * (b - bc0);
        acc = std::sqrt(acc);
        if(acc < min_acc)
        {
            min_acc = acc;
            min_idx = y;
        }
    }

    // Get the configuration from the nearest GeMM shape
    MatMulKernelInfo desc;
    desc.adj_lhs                = adj_lhs;
    desc.adj_rhs                = adj_rhs;
    desc.m0                     = configs[min_idx][4];
    desc.n0                     = configs[min_idx][5];
    desc.k0                     = configs[min_idx][6];
    desc.export_rhs_to_cl_image = configs[min_idx][7];

    return desc;
}
} // namespace cl_matmul
} // namespace arm_compute
