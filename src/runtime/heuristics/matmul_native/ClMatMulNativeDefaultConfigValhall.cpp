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
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeDefaultConfigValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeHelpers.h"

#include <utility>

namespace arm_compute
{
namespace cl_matmul
{
ClMatMulNativeDefaultConfigValhall::ClMatMulNativeDefaultConfigValhall(GPUTarget gpu) : IClMatMulNativeKernelConfig(gpu)
{
}

MatMulKernelInfo
ClMatMulNativeDefaultConfigValhall::configure(const ITensorInfo *lhs, const ITensorInfo *rhs, const MatMulInfo &info)
{
    using ConfigurationFunctionExecutorPtr = MatMulKernelInfo (ClMatMulNativeDefaultConfigValhall::*)(
        unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info);

    ClMatMulNativeConfigArray<ConfigurationFunctionExecutorPtr> configs_G710(
        &ClMatMulNativeDefaultConfigValhall::configure_G710_f32,
        &ClMatMulNativeDefaultConfigValhall::configure_G710_f16,
        &ClMatMulNativeDefaultConfigValhall::configure_G710_u8);

    ClMatMulNativeConfigArray<ConfigurationFunctionExecutorPtr> configs_G715(
        &ClMatMulNativeDefaultConfigValhall::configure_G715_f32,
        &ClMatMulNativeDefaultConfigValhall::configure_G715_f16,
        &ClMatMulNativeDefaultConfigValhall::configure_G715_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;
    switch (_target)
    {
        case GPUTarget::G715:
        case GPUTarget::G615:
            func = configs_G715.get_function(lhs->data_type());
            break;
        case GPUTarget::G710:
        default:
            func = configs_G710.get_function(lhs->data_type());
            break;
    }

    const bool adj_lhs = info.adj_lhs();
    const bool adj_rhs = info.adj_rhs();

    TensorShape lhs_shape = lhs->tensor_shape();
    TensorShape rhs_shape = rhs->tensor_shape();

    const bool is_batched = lhs_shape.num_dimensions() > 2;

    if (is_batched == true)
    {
        lhs_shape.collapse_from(2);
    }

    const unsigned int m = adj_lhs ? lhs_shape.x() : lhs_shape.y();
    const unsigned int n = adj_rhs ? rhs_shape.y() : rhs_shape.x();
    const unsigned int k = adj_lhs ? lhs_shape.y() : lhs_shape.x();
    const unsigned int b = lhs_shape.z();

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not supported for matmul native");
    return (this->*func)(m, n, k, b, rhs->lock_paddings(), info);
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G715_f32(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    ARM_COMPUTE_UNUSED(m, n, k, b, rhs_lock_padding);
    return {info.adj_lhs(), info.adj_rhs(), /* m0 */ 1, /* n0 */ 4, /* k0 */ 1, /* export_to_cl_image */ false};
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G715_f16(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    return configure_G715_f32(m, n, k, b, rhs_lock_padding, info);
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G715_u8(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    ARM_COMPUTE_UNUSED(m, n, k, b, rhs_lock_padding);
    return {info.adj_lhs(), info.adj_rhs(), /* m0 */ 4, /* n0 */ 16, /* k0 */ 4, /* export_to_cl_image */ false};
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G710_f32(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_nt = {
        {3136, 64, 64, 36, 4, 4, 16, 1}, {4096, 48, 32, 36, 4, 4, 4, 1},   {688, 92, 68, 32, 2, 8, 4, 1},
        {24, 464, 412, 24, 2, 8, 4, 1},  {112, 184, 144, 28, 4, 4, 16, 1}, {5776, 64, 32, 36, 2, 4, 16, 1},
        {1568, 64, 40, 36, 2, 8, 8, 1},  {2920, 64, 64, 24, 4, 4, 16, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_nt_nt = {
        {3136, 64, 64, 36, 4, 4, 8, 0}, {4096, 48, 32, 36, 4, 4, 8, 0},  {688, 92, 68, 32, 5, 4, 4, 0},
        {24, 464, 412, 24, 6, 2, 8, 0}, {112, 184, 144, 28, 6, 4, 4, 0}, {5776, 64, 32, 36, 5, 4, 4, 0},
        {1568, 64, 40, 36, 4, 4, 8, 0}, {2920, 64, 64, 24, 4, 4, 8, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_t = {
        {3136, 64, 64, 36, 4, 4, 4, 1}, {4096, 48, 32, 36, 2, 2, 16, 1},  {688, 92, 68, 32, 4, 4, 4, 1},
        {24, 464, 412, 24, 6, 2, 8, 1}, {112, 184, 144, 28, 4, 2, 16, 1}, {5776, 64, 32, 36, 4, 4, 4, 1},
        {1568, 64, 40, 36, 4, 4, 8, 1}, {2920, 64, 64, 24, 4, 4, 4, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_nt_t = {
        {3136, 64, 64, 36, 5, 4, 4, 0}, {4096, 48, 32, 36, 5, 4, 4, 0},  {688, 92, 68, 32, 5, 4, 4, 0},
        {24, 464, 412, 24, 6, 2, 4, 0}, {112, 184, 144, 28, 5, 4, 4, 0}, {5776, 64, 32, 36, 5, 4, 4, 0},
        {1568, 64, 40, 36, 5, 4, 4, 0}, {2920, 64, 64, 24, 6, 2, 4, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_nt = {
        {3136, 64, 64, 36, 4, 4, 16, 1}, {4096, 48, 32, 36, 4, 4, 4, 1},   {688, 92, 68, 32, 2, 8, 4, 1},
        {24, 464, 412, 24, 2, 8, 4, 1},  {112, 184, 144, 28, 4, 4, 16, 1}, {5776, 64, 32, 36, 2, 8, 8, 1},
        {1568, 64, 40, 36, 4, 4, 8, 1},  {2920, 64, 64, 24, 4, 4, 16, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_t_nt = {
        {3136, 64, 64, 36, 4, 4, 4, 0}, {4096, 48, 32, 36, 4, 4, 4, 0},  {688, 92, 68, 32, 4, 4, 4, 0},
        {24, 464, 412, 24, 4, 4, 4, 0}, {112, 184, 144, 28, 4, 4, 4, 0}, {5776, 64, 32, 36, 4, 4, 8, 0},
        {1568, 64, 40, 36, 4, 4, 4, 0}, {2920, 64, 64, 24, 4, 4, 4, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_t = {
        {3136, 64, 64, 36, 4, 4, 4, 1},  {4096, 48, 32, 36, 4, 4, 4, 1},  {688, 92, 68, 32, 4, 4, 4, 1},
        {24, 464, 412, 24, 2, 2, 16, 1}, {112, 184, 144, 28, 4, 4, 4, 1}, {5776, 64, 32, 36, 4, 4, 4, 1},
        {1568, 64, 40, 36, 4, 4, 4, 1},  {2920, 64, 64, 24, 4, 4, 4, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_t_t = {
        {3136, 64, 64, 36, 4, 4, 4, 0}, {4096, 48, 32, 36, 4, 4, 4, 0},  {688, 92, 68, 32, 4, 4, 4, 0},
        {24, 464, 412, 24, 4, 2, 8, 0}, {112, 184, 144, 28, 4, 4, 4, 0}, {5776, 64, 32, 36, 4, 4, 4, 0},
        {1568, 64, 40, 36, 4, 4, 4, 0}, {2920, 64, 64, 24, 4, 4, 4, 0}};

    const bool adj_lhs = info.adj_lhs();
    const bool adj_rhs = info.adj_rhs();

    const MatMulNativeConfigsMatrix *configs_best_to_use     = nullptr;
    const MatMulNativeConfigsMatrix *configs_fallback_to_use = nullptr;

    if ((adj_lhs == false) && (adj_rhs == false))
    {
        configs_best_to_use     = &configs_mnkb_best_nt_nt;
        configs_fallback_to_use = &configs_mnkb_fallback_nt_nt;
    }
    else if ((adj_lhs == false) && (adj_rhs == true))
    {
        configs_best_to_use     = &configs_mnkb_best_nt_t;
        configs_fallback_to_use = &configs_mnkb_fallback_nt_t;
    }
    else if ((adj_lhs == true) && (adj_rhs == false))
    {
        configs_best_to_use     = &configs_mnkb_best_t_nt;
        configs_fallback_to_use = &configs_mnkb_fallback_t_nt;
    }
    else
    {
        configs_best_to_use     = &configs_mnkb_best_t_t;
        configs_fallback_to_use = &configs_mnkb_fallback_t_t;
    }

    MatMulKernelInfo desc0 = find_info(*configs_best_to_use, adj_lhs, adj_rhs, m, n, k, b);
    MatMulKernelInfo desc1 = find_info(*configs_fallback_to_use, adj_lhs, adj_rhs, m, n, k, b);

    return select_info(desc0, desc1, m, n, k, b, DataType::F32, rhs_lock_padding);
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G710_f16(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_nt = {
        {3136, 64, 64, 36, 4, 4, 16, 1}, {4096, 48, 32, 36, 4, 4, 8, 1},   {688, 92, 68, 32, 4, 4, 16, 1},
        {24, 464, 412, 24, 4, 4, 4, 1},  {112, 184, 144, 28, 4, 4, 16, 1}, {5776, 64, 32, 36, 4, 4, 8, 1},
        {1568, 64, 40, 36, 4, 4, 8, 1},  {2920, 64, 64, 24, 4, 4, 16, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_nt_nt = {
        {3136, 64, 64, 36, 6, 4, 8, 0}, {4096, 48, 32, 36, 6, 4, 8, 0},  {688, 92, 68, 32, 6, 4, 8, 0},
        {24, 464, 412, 24, 4, 4, 8, 0}, {112, 184, 144, 28, 6, 4, 8, 0}, {5776, 64, 32, 36, 6, 4, 8, 0},
        {1568, 64, 40, 36, 6, 4, 8, 0}, {2920, 64, 64, 24, 6, 4, 8, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_t = {
        {3136, 64, 64, 36, 6, 4, 8, 1}, {4096, 48, 32, 36, 6, 4, 8, 1},   {688, 92, 68, 32, 4, 4, 4, 1},
        {24, 464, 412, 24, 6, 2, 4, 1}, {112, 184, 144, 28, 4, 2, 16, 1}, {5776, 64, 32, 36, 6, 4, 8, 1},
        {1568, 64, 40, 36, 6, 4, 8, 1}, {2920, 64, 64, 24, 6, 4, 8, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_nt_t = {
        {3136, 64, 64, 36, 6, 2, 16, 0}, {4096, 48, 32, 36, 5, 4, 8, 0},   {688, 92, 68, 32, 6, 2, 16, 0},
        {24, 464, 412, 24, 6, 2, 16, 0}, {112, 184, 144, 28, 6, 2, 16, 0}, {5776, 64, 32, 36, 5, 4, 8, 0},
        {1568, 64, 40, 36, 5, 4, 8, 0},  {2920, 64, 64, 24, 6, 2, 16, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_nt = {
        {3136, 64, 64, 36, 4, 4, 16, 1}, {4096, 48, 32, 36, 4, 4, 4, 1},  {688, 92, 68, 32, 4, 4, 4, 1},
        {24, 464, 412, 24, 4, 4, 4, 1},  {112, 184, 144, 28, 4, 4, 4, 1}, {5776, 64, 32, 36, 4, 4, 4, 1},
        {1568, 64, 40, 36, 4, 4, 4, 1},  {2920, 64, 64, 24, 4, 4, 4, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_t_nt = {
        {3136, 64, 64, 36, 4, 4, 4, 0}, {4096, 48, 32, 36, 4, 4, 4, 0},  {688, 92, 68, 32, 4, 4, 4, 0},
        {24, 464, 412, 24, 4, 4, 4, 0}, {112, 184, 144, 28, 4, 4, 4, 0}, {5776, 64, 32, 36, 4, 4, 4, 0},
        {1568, 64, 40, 36, 4, 4, 4, 0}, {2920, 64, 64, 24, 4, 4, 4, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_t = {
        {3136, 64, 64, 36, 4, 4, 16, 1}, {4096, 48, 32, 36, 4, 4, 8, 1},   {688, 92, 68, 32, 4, 4, 4, 1},
        {24, 464, 412, 24, 4, 2, 8, 1},  {112, 184, 144, 28, 4, 2, 16, 1}, {5776, 64, 32, 36, 4, 4, 16, 1},
        {1568, 64, 40, 36, 4, 4, 8, 1},  {2920, 64, 64, 24, 4, 4, 16, 1}};

    const MatMulNativeConfigsMatrix configs_mnkb_fallback_t_t = {
        {3136, 64, 64, 36, 4, 4, 8, 0}, {4096, 48, 32, 36, 4, 4, 8, 0},  {688, 92, 68, 32, 4, 4, 8, 0},
        {24, 464, 412, 24, 4, 4, 8, 0}, {112, 184, 144, 28, 4, 4, 8, 0}, {5776, 64, 32, 36, 4, 4, 8, 0},
        {1568, 64, 40, 36, 4, 4, 8, 0}, {2920, 64, 64, 24, 4, 4, 8, 0}};

    const bool adj_lhs = info.adj_lhs();
    const bool adj_rhs = info.adj_rhs();

    const MatMulNativeConfigsMatrix *configs_best_to_use     = nullptr;
    const MatMulNativeConfigsMatrix *configs_fallback_to_use = nullptr;

    if ((adj_lhs == false) && (adj_rhs == false))
    {
        configs_best_to_use     = &configs_mnkb_best_nt_nt;
        configs_fallback_to_use = &configs_mnkb_fallback_nt_nt;
    }
    else if ((adj_lhs == false) && (adj_rhs == true))
    {
        configs_best_to_use     = &configs_mnkb_best_nt_t;
        configs_fallback_to_use = &configs_mnkb_fallback_nt_t;
    }
    else if ((adj_lhs == true) && (adj_rhs == false))
    {
        configs_best_to_use     = &configs_mnkb_best_t_nt;
        configs_fallback_to_use = &configs_mnkb_fallback_t_nt;
    }
    else
    {
        configs_best_to_use     = &configs_mnkb_best_t_t;
        configs_fallback_to_use = &configs_mnkb_fallback_t_t;
    }

    MatMulKernelInfo desc0 = find_info(*configs_best_to_use, adj_lhs, adj_rhs, m, n, k, b);
    MatMulKernelInfo desc1 = find_info(*configs_fallback_to_use, adj_lhs, adj_rhs, m, n, k, b);

    return select_info(desc0, desc1, m, n, k, b, DataType::F16, rhs_lock_padding);
}

MatMulKernelInfo ClMatMulNativeDefaultConfigValhall::configure_G710_u8(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool rhs_lock_padding, const MatMulInfo &info)
{
    ARM_COMPUTE_UNUSED(rhs_lock_padding);

    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_nt = {
        {3136, 64, 64, 36, 6, 4, 4, 0}, {4096, 48, 32, 36, 6, 4, 4, 0},  {688, 92, 68, 32, 2, 8, 4, 0},
        {24, 464, 412, 24, 4, 4, 4, 0}, {112, 184, 144, 28, 6, 4, 4, 0}, {5776, 64, 32, 36, 6, 4, 4, 0},
        {1568, 64, 40, 36, 6, 4, 4, 0}, {2920, 64, 64, 24, 5, 4, 4, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_nt_t = {
        {3136, 64, 64, 36, 4, 4, 16, 0}, {4096, 48, 32, 36, 4, 4, 16, 0},  {688, 92, 68, 32, 4, 4, 16, 0},
        {24, 464, 412, 24, 6, 2, 16, 0}, {112, 184, 144, 28, 4, 4, 16, 0}, {5776, 64, 32, 36, 4, 4, 16, 0},
        {1568, 64, 40, 36, 6, 4, 4, 0},  {2920, 64, 64, 24, 4, 4, 16, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_nt = {
        {3136, 64, 64, 36, 4, 4, 8, 0}, {4096, 48, 32, 36, 4, 4, 8, 0},  {688, 92, 68, 32, 4, 4, 4, 0},
        {24, 464, 412, 24, 4, 4, 4, 0}, {112, 184, 144, 28, 4, 4, 8, 0}, {5776, 64, 32, 36, 4, 4, 8, 0},
        {1568, 64, 40, 36, 4, 4, 8, 0}, {2920, 64, 64, 24, 4, 4, 8, 0}};

    const MatMulNativeConfigsMatrix configs_mnkb_best_t_t = {
        {3136, 64, 64, 36, 4, 2, 16, 0}, {4096, 48, 32, 36, 4, 4, 4, 0},   {688, 92, 68, 32, 4, 4, 8, 0},
        {24, 464, 412, 24, 4, 2, 16, 0}, {112, 184, 144, 28, 4, 2, 16, 0}, {5776, 64, 32, 36, 4, 4, 4, 0},
        {1568, 64, 40, 36, 4, 4, 8, 0},  {2920, 64, 64, 24, 4, 2, 16, 0}};

    const bool adj_lhs = info.adj_lhs();
    const bool adj_rhs = info.adj_rhs();

    if ((adj_lhs == false) && (adj_rhs == false))
    {
        return find_info(configs_mnkb_best_nt_nt, adj_lhs, adj_rhs, m, n, k, b);
    }
    else if ((adj_lhs == false) && (adj_rhs == true))
    {
        return find_info(configs_mnkb_best_nt_t, adj_lhs, adj_rhs, m, n, k, b);
    }
    else if ((adj_lhs == true) && (adj_rhs == false))
    {
        return find_info(configs_mnkb_best_t_nt, adj_lhs, adj_rhs, m, n, k, b);
    }
    else
    {
        return find_info(configs_mnkb_best_t_t, adj_lhs, adj_rhs, m, n, k, b);
    }
}
} // namespace cl_matmul
} // namespace arm_compute
