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
#include "src/gpu/cl/operators/ClMatMul.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::opencl::kernels;
ClMatMul::ClMatMul()
    : _native_matmul_kernel(std::make_unique<ClMatMulNativeKernel>())
{
}
ClMatMul::~ClMatMul()
{
}
Status ClMatMul::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *output, const MatMulInfo &matmul_info)
{
    MatMulKernelInfo kernel_info;
    kernel_info.adj_lhs = matmul_info.adj_lhs();
    kernel_info.adj_rhs = matmul_info.adj_rhs();
    return ClMatMulNativeKernel::validate(lhs, rhs, output, kernel_info);
}
void ClMatMul::configure(const CLCompileContext &compile_context, ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *output, const MatMulInfo &matmul_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, output);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, output, matmul_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(lhs, rhs, output, matmul_info));
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Placeholder: Getting the heuristics calculated values for M0, N0, K0, and whether to export RHS to texture pipe

    // Filling the MatMul Kernel info
    MatMulKernelInfo kernel_info;
    kernel_info.adj_lhs                = matmul_info.adj_lhs();
    kernel_info.adj_rhs                = matmul_info.adj_rhs();
    kernel_info.m0                     = 1;     // to be properly calculated from heuristics
    kernel_info.n0                     = 4;     // to be properly calculated from heuristics
    kernel_info.k0                     = 4;     // to be properly calculated from heuristics
    kernel_info.export_rhs_to_cl_image = false; // to be properly determined from heuristics

    // Set the target for the kernels
    _native_matmul_kernel->set_target(gpu_target);

    // Configure the native matrix multiply kernel
    _native_matmul_kernel->configure(compile_context, lhs, rhs, output, kernel_info);
}
void ClMatMul::run(ITensorPack &tensors)
{
    CLScheduler::get().enqueue_op(*_native_matmul_kernel, tensors, true);
}
} // namespace opencl
} // namespace arm_compute
