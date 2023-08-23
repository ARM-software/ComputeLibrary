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
#include "arm_compute/runtime/CL/functions/CLMatMul.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "src/gpu/cl/operators/ClMatMul.h"

namespace arm_compute
{
using OperatorType = opencl::ClMatMul;

struct CLMatMul::Impl
{
    std::unique_ptr<OperatorType> op{ nullptr };
    ITensorPack                   run_pack{};
};
CLMatMul::CLMatMul()
    : _impl(std::make_unique<Impl>())
{
}

CLMatMul::~CLMatMul() = default;

void CLMatMul::configure(ICLTensor *lhs, ICLTensor *rhs, ICLTensor *output, const MatMulInfo &matmul_info, const GpuMatMulSettings &settings, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(settings);
    configure(CLKernelLibrary::get().get_compile_context(), lhs, rhs, output, matmul_info, settings, act_info);
}

void CLMatMul::configure(const CLCompileContext &compile_context, ICLTensor *lhs, ICLTensor *rhs, ICLTensor *output, const MatMulInfo &matmul_info, const GpuMatMulSettings &settings,
                         const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, output);
    ARM_COMPUTE_UNUSED(settings);

    _impl->op = std::make_unique<OperatorType>();
    _impl->op->configure(compile_context, lhs->info(), rhs->info(), output->info(), matmul_info, act_info);
    _impl->run_pack = { { ACL_SRC_0, lhs }, { ACL_SRC_1, rhs }, { ACL_DST, output } };
}

Status CLMatMul::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *output, const MatMulInfo &matmul_info, const ActivationLayerInfo &act_info)
{
    return OperatorType::validate(lhs, rhs, output, matmul_info, act_info);
}

void CLMatMul::run()
{
    _impl->op->run(_impl->run_pack);
}

} // namespace arm_compute
