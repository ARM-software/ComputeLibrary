/*
 * Copyright (c) 2024 Arm Limited.
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

#include "arm_compute/runtime/CL/functions/CLScatter.h"

#include "arm_compute/function_info/ScatterInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "src/gpu/cl/operators/ClScatter.h"

namespace arm_compute
{
using OperatorType = opencl::ClScatter;

struct CLScatter::Impl
{
    std::unique_ptr<OperatorType> op{nullptr};
    ITensorPack                   run_pack{};
};

CLScatter::CLScatter() : _impl(std::make_unique<Impl>())
{
}

CLScatter::~CLScatter() = default;

void CLScatter::configure(const ICLTensor   *src,
                          const ICLTensor   *updates,
                          const ICLTensor   *indices,
                          ICLTensor         *output,
                          const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    configure(CLKernelLibrary::get().get_compile_context(), src, updates, indices, output, info);
}

void CLScatter::configure(const CLCompileContext &compile_context,
                          const ICLTensor        *src,
                          const ICLTensor        *updates,
                          const ICLTensor        *indices,
                          ICLTensor              *output,
                          const ScatterInfo      &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, indices, output);

    _impl->op = std::make_unique<OperatorType>();
    _impl->op->configure(compile_context, src->info(), updates->info(), indices->info(), output->info(), info);
    _impl->run_pack = {{ACL_SRC_0, src}, {ACL_SRC_1, updates}, {ACL_SRC_2, indices}, {ACL_DST, output}};
}

Status CLScatter::validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *output,
                           const ScatterInfo &info)
{
    return OperatorType::validate(src, updates, indices, output, info);
}

void CLScatter::run()
{
    _impl->op->run(_impl->run_pack);
}

} // namespace arm_compute
