/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/ICLKernel.h"
#include "src/runtime/gpu/cl/operators/ClPooling.h"

namespace arm_compute
{
struct CLPoolingLayer::Impl
{
    const ICLTensor                   *src{ nullptr };
    ICLTensor                         *dst{ nullptr };
    ICLTensor                         *indices{ nullptr };
    std::unique_ptr<opencl::ClPooling> op{ nullptr };
};

CLPoolingLayer::CLPoolingLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLPoolingLayer::~CLPoolingLayer() = default;

void CLPoolingLayer::configure(ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, pool_info, indices);
}

void CLPoolingLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices)
{
    _impl->src     = input;
    _impl->dst     = output;
    _impl->indices = indices;

    _impl->op = std::make_unique<opencl::ClPooling>();
    _impl->op->configure(compile_context, input->info(), output->info(), pool_info, (indices) ? indices->info() : nullptr);
}

Status CLPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    return opencl::ClPooling::validate(input, output, pool_info, indices);
}

void CLPoolingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);
    pack.add_tensor(TensorType::ACL_DST_1, _impl->indices);
    _impl->op->run(pack);
}
} // namespace arm_compute
