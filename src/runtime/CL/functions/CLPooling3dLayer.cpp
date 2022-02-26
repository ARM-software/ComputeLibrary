/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPooling3dLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClPool3d.h"

namespace arm_compute
{
struct CLPooling3dLayer::Impl
{
    const ICLTensor                  *src{ nullptr };
    ICLTensor                        *dst{ nullptr };
    ICLTensor                        *indices{ nullptr };
    std::unique_ptr<opencl::ClPool3d> op{ nullptr };
};

CLPooling3dLayer::CLPooling3dLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLPooling3dLayer::~CLPooling3dLayer() = default;

void CLPooling3dLayer::configure(const ICLTensor *input, ICLTensor *output, const Pooling3dLayerInfo &pool_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, pool_info);
}

void CLPooling3dLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Pooling3dLayerInfo &pool_info)
{
    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<opencl::ClPool3d>();
    _impl->op->configure(compile_context, input->info(), output->info(), pool_info);
}

Status CLPooling3dLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const Pooling3dLayerInfo &pool_info)
{
    return opencl::ClPool3d::validate(input, output, pool_info);
}

void CLPooling3dLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
