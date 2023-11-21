/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLConv3D.h"

#include "arm_compute/core/CL/ICLTensor.h"

#include "src/gpu/cl/operators/ClDirectConv3d.h"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct CLConv3D::Impl
{
    const ICLTensor                        *src{nullptr};
    const ICLTensor                        *weights{nullptr};
    const ICLTensor                        *biases{nullptr};
    ICLTensor                              *dst{nullptr};
    std::unique_ptr<opencl::ClDirectConv3d> op{nullptr};
};

CLConv3D::CLConv3D() : _impl(std::make_unique<Impl>())
{
}

CLConv3D::~CLConv3D() = default;

void CLConv3D::configure(const ICLTensor  *src,
                         const ICLTensor  *weights,
                         const ICLTensor  *biases,
                         ICLTensor        *dst,
                         const Conv3dInfo &conv3d_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), src, weights, biases, dst, conv3d_info);
}

void CLConv3D::configure(const CLCompileContext &compile_context,
                         const ICLTensor        *src,
                         const ICLTensor        *weights,
                         const ICLTensor        *biases,
                         ICLTensor              *dst,
                         const Conv3dInfo       &conv3d_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CLConv3D::validate(
        src->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), dst->info(), conv3d_info));

    _impl->src     = src;
    _impl->weights = weights;
    _impl->biases  = biases;
    _impl->dst     = dst;

    _impl->op = std::make_unique<opencl::ClDirectConv3d>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->weights->info(),
                         _impl->biases ? _impl->biases->info() : nullptr, _impl->dst->info(), conv3d_info);
}

Status CLConv3D::validate(const ITensorInfo *src,
                          const ITensorInfo *weights,
                          const ITensorInfo *biases,
                          const ITensorInfo *dst,
                          const Conv3dInfo  &conv3d_info)
{
    return opencl::ClDirectConv3d::validate(src, weights, biases, dst, conv3d_info);
}

void CLConv3D::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weights);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->biases);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
