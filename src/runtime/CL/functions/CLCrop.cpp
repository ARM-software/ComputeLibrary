/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLCrop.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClCrop.h"

#include <utility>

namespace arm_compute
{
struct CLCrop::Impl
{
    const ICLTensor                *src{ nullptr };
    ICLTensor                      *dst{ nullptr };
    std::unique_ptr<opencl::ClCrop> op{ nullptr };
};

CLCrop::CLCrop()
    : _impl(std::make_unique<Impl>())
{
}
CLCrop::CLCrop(CLCrop &&) = default;
CLCrop &CLCrop::operator=(CLCrop &&) = default;
CLCrop::~CLCrop()                    = default;

void CLCrop::configure(const ICLTensor *src, ICLTensor *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value,
                       Window *dst_window)
{
    configure(CLKernelLibrary::get().get_compile_context(), src, dst, start, end, batch_index, extrapolation_value, dst_window);
}

void CLCrop::configure(const CLCompileContext &compile_context, const ICLTensor *src, ICLTensor *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value,
                       Window *dst_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    _impl->src = src;
    _impl->dst = dst;

    _impl->op = std::make_unique<opencl::ClCrop>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), start, end, batch_index, extrapolation_value, dst_window);
}

Status CLCrop::validate(const ITensorInfo *input, const ITensorInfo *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value, Window *dst_window)
{
    return opencl::ClCrop::validate(input, output, start, end, batch_index, extrapolation_value, dst_window);
}

void CLCrop::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
