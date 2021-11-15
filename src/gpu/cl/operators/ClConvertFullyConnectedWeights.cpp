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
#include "src/gpu/cl/operators/ClConvertFullyConnectedWeights.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClConvertFullyConnectedWeightsKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace opencl
{
void ClConvertFullyConnectedWeights::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const TensorShape &original_src_shape, DataLayout data_layout)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, original_src_shape, data_layout);
    auto k = std::make_unique<kernels::ClConvertFullyConnectedWeightsKernel>();
    k->configure(compile_context, src, dst, original_src_shape, data_layout);
    _kernel = std::move(k);
}

Status ClConvertFullyConnectedWeights::validate(const ITensorInfo *src, const ITensorInfo *dst, const TensorShape &original_src_shape, DataLayout data_layout)
{
    return kernels::ClConvertFullyConnectedWeightsKernel::validate(src, dst, original_src_shape, data_layout);
}
} // namespace opencl
} // namespace arm_compute