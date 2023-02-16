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
#include "src/gpu/cl/operators/ClTransposedConvolution.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClTransposedConvolutionKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClTransposedConvolution::configure(const CLCompileContext &compile_context, const ITensorInfo *input, const ITensorInfo *weights,
                                        const ITensorInfo *biases, ITensorInfo *output, const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, deconv_info);
    auto kernel_object = std::make_unique<kernels::ClTransposedConvolutionKernel>();
    kernel_object->set_target(CLScheduler::get().target());
    kernel_object->configure(compile_context, input, weights, biases, output, deconv_info);
    _transposed_conv_kernel = std::move(kernel_object);
}

Status ClTransposedConvolution::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases,
                                         const ITensorInfo *output, const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClTransposedConvolutionKernel::validate(input, weights, biases, output, deconv_info));
    return Status{};
}

void ClTransposedConvolution::run(ITensorPack &tensors)
{
    CLScheduler::get().enqueue_op(*_transposed_conv_kernel.get(), tensors, false);
}
} // namespace opencl
} // namespace arm_compute
