/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLScale.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLScaleKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
void CLScale::configure(ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, info);
}

void CLScale::configure(ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding,
                        bool align_corners)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy, use_padding, align_corners });
}

void CLScale::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLScaleKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, input, output, info);
    _kernel = std::move(k);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_kernel);

    auto border_mode_to_use = info.border_mode;
    // In the case of NHWC we can't have undefined border mode as this would require to access elements outside z dimension,
    // so we treat it like border constant.
    if(info.border_mode == BorderMode::UNDEFINED && input->info()->data_layout() == DataLayout::NHWC)
    {
        border_mode_to_use = BorderMode::CONSTANT;
    }
    _border_handler.configure(compile_context, input, _kernel->border_size(), border_mode_to_use, info.constant_border_value);
}

void CLScale::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value,
                        SamplingPolicy sampling_policy, bool use_padding, bool align_corners)
{
    configure(compile_context, input, output, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy, use_padding, align_corners });
}

Status CLScale::validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy,
                         bool use_padding, bool align_corners)
{
    return CLScale::validate(input, output, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy, use_padding, align_corners });
}

Status CLScale::validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info)
{
    return CLScaleKernel::validate(input, output, info);
}
} // namespace arm_compute
