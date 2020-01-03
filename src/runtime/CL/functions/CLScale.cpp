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
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void CLScale::configure(ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding,
                        bool align_corners)
{
    ARM_COMPUTE_UNUSED(use_padding, align_corners);
    auto k = arm_compute::support::cpp14::make_unique<CLScaleKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(input, output, policy, border_mode, sampling_policy);
    _kernel = std::move(k);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_kernel);

    // In the case of NHWC we can't have undefined border mode as this would require to access elements outside z dimension,
    // so we treat it like border constant.
    if(border_mode == BorderMode::UNDEFINED && input->info()->data_layout() == DataLayout::NHWC)
    {
        border_mode = BorderMode::CONSTANT;
    }
    _border_handler.configure(input, _kernel->border_size(), border_mode, constant_border_value);
}

Status CLScale::validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy,
                         bool use_padding, bool align_corners)
{
    ARM_COMPUTE_UNUSED(constant_border_value, use_padding, align_corners);
    return CLScaleKernel::validate(input, output, policy, border_mode, sampling_policy);
}
