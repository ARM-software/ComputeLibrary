/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLDeconvolutionLayer::CLDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _function()
{
}

void CLDeconvolutionLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info,
                                     unsigned int inner_border_right, unsigned int inner_border_top, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_UNUSED(inner_border_right, inner_border_top);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    auto f = arm_compute::support::cpp14::make_unique<CLDirectDeconvolutionLayer>();
    f->configure(input, weights, bias, output, deconv_info, weights_info);
    _function = std::move(f);
}

Status CLDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &deconv_info,
                                      unsigned int inner_border_right, unsigned int inner_border_top, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(inner_border_right, inner_border_top);
    ARM_COMPUTE_RETURN_ON_ERROR(CLDirectDeconvolutionLayer::validate(input, weights, bias, output, deconv_info, weights_info));
    return Status{};
}

void CLDeconvolutionLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info,
                                     const WeightsInfo &weights_info)
{
    configure(input, weights, bias, output, deconv_info, 0, 0, weights_info);
}

Status CLDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &deconv_info,
                                      const WeightsInfo &weights_info)
{
    return CLDeconvolutionLayer::validate(input, weights, bias, output, deconv_info, 0, 0, weights_info);
}

void CLDeconvolutionLayer::run()
{
    prepare();
    _function->run();
}

void CLDeconvolutionLayer::prepare()
{
    _function->prepare();
}
