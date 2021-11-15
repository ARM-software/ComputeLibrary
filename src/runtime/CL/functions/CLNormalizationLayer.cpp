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

#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLNormalizationLayerKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
CLNormalizationLayer::CLNormalizationLayer()
    : _norm_kernel(std::make_unique<CLNormalizationLayerKernel>()),
      _border_handler(std::make_unique<CLFillBorderKernel>())
{
}

CLNormalizationLayer::~CLNormalizationLayer() = default;

void CLNormalizationLayer::configure(ICLTensor *input, ICLTensor *output, const NormalizationLayerInfo &norm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, norm_info);
}

void CLNormalizationLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const NormalizationLayerInfo &norm_info)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_LOG_PARAMS(input, output, norm_info);

    // Configure normalization kernel
    _norm_kernel->configure(compile_context, input, output, norm_info);

    if(!_norm_kernel->border_size().empty())
    {
        // Fill the border by 3 elements since we need vload4 in the IN_MAP normalization kernel
        _border_handler->configure(compile_context, input, _norm_kernel->border_size(), BorderMode::CONSTANT, PixelValue());
    }
}

Status CLNormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const NormalizationLayerInfo &norm_info)
{
    return CLNormalizationLayerKernel::validate(input, output, norm_info);
}

void CLNormalizationLayer::run()
{
    if(!_norm_kernel->border_size().empty())
    {
        // Run border handler
        CLScheduler::get().enqueue(*_border_handler, false);
    }

    // Run normalization kernel
    CLScheduler::get().enqueue(*_norm_kernel);
}
} // namespace arm_compute
