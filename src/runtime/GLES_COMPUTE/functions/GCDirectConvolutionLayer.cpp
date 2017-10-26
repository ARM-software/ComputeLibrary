/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDirectConvolutionLayer.h"

#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDirectConvolutionLayerKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void GCDirectConvolutionLayer::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info)
{
    int kernel_size = weights->info()->dimension(0);

    if(kernel_size == 1)
    {
        auto k = arm_compute::support::cpp14::make_unique<GCDirectConvolutionLayer1x1Kernel>();
        k->configure(input, weights, biases, output, conv_info);
        _kernel = std::move(k);
    }
    else if(kernel_size == 3)
    {
        auto k = arm_compute::support::cpp14::make_unique<GCDirectConvolutionLayer3x3Kernel>();
        k->configure(input, weights, biases, output, conv_info);
        _kernel = std::move(k);
    }
    else if(kernel_size == 5)
    {
        auto k = arm_compute::support::cpp14::make_unique<GCDirectConvolutionLayer5x5Kernel>();
        k->configure(input, weights, biases, output, conv_info);
        _kernel = std::move(k);
    }
    else
    {
        ARM_COMPUTE_ERROR("kernel size unsupported!");
        return;
    }

    _border_handler.configure(input, _kernel->border_size(), BorderMode::CONSTANT, PixelValue(0));
}
