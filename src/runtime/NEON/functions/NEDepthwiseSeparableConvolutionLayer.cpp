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
#include "arm_compute/runtime/NEON/functions/NEDepthwiseSeparableConvolutionLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEDepthwiseSeparableConvolutionLayer::NEDepthwiseSeparableConvolutionLayer()
    : _depthwise_conv(), _pointwise_conv()
{
}

void NEDepthwiseSeparableConvolutionLayer::configure(ITensor *input, const ITensor *depthwise_weights, const ITensor *depthwise_biases, ITensor *depthwise_out,
                                                     const ITensor *pointwise_weights, const ITensor *pointwise_biases, ITensor *output,
                                                     const PadStrideInfo &depthwise_conv_info, const PadStrideInfo &pointwise_conv_info)
{
    _depthwise_conv.configure(input, depthwise_weights, depthwise_biases, depthwise_out, depthwise_conv_info);
    _pointwise_conv.configure(depthwise_out, pointwise_weights, pointwise_biases, output, pointwise_conv_info);
}

void NEDepthwiseSeparableConvolutionLayer::run()
{
    _depthwise_conv.run();
    _pointwise_conv.run();
}