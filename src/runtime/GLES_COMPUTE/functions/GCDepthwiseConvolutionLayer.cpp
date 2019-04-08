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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDepthwiseConvolutionLayer.h"

#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

GCDepthwiseConvolutionLayer3x3::GCDepthwiseConvolutionLayer3x3()
    : _kernel(nullptr), _border_handler(), _shift_handler(), _activationlayer_function(), _is_activationlayer_enabled(false)
{
}

void GCDepthwiseConvolutionLayer3x3::configure(IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info,
                                               unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON(dilation.x() != 1 || dilation.y() != 1);
    ARM_COMPUTE_UNUSED(dilation);
    auto k = arm_compute::support::cpp14::make_unique<GCDepthwiseConvolutionLayer3x3Kernel>();
    k->configure(input, weights, biases, output, conv_info, depth_multiplier);
    _kernel = std::move(k);

    // Configure border handler
    _border_handler.configure(input, _kernel->border_size(), BorderMode::CONSTANT, PixelValue());

    _shift_handler.configure(input);

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

void GCDepthwiseConvolutionLayer3x3::run()
{
    GCScheduler::get().dispatch(_shift_handler, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_border_handler, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(*_kernel);

    // Run Activation Layer
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}
