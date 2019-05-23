/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConcatenateLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
GCConcatenateLayer::GCConcatenateLayer()
    : _concat_kernels(),
      _num_inputs(0),
      _axis(Window::DimZ)
{
}

void GCConcatenateLayer::configure(std::vector<IGCTensor *> inputs_vector, IGCTensor *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(inputs_vector.size() < 2);

    _num_inputs = inputs_vector.size();
    _axis       = axis;

    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());

    unsigned int offset = 0;
    switch(axis)
    {
        case Window::DimZ:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = support::cpp14::make_unique<GCDepthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->info()->dimension(axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
}

void GCConcatenateLayer::run()
{
    for(auto &kernel : _concat_kernels)
    {
        GCScheduler::get().dispatch(*kernel, true);
    }
}
} // namespace arm_compute
