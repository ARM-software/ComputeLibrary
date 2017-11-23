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
#include "arm_compute/runtime/CL/functions/CLDepthConcatenateLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLDepthConcatenateLayer::CLDepthConcatenateLayer() // NOLINT
    : _inputs_vector(),
      _concat_kernels_vector(),
      _border_handlers_vector(),
      _num_inputs(0)
{
}

void CLDepthConcatenateLayer::configure(std::vector<ICLTensor *> inputs_vector, ICLTensor *output) // NOLINT
{
    ARM_COMPUTE_ERROR_ON(inputs_vector.size() < 2);

    _num_inputs = inputs_vector.size();

    unsigned int depth_offset = 0;

    _concat_kernels_vector  = arm_compute::support::cpp14::make_unique<CLDepthConcatenateLayerKernel[]>(_num_inputs);
    _border_handlers_vector = arm_compute::support::cpp14::make_unique<CLFillBorderKernel[]>(_num_inputs);

    TensorShape output_shape = calculate_depth_concatenate_shape(inputs_vector);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type(), inputs_vector[0]->info()->fixed_point_position());

    for(unsigned int i = 0; i < _num_inputs; i++)
    {
        _concat_kernels_vector[i].configure(inputs_vector.at(i), depth_offset, output);
        _border_handlers_vector[i].configure(inputs_vector.at(i), _concat_kernels_vector[i].border_size(), BorderMode::CONSTANT, PixelValue(0));

        depth_offset += inputs_vector.at(i)->info()->dimension(2);
    }
}

void CLDepthConcatenateLayer::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    for(unsigned i = 0; i < _num_inputs; i++)
    {
        CLScheduler::get().enqueue(_border_handlers_vector[i], false);
        CLScheduler::get().enqueue(_concat_kernels_vector[i], true);
    }
}
