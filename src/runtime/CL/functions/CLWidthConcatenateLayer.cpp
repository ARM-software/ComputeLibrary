/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLWidthConcatenateLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLWidthConcatenateLayer::CLWidthConcatenateLayer() // NOLINT
    : _concat_kernels_vector(),
      _num_inputs(0)
{
}

Status CLWidthConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output) // NOLINT
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    // Output auto inizialitation if not yet initialized
    TensorInfo  tmp_output_info = *output->clone();
    TensorShape output_shape    = arm_compute::misc::shape_calculator::calculate_width_concatenate_shape(inputs_vector);
    auto_init_if_empty(tmp_output_info, output_shape, 1, inputs_vector[0]->data_type());

    unsigned int width_offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayerKernel::validate(input, width_offset, &tmp_output_info));
        width_offset += input->dimension(0);
    }

    return Status{};
}

void CLWidthConcatenateLayer::configure(std::vector<ICLTensor *> inputs_vector, ICLTensor *output) // NOLINT
{
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < _num_inputs; i++)
    {
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_width_concatenate_shape(inputs_vector);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(CLWidthConcatenateLayer::validate(inputs_vector_info, output->info()));

    unsigned int width_offset = 0;

    _concat_kernels_vector = arm_compute::support::cpp14::make_unique<CLWidthConcatenateLayerKernel[]>(_num_inputs);

    for(unsigned int i = 0; i < _num_inputs; i++)
    {
        _concat_kernels_vector[i].configure(inputs_vector.at(i), width_offset, output);
        width_offset += inputs_vector.at(i)->info()->dimension(0);
    }
}

void CLWidthConcatenateLayer::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    for(unsigned i = 0; i < _num_inputs; i++)
    {
        CLScheduler::get().enqueue(_concat_kernels_vector[i], true);
    }
}
