/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLSplit.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLSplit::CLSplit()
    : _outputs_vector(), _slice_functions(), _num_outputs(0)
{
}

void CLSplit::configure(const ICLTensor *input, const std::vector<ICLTensor *> &outputs, unsigned int axis)
{
    // Create Slice functions
    _num_outputs = outputs.size();
    _slice_functions.resize(_num_outputs);

    // Get output shape
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_split_shape(input->info(), axis, _num_outputs);

    // Extract output tensor info
    std::vector<ITensorInfo *> outputs_info;
    for(auto &output : outputs)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(output);
        outputs_info.emplace_back(output->info());
    }

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(CLSplit::validate(input->info(), outputs_info, axis));

    const size_t axis_split_step = output_shape[axis];
    unsigned int axis_offset     = 0;

    // Start/End coordinates
    Coordinates start_coords;
    Coordinates end_coords;
    for(unsigned int d = 0; d < output_shape.num_dimensions(); ++d)
    {
        end_coords.set(d, -1);
    }

    for(unsigned int i = 0; i < _num_outputs; i++)
    {
        // Update coordinate on axis
        start_coords.set(axis, axis_offset);
        end_coords.set(axis, axis_offset + axis_split_step);

        // Configure slice function
        _slice_functions[i].configure(input, outputs[i], start_coords, end_coords);

        // Set valid region from shape
        outputs[i]->info()->set_valid_region(ValidRegion(Coordinates(), output_shape));

        // Update axis offset
        axis_offset += axis_split_step;
    }
}

Status CLSplit::validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &outputs, unsigned int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(axis >= input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(outputs.size() < 2);

    // Get output shape
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_split_shape(input, axis, outputs.size());
    ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);

    const size_t axis_split_step = output_shape[axis];
    unsigned int axis_offset     = 0;

    // Start/End coordinates
    Coordinates start_coords;
    Coordinates end_coords;
    for(unsigned int d = 0; d < output_shape.num_dimensions(); ++d)
    {
        end_coords.set(d, -1);
    }

    // Validate output tensors
    for(const auto &output : outputs)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

        // Output auto inizialitation if not yet initialized
        TensorInfo tmp_output_info = *output->clone();
        auto_init_if_empty(tmp_output_info, input->clone()->set_is_resizable(true).set_tensor_shape(output_shape));

        // Update coordinate on axis
        start_coords.set(axis, axis_offset);
        end_coords.set(axis, axis_offset + axis_split_step);

        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(input, output, start_coords, end_coords));
        axis_offset += axis_split_step;
    }

    return Status{};
}

void CLSplit::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    for(unsigned i = 0; i < _num_outputs; ++i)
    {
        _slice_functions[i].run();
    }
}
} // namespace arm_compute
