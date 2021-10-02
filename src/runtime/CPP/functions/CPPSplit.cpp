/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPSplit.h"

#ifdef ARM_COMPUTE_CPU_ENABLED // NEON Build is activated
#include "arm_compute/runtime/NEON/functions/NESlice.h"
#endif /* ARM_COMPUTE_CPU_ENABLED */

#ifdef ARM_COMPUTE_OPENCL_ENABLED // OPENCL build is activated
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#endif /* ARM_COMPUTE_OPENCL_ENABLED */

#include "src/common/utils/Log.h"

namespace arm_compute
{
/** Basic function to split a tensor along a given axis */

template <typename SliceType, typename TensorInterfaceType>
CPPSplit<SliceType, TensorInterfaceType>::CPPSplit()
    : _outputs_vector(), _slice_functions(), _num_outputs(0)
{
}

template <typename SliceType, typename TensorInterfaceType>
Status CPPSplit<SliceType, TensorInterfaceType>::validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &outputs, unsigned int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(axis >= input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(outputs.size() < 2);

    // Get output shape
    TensorShape  output_shape{};
    unsigned int total_output_shape_size = 0;

    // Sum the output sizes and fall back to evenly-sized splits if any are zero
    const bool using_split_shapes = std::none_of(outputs.begin(), outputs.end(), [&total_output_shape_size](ITensorInfo * info)
    {
        unsigned int output_shape_size = info->tensor_shape().total_size();
        total_output_shape_size += output_shape_size;
        return output_shape_size == 0;
    });

    if(using_split_shapes)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().total_size() != total_output_shape_size);
    }
    else
    {
        output_shape = arm_compute::misc::shape_calculator::compute_split_shape(input, axis, outputs.size());
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);
    }

    // Validate output tensors
    unsigned int axis_offset = 0;
    for(const auto &output : outputs)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
        if(using_split_shapes)
        {
            output_shape = output->tensor_shape();
            ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);
        }

        const size_t axis_split_step = output_shape[axis];

        // Start/End coordinates
        Coordinates start_coords;
        Coordinates end_coords;
        for(unsigned int d = 0; d < output_shape.num_dimensions(); ++d)
        {
            end_coords.set(d, -1);
        }

        // Output auto inizialitation if not yet initialized
        TensorInfo tmp_output_info = *output->clone();
        if(tmp_output_info.tensor_shape().total_size() == 0)
        {
            tmp_output_info = input->clone()->set_is_resizable(true).set_tensor_shape(output_shape);
        }

        // Update coordinate on axis
        start_coords.set(axis, axis_offset);
        end_coords.set(axis, axis_offset + axis_split_step);

        ARM_COMPUTE_RETURN_ON_ERROR(SliceType::validate(input, output, start_coords, end_coords));
        axis_offset += axis_split_step;
    }

    return Status{};
}

template <typename SliceType, typename TensorInterfaceType>
void CPPSplit<SliceType, TensorInterfaceType>::configure(const TensorInterfaceType *input, const std::vector<TensorInterfaceType *> &outputs, unsigned int axis)
{
    // (TensorInterfaceType*)
    ARM_COMPUTE_LOG_PARAMS(input, outputs, axis);

    // Create Slice functions
    _num_outputs = outputs.size();
    _slice_functions.resize(_num_outputs);

    // Extract output tensor info
    std::vector<ITensorInfo *> outputs_info;
    for(auto &output : outputs)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(output);
        outputs_info.emplace_back(output->info());
    }

    // If any of the outputs have a zero size, fall-back to using evenly-sized output splits
    const bool outputs_have_sizes = std::none_of(outputs_info.begin(), outputs_info.end(), [](ITensorInfo * info)
    {
        return info->tensor_shape().total_size() == 0;
    });

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(CPPSplit::validate(input->info(), outputs_info, axis));

    unsigned int axis_offset = 0;
    unsigned int i           = 0;

    for(const auto &output_info : outputs_info)
    {
        // Get output shape
        TensorShape output_shape = (outputs_have_sizes ?
                                    output_info->tensor_shape() :
                                    arm_compute::misc::shape_calculator::compute_split_shape(input->info(), axis, _num_outputs));

        const size_t axis_split_step = output_shape[axis];

        // Start/End coordinates
        Coordinates start_coords;
        Coordinates end_coords;

        for(unsigned int d = 0; d < output_shape.num_dimensions(); ++d)
        {
            end_coords.set(d, -1);
        }

        // Update coordinate on axis
        start_coords.set(axis, axis_offset);
        end_coords.set(axis, axis_offset + axis_split_step);

        // Configure slice function
        _slice_functions[i].configure(input, outputs[i], start_coords, end_coords);

        // Set valid region from shape
        outputs[i]->info()->set_valid_region(ValidRegion(Coordinates(), output_shape));

        // Update axis offset
        axis_offset += axis_split_step;
        ++i;
    }
}

// Instantiate CPPSplit for NESlice and CLSlice types to enable linking to the above templated CPPSplit's methods
#ifdef ARM_COMPUTE_CPU_ENABLED // NEON Build is activated
template class CPPSplit<NESlice, ITensor>;
#endif /* ARM_COMPUTE_CPU_ENABLED */

#ifdef ARM_COMPUTE_OPENCL_ENABLED // OPENCL build is activated
template class CPPSplit<CLSlice, ICLTensor>;
#endif /* ARM_COMPUTE_OPENCL_ENABLED */
} // namespace arm_compute
