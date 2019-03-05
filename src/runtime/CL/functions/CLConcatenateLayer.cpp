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
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"

#include "arm_compute/core/CL/kernels/CLHeightConcatenateLayerKernel.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLDepthConcatenateLayer.h"
#include "arm_compute/runtime/CL/functions/CLWidthConcatenateLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLConcatenateLayer::CLConcatenateLayer()
    : _concat_function(nullptr),
      _hconcat_kernels(),
      _num_inputs(0),
      _axis(Window::DimX)
{
}

Status CLConcatenateLayer::validate_h_concatenate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output) // NOLINT
{
    const unsigned int num_inputs = inputs_vector.size();

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(num_inputs < 2);

    // Output auto inizialitation if not yet initialized
    TensorInfo        tmp_output_info = *output->clone();
    const TensorShape output_shape    = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimY);
    auto_init_if_empty(tmp_output_info, output_shape, 1, inputs_vector[0]->data_type());

    unsigned int height_offset = 0;
    // Validate generic case of WidthConcatenate kernel
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        ARM_COMPUTE_RETURN_ON_ERROR(CLHeightConcatenateLayerKernel::validate(input, height_offset, &tmp_output_info));
        height_offset += input->dimension(Window::DimY);
    }

    return Status{};
}

void CLConcatenateLayer::configure_h_concatenate(std::vector<ICLTensor *> inputs_vector, ICLTensor *output) // NOLINT
{
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info(inputs_vector.size());
    std::transform(inputs_vector.begin(), inputs_vector.end(), inputs_vector_info.begin(), [](ICLTensor * t)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(t);
        return t->info();
    });

    const TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimY);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());

    ARM_COMPUTE_ERROR_THROW_ON(CLConcatenateLayer::validate_h_concatenate(inputs_vector_info, output->info()));

    // Configure generic case WidthConcatenate kernels
    _hconcat_kernels = arm_compute::support::cpp14::make_unique<CLHeightConcatenateLayerKernel[]>(_num_inputs);

    unsigned int height_offset = 0;
    unsigned int i             = 0;
    std::transform(inputs_vector.begin(), inputs_vector.end(), inputs_vector.begin(), [&](ICLTensor * t)
    {
        auto &kernel = _hconcat_kernels[i++];
        kernel.configure(t, height_offset, output);
        height_offset += t->info()->dimension(Window::DimY);
        return t;
    });
}

void CLConcatenateLayer::configure(const std::vector<ICLTensor *> &inputs_vector, ICLTensor *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _axis = get_data_layout_dimension_index(output->info()->data_layout(), axis);
    switch(_axis)
    {
        case 0:
        {
            auto func = support::cpp14::make_unique<CLWidthConcatenateLayer>();
            func->configure(inputs_vector, output);
            _concat_function = std::move(func);
            break;
        }
        case 1:
        {
            configure_h_concatenate(inputs_vector, output);
            break;
        }
        case 2:
        {
            auto func = support::cpp14::make_unique<CLDepthConcatenateLayer>();
            func->configure(inputs_vector, output);
            _concat_function = std::move(func);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Concatenation is supported across width, height and depth only!");
    }
}

Status CLConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(output == nullptr);

    switch(get_data_layout_dimension_index(output->data_layout(), axis))
    {
        case 0:
            ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayer::validate(inputs_vector, output));
            break;
        case 1:
            ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate_h_concatenate(inputs_vector, output));
            break;
        case 2:
            ARM_COMPUTE_RETURN_ON_ERROR(CLDepthConcatenateLayer::validate(inputs_vector, output));
            break;
        default:
            ARM_COMPUTE_RETURN_ERROR_MSG("Concatenation is supported across width and depth only!");
    }
    return Status{};
}

void CLConcatenateLayer::run()
{
    switch(_axis)
    {
        case 0:
        case 2:
        {
            ARM_COMPUTE_ERROR_ON(_concat_function == nullptr);
            _concat_function->run();
            break;
        }
        case 1:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                CLScheduler::get().enqueue(_hconcat_kernels[i], true);
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Axis not supported");
            break;
        }
    }
}
} // namespace arm_compute
