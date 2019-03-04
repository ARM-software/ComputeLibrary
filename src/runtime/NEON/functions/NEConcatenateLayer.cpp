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
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"

#include "arm_compute/runtime/NEON/functions/NEDepthConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWidthConcatenateLayer.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
NEConcatenateLayer::NEConcatenateLayer()
    : _concat_function(nullptr),
      _hconcat_kernels(),
      _num_inputs(0),
      _axis(Window::DimX)
{
}

Status NEConcatenateLayer::validate_h_concatenate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    // Output auto inizialitation if not yet initialized
    TensorInfo  tmp_output_info = *output->clone();
    TensorShape output_shape    = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimY);
    auto_init_if_empty(tmp_output_info, output_shape, 1, inputs_vector[0]->data_type());

    unsigned int offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        ARM_COMPUTE_RETURN_ON_ERROR(NEHeightConcatenateLayerKernel::validate(input, offset, &tmp_output_info));
        offset += input->dimension(Window::DimY);
    }

    return Status{};
}

void NEConcatenateLayer::configure_h_concatenate(std::vector<ITensor *> inputs_vector, ITensor *output)
{
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(inputs_vector.at(i));
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimY);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(validate_h_concatenate(inputs_vector_info, output->info()));

    unsigned int offset = 0;

    _hconcat_kernels = arm_compute::support::cpp14::make_unique<NEHeightConcatenateLayerKernel[]>(_num_inputs);

    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        _hconcat_kernels[i].configure(inputs_vector.at(i), offset, output);
        offset += inputs_vector.at(i)->info()->dimension(Window::DimY);
    }
}

void NEConcatenateLayer::configure(const std::vector<ITensor *> &inputs_vector, ITensor *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _axis = get_data_layout_dimension_index(output->info()->data_layout(), axis);
    switch(_axis)
    {
        case 0:
        {
            auto func = support::cpp14::make_unique<NEWidthConcatenateLayer>();
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
            auto func = support::cpp14::make_unique<NEDepthConcatenateLayer>();
            func->configure(inputs_vector, output);
            _concat_function = std::move(func);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Concatenation is supported across width and depth only!");
    }
}

Status NEConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(output == nullptr);

    switch(get_data_layout_dimension_index(output->data_layout(), axis))
    {
        case 0:
            ARM_COMPUTE_RETURN_ON_ERROR(NEWidthConcatenateLayer::validate(inputs_vector, output));
            break;
        case 1:
            ARM_COMPUTE_RETURN_ON_ERROR(NEConcatenateLayer::validate_h_concatenate(inputs_vector, output));
            break;
        case 2:
            ARM_COMPUTE_RETURN_ON_ERROR(NEDepthConcatenateLayer::validate(inputs_vector, output));
            break;
        default:
            ARM_COMPUTE_RETURN_ERROR_MSG("Concatenation is supported across width and depth only!");
    }
    return Status{};
}

void NEConcatenateLayer::run()
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
            for(unsigned i = 0; i < _num_inputs; ++i)
            {
                NEScheduler::get().schedule(_hconcat_kernels.get() + i, Window::DimY);
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Axis not supported.");
            break;
        }
    }
}
} // namespace arm_compute
