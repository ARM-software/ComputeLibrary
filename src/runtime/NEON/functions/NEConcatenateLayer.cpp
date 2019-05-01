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
    : _concat_kernels(),
      _num_inputs(0),
      _axis(Window::DimX)
{
}

void NEConcatenateLayer::configure(const std::vector<ITensor *> &inputs_vector, ITensor *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _axis       = axis;
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info;
    inputs_vector_info.reserve(_num_inputs);
    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(inputs_vector.at(i));
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, _axis);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(NEConcatenateLayer::validate(inputs_vector_info, output->info(), axis));

    unsigned int offset = 0;

    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        switch(_axis)
        {
            case Window::DimX:
            {
                auto kernel = support::cpp14::make_unique<NEWidthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimY:
            {
                auto kernel = support::cpp14::make_unique<NEHeightConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimZ:
            {
                auto kernel = support::cpp14::make_unique<NEDepthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += inputs_vector.at(i)->info()->dimension(_axis);
    }
}

Status NEConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    unsigned int offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        switch(axis)
        {
            case Window::DimX:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEWidthConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            case Window::DimY:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEHeightConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            case Window::DimZ:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEDepthConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += input->dimension(axis);
    }

    if(output->total_size() != 0)
    {
        TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    return Status{};
}

void NEConcatenateLayer::run()
{
    for(auto &kernel : _concat_kernels)
    {
        NEScheduler::get().schedule(kernel.get(), _axis);
    }
}
} // namespace arm_compute
