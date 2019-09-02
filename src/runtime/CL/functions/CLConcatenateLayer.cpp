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

#include "arm_compute/core/CL/kernels/CLBatchConcatenateLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthConcatenateLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLHeightConcatenateLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLWidthConcatenate2TensorsKernel.h"
#include "arm_compute/core/CL/kernels/CLWidthConcatenate4TensorsKernel.h"
#include "arm_compute/core/CL/kernels/CLWidthConcatenateLayerKernel.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLConcatenateLayer::CLConcatenateLayer()
    : _concat_kernels(),
      _num_inputs(0),
      _axis(Window::DimX)
{
}

void CLConcatenateLayer::configure(std::vector<ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis)
{
    configure_internal(std::move(inputs_vector), output, axis);
}

void CLConcatenateLayer::configure(std::vector<const ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis)
{
    configure_internal(std::move(inputs_vector), output, axis);
}

Status CLConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    return validate_internal(inputs_vector, output, axis);
}

Status CLConcatenateLayer::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    return validate_internal(inputs_vector, output, axis);
}

template <typename TensorType>
void CLConcatenateLayer::configure_internal(std::vector<TensorType *> &&inputs_vector, ICLTensor *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _axis       = axis;
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info(inputs_vector.size());
    std::transform(inputs_vector.begin(), inputs_vector.end(), inputs_vector_info.begin(), [](TensorType * t)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(t);
        return t->info();
    });
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, _axis);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(CLConcatenateLayer::validate(inputs_vector_info, output->info(), axis));

    unsigned int offset = 0;
    switch(_axis)
    {
        case Window::DimX:
        {
            switch(_num_inputs)
            {
                case 2:
                {
                    // Configure WidthConcatenate2Tensors kernel
                    auto kernel = support::cpp14::make_unique<CLWidthConcatenate2TensorsKernel>();
                    kernel->configure(inputs_vector.at(0), inputs_vector.at(1), output);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                case 4:
                {
                    // Configure WidthConcatenate4Tensors kernel
                    auto kernel = support::cpp14::make_unique<CLWidthConcatenate4TensorsKernel>();
                    kernel->configure(inputs_vector.at(0), inputs_vector.at(1), inputs_vector.at(2), inputs_vector.at(3), output);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                default:
                {
                    // Configure generic case WidthConcatenate kernels
                    for(unsigned int i = 0; i < _num_inputs; ++i)
                    {
                        auto kernel = support::cpp14::make_unique<CLWidthConcatenateLayerKernel>();
                        kernel->configure(inputs_vector.at(i), offset, output);
                        offset += inputs_vector.at(i)->info()->dimension(_axis);
                        _concat_kernels.emplace_back(std::move(kernel));
                    }
                    break;
                }
            }
            break;
        }
        case Window::DimY:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = support::cpp14::make_unique<CLHeightConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->info()->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case Window::DimZ:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = support::cpp14::make_unique<CLDepthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->info()->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case 3:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = support::cpp14::make_unique<CLBatchConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->info()->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
}

template <typename TensorInfoType>
Status CLConcatenateLayer::validate_internal(const std::vector<TensorInfoType *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(output == nullptr);
    const unsigned int num_inputs = inputs_vector.size();

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(num_inputs < 2);

    unsigned int offset = 0;
    switch(axis)
    {
        case Window::DimX:
        {
            switch(num_inputs)
            {
                case 2:
                    // Validate WidthConcatenate2Tensors kernels if there are 2 inputs
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(inputs_vector[0], inputs_vector[1]);
                    ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenate2TensorsKernel::validate(inputs_vector[0], inputs_vector[1], output));
                    break;
                case 4:
                    // Validate WidthConcatenate4Tensors kernels if there are 4 inputs
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(inputs_vector[0], inputs_vector[1], inputs_vector[2], inputs_vector[3]);
                    ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenate4TensorsKernel::validate(inputs_vector[0], inputs_vector[1], inputs_vector[2], inputs_vector[3], output));
                    break;
                default:
                    // Validate generic case of WidthConcatenate kernel
                    for(const auto &input : inputs_vector)
                    {
                        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
                        ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayerKernel::validate(input, offset, output));
                        offset += input->dimension(axis);
                    }
                    break;
            }
            break;
        }
        case Window::DimY:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLHeightConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        case Window::DimZ:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLDepthConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        case 3:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLBatchConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }

    if(output->total_size() != 0)
    {
        TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    return Status{};
}

void CLConcatenateLayer::run()
{
    for(auto &kernel : _concat_kernels)
    {
        CLScheduler::get().enqueue(*kernel, true);
    }
}
} // namespace arm_compute
