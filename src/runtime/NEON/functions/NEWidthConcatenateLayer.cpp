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
#include "arm_compute/runtime/NEON/functions/NEWidthConcatenateLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEWidthConcatenateLayer::NEWidthConcatenateLayer()
    : _concat_kernels_vector(),
      _num_inputs(0)
{
}

template <typename TensorInfoType, typename>
inline Status NEWidthConcatenateLayer::validate_internal(const std::vector<TensorInfoType *> &inputs_vector, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    // Output auto inizialitation if not yet initialized
    TensorInfo  tmp_output_info = *output->clone();
    TensorShape output_shape    = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimX);
    auto_init_if_empty(tmp_output_info, output_shape, 1, inputs_vector[0]->data_type());

    unsigned int width_offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        ARM_COMPUTE_RETURN_ON_ERROR(NEWidthConcatenateLayerKernel::validate(input, width_offset, &tmp_output_info));
        width_offset += input->dimension(0);
    }

    return Status{};
}
template <typename TensorType, typename>
inline void NEWidthConcatenateLayer::configure_internal(std::vector<TensorType *> &&inputs_vector, ITensor *output)
{
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimX);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(NEWidthConcatenateLayer::validate(inputs_vector_info, output->info()));

    unsigned int width_offset = 0;

    _concat_kernels_vector.resize(_num_inputs);

    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        _concat_kernels_vector[i].configure(inputs_vector.at(i), width_offset, output);
        width_offset += inputs_vector.at(i)->info()->dimension(0);
    }
}

void NEWidthConcatenateLayer::configure(std::vector<ITensor *> inputs_vector, ITensor *output)
{
    configure_internal(std::move(inputs_vector), output);
}

void NEWidthConcatenateLayer::configure(std::vector<const ITensor *> inputs_vector, ITensor *output)
{
    configure_internal(std::move(inputs_vector), output);
}

Status NEWidthConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output)
{
    return validate_internal(inputs_vector, output);
}

Status NEWidthConcatenateLayer::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output)
{
    return validate_internal(inputs_vector, output);
}

void NEWidthConcatenateLayer::run()
{
    for(unsigned i = 0; i < _num_inputs; ++i)
    {
        NEScheduler::get().schedule(&_concat_kernels_vector[i], Window::DimY);
    }
}
