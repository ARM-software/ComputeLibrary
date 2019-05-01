/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenateLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEDepthConcatenateLayer::NEDepthConcatenateLayer() // NOLINT
    : _inputs_vector(),
      _concat_kernels_vector(),
      _border_handlers_vector(),
      _num_inputs(0)
{
}

void NEDepthConcatenateLayer::configure(const std::vector<ITensor *> &inputs_vector, ITensor *output) // NOLINT
{
    _num_inputs = inputs_vector.size();

    std::vector<ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < _num_inputs; i++)
    {
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector_info, Window::DimZ);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, inputs_vector[0]->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(NEDepthConcatenateLayer::validate(inputs_vector_info, output->info()));

    unsigned int depth_offset = 0;
    _concat_kernels_vector.reserve(_num_inputs);
    _border_handlers_vector.reserve(_num_inputs);
    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        auto concat_kernel = support::cpp14::make_unique<NEDepthConcatenateLayerKernel>();
        auto border_kernel = support::cpp14::make_unique<NEFillBorderKernel>();
        concat_kernel->configure(inputs_vector.at(i), depth_offset, output);
        border_kernel->configure(inputs_vector.at(i), concat_kernel->border_size(), BorderMode::CONSTANT, PixelValue(static_cast<float>(0.f)));
        _border_handlers_vector.emplace_back(std::move(border_kernel));
        _concat_kernels_vector.emplace_back(std::move(concat_kernel));

        depth_offset += inputs_vector.at(i)->info()->dimension(2);
    }

    // Set valid region from shape
    output->info()->set_valid_region(ValidRegion(Coordinates(), output_shape));
}

Status NEDepthConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    // Output auto inizialitation if not yet initialized
    TensorInfo  tmp_output_info = *output->clone();
    TensorShape output_shape    = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, Window::DimZ);
    auto_init_if_empty(tmp_output_info, output_shape, 1, inputs_vector[0]->data_type());

    unsigned int depth_offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        ARM_COMPUTE_RETURN_ON_ERROR(NEDepthConcatenateLayerKernel::validate(input, depth_offset, &tmp_output_info));
        depth_offset += input->dimension(2);
    }

    return Status{};
}

void NEDepthConcatenateLayer::run()
{
    for(unsigned i = 0; i < _num_inputs; ++i)
    {
        NEScheduler::get().schedule(_border_handlers_vector[i].get(), Window::DimX);
        NEScheduler::get().schedule(_concat_kernels_vector[i].get(), Window::DimX);
    }
}
