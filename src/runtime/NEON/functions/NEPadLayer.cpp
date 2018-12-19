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
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
TensorInfo get_expected_output_tensorinfo(const ITensorInfo &input, const PaddingList &paddings)
{
    const TensorShape expected_output_shape = arm_compute::misc::shape_calculator::compute_padded_shape(input.tensor_shape(), paddings);
    const TensorInfo  expected_output_info  = input.clone()->set_tensor_shape(expected_output_shape);
    return expected_output_info;
}

Status validate_arguments(const ITensorInfo &input, ITensorInfo &output, const PaddingList &paddings)
{
    const TensorInfo expected_output_info = get_expected_output_tensorinfo(input, paddings);
    auto_init_if_empty(output, expected_output_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&output, &expected_output_info);

    return Status{};
}

Coordinates get_subtensor_coords(const PaddingList &paddings)
{
    Coordinates coords;
    for(unsigned int i = 0; i < paddings.size(); ++i)
    {
        coords.set(i, paddings[i].first);
    }

    return coords;
}
} // namespace

NEPadLayer::NEPadLayer()
    : _memset_kernel(), _copy_kernel(), _output_subtensor()
{
}

void NEPadLayer::configure(ITensor *input, ITensor *output, const PaddingList &padding, PixelValue constant_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_THROW_ON_ERROR(NEPadLayer::validate(input->info(), output->info(), padding, constant_value));

    // Auto-init
    auto_init_if_empty(*output->info(), get_expected_output_tensorinfo(*input->info(), padding));

    // Create SubTensor (Can use sub-tensor as the kernels to be executed do not require padding)
    _output_subtensor = SubTensor(output, input->info()->tensor_shape(), get_subtensor_coords(padding), true);

    // Set the pages of the output to the specified value
    _memset_kernel.configure(output, constant_value);

    // Copy the input to the output
    _copy_kernel.configure(input, &_output_subtensor);
}

Status NEPadLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value)
{
    ARM_COMPUTE_UNUSED(constant_value);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    auto output_clone = output->clone();

    SubTensorInfo output_subtensor_info(output_clone.get(), input->tensor_shape(), get_subtensor_coords(padding), true);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input, *output_clone, padding));
    ARM_COMPUTE_RETURN_ON_ERROR(NECopyKernel::validate(input, &output_subtensor_info));

    return Status{};
}

void NEPadLayer::run()
{
    NEScheduler::get().schedule(&_memset_kernel, Window::DimY);
    NEScheduler::get().schedule(&_copy_kernel, Window::DimY);
}
} // namespace arm_compute
