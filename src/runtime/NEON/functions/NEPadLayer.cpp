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

uint32_t last_padding_dimension(const PaddingList &padding)
{
    int last_padding_dim = padding.size() - 1;
    for(; last_padding_dim >= 0; --last_padding_dim)
    {
        if(padding[last_padding_dim].first > 0 || padding[last_padding_dim].second > 0)
        {
            break;
        }
    }
    return static_cast<uint32_t>(last_padding_dim);
}
} // namespace

NEPadLayer::NEPadLayer()
    : _copy_kernel(), _mode(), _padding(), _memset_kernel(), _num_dimensions(0), _slice_functions(), _concat_functions(), _slice_results(), _concat_results(), _output_subtensor()
{
}

void NEPadLayer::configure_constant_mode(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value)
{
    // Auto-init
    auto_init_if_empty(*output->info(), get_expected_output_tensorinfo(*input->info(), padding));

    // Create SubTensor (Can use sub-tensor as the kernels to be executed do not require padding)
    _output_subtensor = SubTensor(output, input->info()->tensor_shape(), get_subtensor_coords(padding), true);

    // Set the pages of the output to the specified value
    _memset_kernel.configure(output, constant_value);

    // Copy the input to the output
    _copy_kernel.configure(input, &_output_subtensor);
}

void NEPadLayer::configure_reflect_symmetric_mode(ITensor *input, ITensor *output)
{
    // Reflecting can be performed by effectively unfolding the input as follows:
    // For each dimension starting at DimX:
    //      For before and after:
    //          Use strided slice to extract and reverse the part of the
    //          input / previously produced tensor required for the padding.
    //      Concatenate the before and after padding with the input / previously
    //      produced tensor along the current dimension.

    // Two strided slice functions will be required for each dimension padded as well as a
    // concatenate function and the tensors to hold the temporary results.
    _slice_functions.resize(2 * _num_dimensions);
    _slice_results.resize(2 * _num_dimensions);
    _concat_functions.resize(_num_dimensions);
    _concat_results.resize(_num_dimensions - 1);

    Coordinates starts_before{};
    Coordinates ends_before{};
    Coordinates starts_after{};
    Coordinates ends_after{};
    Coordinates strides{};
    ITensor    *prev = input;
    for(uint32_t i = 0; i < _num_dimensions; ++i)
    {
        // Values in strides from the previous dimensions need to be set to 1 to avoid reversing again.
        if(i > 0)
        {
            strides.set(i - 1, 1);
        }

        if(_padding[i].first > 0 || _padding[i].second > 0)
        {
            // Set the starts, ends, and strides values for the current dimension.
            // Due to the bit masks passed to strided slice, the values below the current dimension in
            // starts and ends will be ignored so do not need to be modified.
            if(_mode == PaddingMode::REFLECT)
            {
                starts_before.set(i, _padding[i].first);
                ends_before.set(i, 0);
                starts_after.set(i, input->info()->dimension(i) - 2);
                ends_after.set(i, input->info()->dimension(i) - _padding[i].second - 2);
                strides.set(i, -1);
            }
            else
            {
                starts_before.set(i, _padding[i].first - 1);
                ends_before.set(i, -1);
                starts_after.set(i, input->info()->dimension(i) - 1);
                ends_after.set(i, input->info()->dimension(i) - _padding[i].second - 1);
                strides.set(i, -1);
            }

            // Strided slice wraps negative indexes around to the end of the range,
            // instead this should indicate use of the full range and so the bit mask will be modified.
            const int32_t begin_mask_before = starts_before[i] < 0 ? ~0 : ~(1u << i);
            const int32_t end_mask_before   = ends_before[i] < 0 ? ~0 : ~(1u << i);
            const int32_t begin_mask_after  = starts_after[i] < 0 ? ~0 : ~(1u << i);
            const int32_t end_mask_after    = ends_after[i] < 0 ? ~0 : ~(1u << i);

            // Reflect the input values for the padding before and after the input.
            std::vector<ITensor *> concat_vector;
            if(_padding[i].first > 0)
            {
                if(i < prev->info()->num_dimensions())
                {
                    _slice_functions[2 * i].configure(prev, &_slice_results[2 * i], starts_before, ends_before, strides, begin_mask_before, end_mask_before);
                    concat_vector.emplace_back(&_slice_results[2 * i]);
                }
                else
                {
                    // Performing the slice is unnecessary if the result would simply be a copy of the tensor.
                    concat_vector.push_back(prev);
                }
            }
            concat_vector.push_back(prev);
            if(_padding[i].second > 0)
            {
                if(i < prev->info()->num_dimensions())
                {
                    _slice_functions[2 * i + 1].configure(prev, &_slice_results[2 * i + 1], starts_after, ends_after, strides, begin_mask_after, end_mask_after);
                    concat_vector.emplace_back(&_slice_results[2 * i + 1]);
                }
                else
                {
                    // Performing the slice is unnecessary if the result would simply be a copy of the tensor.
                    concat_vector.push_back(prev);
                }
            }
            // Concatenate the padding before and after with the input.
            ITensor *out = (i == _num_dimensions - 1) ? output : &_concat_results[i];
            _concat_functions[i].configure(concat_vector, out, i);
            if(i != _num_dimensions - 1)
            {
                _concat_results[i].allocator()->allocate();
            }
            prev = out;
        }
        _slice_results[2 * i].allocator()->allocate();
        _slice_results[2 * i + 1].allocator()->allocate();
    }
}

void NEPadLayer::configure(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value, const PaddingMode mode)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), padding, constant_value, mode));

    _padding = padding;
    _mode    = mode;

    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->info()->tensor_shape(), _padding);

    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(padded_shape));

    // Find the last dimension requiring padding so that it is known when to write to output and whether any padding is applied.
    _num_dimensions = last_padding_dimension(padding) + 1;
    if(_num_dimensions > 0)
    {
        switch(_mode)
        {
            case PaddingMode::CONSTANT:
            {
                configure_constant_mode(input, output, padding, constant_value);
                break;
            }
            case PaddingMode::REFLECT:
            case PaddingMode::SYMMETRIC:
            {
                configure_reflect_symmetric_mode(input, output);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Padding mode not supported.");
        }
    }
    else
    {
        // Copy the input to the whole output if no padding is applied
        _copy_kernel.configure(input, output);
    }
}

Status NEPadLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, const PixelValue constant_value, const PaddingMode mode)
{
    ARM_COMPUTE_UNUSED(constant_value);

    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);

    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), padded_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    switch(mode)
    {
        case PaddingMode::CONSTANT:
        {
            auto          output_clone = output->clone();
            SubTensorInfo output_subtensor_info(output_clone.get(), input->tensor_shape(), get_subtensor_coords(padding), true);
            ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input, *output_clone, padding));
            ARM_COMPUTE_RETURN_ON_ERROR(NECopyKernel::validate(input, &output_subtensor_info));
            break;
        }
        case PaddingMode::REFLECT:
        case PaddingMode::SYMMETRIC:
        {
            for(uint32_t i = 0; i < padding.size(); ++i)
            {
                if(mode == PaddingMode::REFLECT)
                {
                    ARM_COMPUTE_RETURN_ERROR_ON(padding[i].first >= input->dimension(i));
                    ARM_COMPUTE_RETURN_ERROR_ON(padding[i].second >= input->dimension(i));
                }
                else
                {
                    ARM_COMPUTE_RETURN_ERROR_ON(padding[i].first > input->dimension(i));
                    ARM_COMPUTE_RETURN_ERROR_ON(padding[i].second > input->dimension(i));
                }
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid mode");
        }
    }
    return Status{};
}

void NEPadLayer::run()
{
    if(_num_dimensions > 0)
    {
        switch(_mode)
        {
            case PaddingMode::CONSTANT:
            {
                NEScheduler::get().schedule(&_memset_kernel, Window::DimY);
                NEScheduler::get().schedule(&_copy_kernel, Window::DimY);
                break;
            }
            case PaddingMode::REFLECT:
            case PaddingMode::SYMMETRIC:
            {
                for(uint32_t i = 0; i < _num_dimensions; ++i)
                {
                    if(_padding[i].first > 0 || _padding[i].second > 0)
                    {
                        if(_padding[i].first > 0 && _slice_results[2 * i].info()->total_size() > 0)
                        {
                            _slice_functions[2 * i].run();
                        }
                        if(_padding[i].second > 0 && _slice_results[2 * i + 1].info()->total_size() > 0)
                        {
                            _slice_functions[2 * i + 1].run();
                        }
                        _concat_functions[i].run();
                    }
                }
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Padding mode not supported.");
        }
    }
    else
    {
        NEScheduler::get().schedule(&_copy_kernel, Window::DimY);
    }
}
} // namespace arm_compute
