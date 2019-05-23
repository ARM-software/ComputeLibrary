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
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLPadLayer::CLPadLayer()
    : _copy_kernel(), _mode(), _padding(), _memset_kernel(), _num_dimensions(0), _slice_functions(), _concat_functions(), _slice_results(), _concat_results()
{
}

void CLPadLayer::configure_constant_mode(ICLTensor *input, ICLTensor *output, const PaddingList &padding, const PixelValue constant_value)
{
    // Set the pages of the output to the constant_value.
    _memset_kernel.configure(output, constant_value);

    // Fill out padding list with zeroes.
    PaddingList padding_extended = padding;
    for(size_t i = padding.size(); i < TensorShape::num_max_dimensions; i++)
    {
        padding_extended.emplace_back(PaddingInfo{ 0, 0 });
    }

    // Create a window within the output tensor where the input will be copied.
    Window copy_window = Window();
    for(uint32_t i = 0; i < output->info()->num_dimensions(); ++i)
    {
        copy_window.set(i, Window::Dimension(padding_extended[i].first, padding_extended[i].first + input->info()->dimension(i), 1));
    }
    // Copy the input to the output, leaving the padding filled with the constant_value.
    _copy_kernel.configure(input, output, PaddingList(), &copy_window);
}

void CLPadLayer::configure_reflect_symmetric_mode(ICLTensor *input, ICLTensor *output)
{
    int64_t last_padding_dimension = _padding.size() - 1;
    // Reflecting can be performed by effectively unfolding the input as follows:
    // For each dimension starting at DimX:
    //      Create a before and after slice, which values depend on the selected padding mode
    //      Concatenate the before and after padding with the tensor to be padded

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
    ICLTensor *prev = input;
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
            std::vector<ICLTensor *> concat_vector;
            if(_padding[i].first > 0)
            {
                if(i < prev->info()->num_dimensions())
                {
                    _slice_functions[2 * i].configure(prev, &_slice_results[2 * i], starts_before, ends_before, strides, begin_mask_before, end_mask_before);
                    concat_vector.push_back(&_slice_results[2 * i]);
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
                    concat_vector.push_back(&_slice_results[2 * i + 1]);
                }
                else
                {
                    // Performing the slice is unnecessary if the result would simply be a copy of the tensor.
                    concat_vector.push_back(prev);
                }
            }
            // Concatenate the padding before and after with the input.
            ICLTensor *out = (static_cast<int32_t>(i) == last_padding_dimension) ? output : &_concat_results[i];
            _concat_functions[i].configure(concat_vector, out, i);
            prev = out;
        }
    }
    for(uint32_t i = 0; i < _num_dimensions; ++i)
    {
        if((static_cast<int32_t>(i) != last_padding_dimension))
        {
            _concat_results[i].allocator()->allocate();
        }
        _slice_results[2 * i].allocator()->allocate();
        _slice_results[2 * i + 1].allocator()->allocate();
    }
}

void CLPadLayer::configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), padding, constant_value, mode));

    _padding = padding;
    _mode    = mode;

    TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->info()->tensor_shape(), _padding);

    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(padded_shape));

    // Find the last dimension requiring padding so that it is known when to write to output and whether any padding is applied.
    int64_t last_padding_dimension = _padding.size() - 1;
    for(; last_padding_dimension >= 0; --last_padding_dimension)
    {
        if(_padding[last_padding_dimension].first > 0 || _padding[last_padding_dimension].second > 0)
        {
            break;
        }
    }
    _num_dimensions = last_padding_dimension + 1;
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

Status CLPadLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > input->num_dimensions());

    TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);

    // Use CLCopyKernel and CLMemsetKernel to validate all padding modes as this includes all of the shape and info validation.
    PaddingList padding_extended = padding;
    for(size_t i = padding.size(); i < TensorShape::num_max_dimensions; i++)
    {
        padding_extended.emplace_back(PaddingInfo{ 0, 0 });
    }

    Window copy_window = Window();
    for(uint32_t i = 0; i < padded_shape.num_dimensions(); ++i)
    {
        copy_window.set(i, Window::Dimension(padding_extended[i].first, padding_extended[i].first + input->dimension(i), 1));
    }
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), padded_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(output, input);
        ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(input, output, PaddingList(), &copy_window));
        ARM_COMPUTE_RETURN_ON_ERROR(CLMemsetKernel::validate(output, constant_value));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(input, &input->clone()->set_tensor_shape(padded_shape), PaddingList(), &copy_window));
        ARM_COMPUTE_RETURN_ON_ERROR(CLMemsetKernel::validate(&input->clone()->set_tensor_shape(padded_shape), constant_value));
    }

    switch(mode)
    {
        case PaddingMode::CONSTANT:
        {
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

void CLPadLayer::run()
{
    if(_num_dimensions > 0)
    {
        switch(_mode)
        {
            case PaddingMode::CONSTANT:
            {
                CLScheduler::get().enqueue(_memset_kernel, false);
                CLScheduler::get().enqueue(_copy_kernel, true);
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
                        CLScheduler::get().sync();
                        _concat_functions[i].run();
                        CLScheduler::get().sync();
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
        CLScheduler::get().enqueue(_copy_kernel, true);
    }
}
} // namespace arm_compute
