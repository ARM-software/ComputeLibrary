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
#include "arm_compute/core/NEON/kernels/NECopyKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding = PaddingList())
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > 4);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, *input);
    return std::make_pair(Status{}, calculate_max_window(*output));
}

std::pair<Status, Window> validate_and_configure_window_with_padding(ITensorInfo *input, ITensorInfo *output, const PaddingList &padding)
{
    const TensorShape input_shape  = input->tensor_shape();
    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input_shape, padding);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(padded_shape));
    // Configure window
    const Window win = calculate_max_window(*output, output->dimension(0));
    return std::make_pair(Status{}, win);
}

} // namespace

NECopyKernel::NECopyKernel()
    : _input(nullptr), _output(nullptr), _padding()
{
}

void NECopyKernel::configure(const ITensor *input, ITensor *output, const PaddingList &padding)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding));

    _input   = input;
    _output  = output;
    _padding = padding;

    std::pair<Status, Window> win_config;

    if(padding.empty())
    {
        win_config = validate_and_configure_window(input->info(), output->info());
    }
    else
    {
        win_config = validate_and_configure_window_with_padding(input->info(), output->info(), padding);
    }

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NECopyKernel::validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output, const PaddingList &padding)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding));

    if(padding.empty())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_with_padding(input->clone().get(), output->clone().get(), padding).first);
    }

    return Status{};
}

void NECopyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_padding.empty())
    {
        Window output_window{ window };
        output_window.set(Window::DimX, Window::Dimension(output_window.x().start(), output_window.x().end(), _input->info()->dimension(0)));
        Window out_slice = output_window.first_slice_window_1D();
        do
        {
            Iterator input_it(_input, out_slice);
            Iterator output_it(_output, out_slice);

            execute_window_loop(out_slice, [&](const Coordinates &)
            {
                memcpy(output_it.ptr(), input_it.ptr(), _output->info()->dimension(0) * _output->info()->element_size());
            },
            input_it, output_it);
        }
        while(output_window.slide_window_slice_1D(out_slice));
    }
    else
    {
        Window input_window{ window };
        input_window.set(Window::DimX, Window::Dimension(0, window.x().end() - _padding[0].first, _input->info()->dimension(0)));

        Iterator     input_it(_input, input_window);
        Iterator     output_it(_output, window);
        const size_t row_size_in_bytes = _input->info()->dimension(0) * _input->info()->element_size();
        execute_window_loop(window, [&](const Coordinates &)
        {
            auto dst_ptr = output_it.ptr() + _padding[0].first * _output->info()->element_size();
            std::memcpy(dst_ptr, input_it.ptr(), row_size_in_bytes);
        },
        input_it, output_it);
    }
}
} // namespace arm_compute
