/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/core/NEON/kernels/NEStackLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/Utils.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input,
                          uint32_t           axis,
                          uint32_t           idx_input,
                          uint32_t           num_tensors,
                          uint32_t           rank,
                          const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use CPU FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(idx_input >= num_tensors);
    ARM_COMPUTE_RETURN_ERROR_ON(axis > input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != rank);

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(),
                                                           compute_stack_shape(*input, axis, num_tensors));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

inline Coordinates
shift_from_axis_and_replace_coordinate(const Coordinates &id, uint32_t axis, uint32_t idx_input, uint32_t num_dims)
{
    Coordinates id_out = id;
    for (uint32_t i = num_dims; i > axis; --i)
    {
        id_out.set(i, id[i - 1]);
    }
    id_out.set(axis, idx_input);
    return id_out;
}

void elementwise_stack(const std::vector<ITensor *> &input, ITensor *output, uint32_t axis, const Window &window)
{
    Window window_out;
    window_out.use_tensor_dimensions(output->info()->tensor_shape());

    const int32_t  num_tensors  = input.size();
    const size_t   element_size = input[0]->info()->element_size();
    const uint32_t num_dims     = static_cast<uint32_t>(input[0]->info()->num_dimensions());

    for (int32_t idx_input = 0; idx_input < num_tensors; ++idx_input)
    {
        Iterator input_it(input[idx_input], window);

        execute_window_loop(
            window,
            [&](const Coordinates &id)
            {
                Coordinates id_out = shift_from_axis_and_replace_coordinate(id, axis, idx_input, num_dims);
                std::memcpy(output->ptr_to_element(id_out), input_it.ptr(), element_size);
            },
            input_it);
    }
}

void memcpy_stack(const std::vector<ITensor *> &input, ITensor *output, uint32_t axis, const Window &window)
{
    const int32_t element_size   = input[0]->info()->element_size();
    const int32_t chunk_size     = input[0]->info()->tensor_shape().total_size_lower(axis) * element_size;
    const int32_t num_tensors    = input.size();
    const int32_t out_chunk_step = chunk_size * num_tensors;

    const int32_t start_x = window.x().start();
    const int32_t end_x   = window.x().end();
    const int32_t start_y = window.y().start();
    const int32_t end_y   = window.y().end();

    uint8_t *out_ptr_base = output->buffer() + output->info()->offset_first_element_in_bytes() + start_x * chunk_size;

    for (int32_t x = start_x; x < end_x; ++x)
    {
        const uint8_t *in_ptr =
            input[x]->buffer() + input[x]->info()->offset_first_element_in_bytes() + start_y * chunk_size;
        uint8_t *out_ptr = out_ptr_base + start_y * out_chunk_step;

        for (int32_t y = start_y; y < end_y; ++y)
        {
            std::memcpy(out_ptr, in_ptr, chunk_size);

            in_ptr += chunk_size;
            out_ptr += out_chunk_step;
        }

        out_ptr_base += chunk_size;
    }
}

} // namespace

NEStackLayerKernel::NEStackLayerKernel() : _input(), _output(nullptr), _axis(), _split_dimension(Window::DimY)
{
}

void NEStackLayerKernel::configure(const std::vector<ITensor *> &input, uint32_t axis, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    const int32_t num_tensors = input.size();
    ARM_COMPUTE_ERROR_ON(num_tensors == 0);

    const uint32_t rank = input[0]->info()->num_dimensions();
    ARM_COMPUTE_UNUSED(rank);

    for (int32_t i = 0; i < num_tensors; ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input[i]);
        ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input[i]->info(), axis, i, num_tensors, rank, output->info()));
    }

    auto_init_if_empty(*output->info(), input[0]->info()->clone()->set_tensor_shape(
                                            compute_stack_shape(*input[0]->info(), axis, num_tensors)));

    _input  = input;
    _output = output;
    _axis   = axis;
}

Status NEStackLayerKernel::validate(const std::vector<ITensorInfo *> &input, uint32_t axis, const ITensorInfo *output)
{
    const int32_t num_tensors = input.size();
    const size_t  rank        = input[0]->num_dimensions();

    for (int32_t i = 0; i < num_tensors; ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input[i]);
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input[i], axis, i, num_tensors, rank, output));
    }

    return Status{};
}

void NEStackLayerKernel::prepare()
{
    // Prepare calculates the window at runtime, in case there is padding being added after configure()
    const ITensorInfo *input_info  = _input[0]->info();
    const int32_t      num_dims    = input_info->num_dimensions();
    const int32_t      num_tensors = _input.size();

    // Check if there are any paddings in the input tensors
    bool has_padding = false;
    for (const ITensor *in : _input)
    {
        if (has_holes(*in->info(), num_dims - 1))
        {
            has_padding = true;
            break;
        }
    }

    has_padding = has_padding || has_holes(*_output->info(), num_dims);

    Window win;
    if (!has_padding)
    {
        _stack_fn = memcpy_stack;

        // 2D execution window (X,Y): [Num_tensors, Dimensions >= axis]
        win.set(Window::DimX, Window::Dimension(0, num_tensors, 1));
        win.set(Window::DimY, Window::Dimension(0, input_info->tensor_shape().total_size_upper(_axis), 1));
    }
    else
    {
        _stack_fn = elementwise_stack;
        win       = calculate_max_window(*input_info);
    }

    INEKernel::configure(win);
}

void NEStackLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    _stack_fn(_input, _output, _axis, window);
}
} // namespace arm_compute
