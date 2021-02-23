/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(idx_input >= num_tensors);
    ARM_COMPUTE_RETURN_ERROR_ON(axis > input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_stack_shape(*input, axis, num_tensors));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int axis, unsigned int num_tensors, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_stack_shape(*input, axis, num_tensors)));

    // Configure kernel window
    Window win = calculate_max_window(*input);

    return std::make_pair(Status{}, win);
}

inline Coordinates shift_from_axis_and_replace_coordinate(const Coordinates &id, unsigned int axis, unsigned int idx_input)
{
    constexpr int max_out_coord = 5; // Input shape is max a 4D shape, output is max 5D
    Coordinates   id_out        = id;
    for(unsigned int i = max_out_coord - 1; i > axis; --i)
    {
        id_out.set(i, id[i - 1]);
    }
    id_out.set(axis, idx_input);
    return id_out;
}
} // namespace

NEStackLayerKernel::NEStackLayerKernel()
    : _input(nullptr), _output(nullptr), _axis(), _idx_input()
{
}

void NEStackLayerKernel::configure(const ITensor *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), axis, idx_input, num_tensors, output->info()));

    _input     = input;
    _output    = output;
    _axis      = axis;
    _idx_input = idx_input;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), axis, num_tensors, output->info());

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEStackLayerKernel::validate(const ITensorInfo *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, axis, idx_input, num_tensors, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), axis, num_tensors, output->clone().get()).first);
    return Status{};
}

void NEStackLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_out;
    window_out.use_tensor_dimensions(_output->info()->tensor_shape());

    Iterator input(_input, window);
    Iterator output(_output, window_out);

    const int stride_x = _output->info()->strides_in_bytes()[0];
    const int stride_y = _output->info()->num_dimensions() >= 1 ? _output->info()->strides_in_bytes()[1] : 0;
    const int stride_z = _output->info()->num_dimensions() >= 2 ? _output->info()->strides_in_bytes()[2] : 0;
    const int stride_w = _output->info()->num_dimensions() >= 3 ? _output->info()->strides_in_bytes()[3] : 0;
    const int stride_k = _output->info()->num_dimensions() >= 4 ? _output->info()->strides_in_bytes()[4] : 0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        Coordinates id_out = shift_from_axis_and_replace_coordinate(id, _axis, _idx_input);
        const int   idx    = id_out[0] * stride_x + id_out[1] * stride_y + id_out[2] * stride_z + id_out[3] * stride_w + id_out[4] * stride_k;
        std::memcpy(output.ptr() + idx, input.ptr(), _input->info()->element_size());
    },
    input);
}
