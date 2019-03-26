/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEFFTDigitReverseKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, unsigned int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(idx, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON(axis != 0);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *idx, unsigned int axis)
{
    ARM_COMPUTE_UNUSED(idx, axis);

    auto_init_if_empty(*output, *input);

    Window win = calculate_max_window(*output, Steps());
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

NEFFTDigitReverseKernel::NEFFTDigitReverseKernel()
    : _input(nullptr), _output(nullptr), _idx(nullptr), _axis(0)
{
}

void NEFFTDigitReverseKernel::configure(const ITensor *input, ITensor *output, const ITensor *idx, unsigned int axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, idx);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), idx->info(), axis));

    _input  = input;
    _output = output;
    _idx    = idx;
    _axis   = axis;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), idx->info(), axis);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEFFTDigitReverseKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, unsigned int axis)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, idx, axis));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), idx->clone().get(), axis).first);
    return Status{};
}

void NEFFTDigitReverseKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    Iterator     out(_output, window);
    const size_t element_size = _input->info()->element_size();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        unsigned int in_index_1d = *reinterpret_cast<unsigned int *>(_idx->ptr_to_element(Coordinates(id.x())));

        auto reverse_id = id;
        reverse_id.set(_axis, in_index_1d);

        memcpy(out.ptr(), _input->ptr_to_element(reverse_id), 2 * element_size);

    },
    out);

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
}
} // namespace arm_compute
