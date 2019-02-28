/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NETileKernel.h"

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
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(multiples.size() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(multiples.empty());
    ARM_COMPUTE_RETURN_ERROR_ON(std::any_of(multiples.begin(), multiples.end(), [](uint32_t e)
    {
        return e == 0;
    }));

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(misc::shape_calculator::compute_tiled_shape(input->tensor_shape(), multiples), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NETileKernel::NETileKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NETileKernel::configure(const ITensor *input, ITensor *output, const Multiples &multiples)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output
    TensorShape tiled_shape = misc::shape_calculator::compute_tiled_shape(input->info()->tensor_shape(), multiples);
    auto_init_if_empty(*output->info(), tiled_shape, 1, input->info()->data_type());

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), multiples));

    _input  = input;
    _output = output;

    // Configure window without padding
    Window win = calculate_max_window(*output->info());
    INEKernel::configure(win);
}

Status NETileKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, multiples));
    return Status{};
}

void NETileKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window output_window{ window };
    output_window.set(Window::DimX, Window::Dimension(output_window.x().start(), output_window.x().end(), _input->info()->dimension(0)));
    Window out_slice = output_window.first_slice_window_1D();

    const auto src_shape = _input->info()->tensor_shape();
    do
    {
        Iterator output_it(_output, out_slice);

        execute_window_loop(out_slice, [&](const Coordinates & id)
        {
            const size_t x = id.x();
            const size_t y = id.y();
            const size_t z = id.z();
            const size_t w = id[3];
            Coordinates  input_coords{ x % src_shape[0], y % src_shape[1], z % src_shape[2], w % src_shape[3] };
            memcpy(output_it.ptr(), _input->ptr_to_element(input_coords), _input->info()->dimension(0) * _input->info()->element_size());
        },
        output_it);
    }
    while(output_window.slide_window_slice_1D(out_slice));
}
} // namespace arm_compute
