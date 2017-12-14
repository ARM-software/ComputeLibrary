/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEL2NormalizeLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cmath>

using namespace arm_compute;

namespace
{
void l2_normalize_X(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window in_slice  = window.first_slice_window_1D();
    Window sum_slice = window_sum.first_slice_window_1D();

    do
    {
        Iterator input_it(in, in_slice);
        Iterator sum_it(sum, sum_slice);
        Iterator output_it(out, in_slice);

        const float       sum_value           = *reinterpret_cast<const float *>(sum_it.ptr());
        const float32x4_t vec_normalize_value = vdupq_n_f32(1.f / std::sqrt(std::max(sum_value, epsilon)));

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const float *>(input_it.ptr());
            const auto out_ptr = reinterpret_cast<float *>(output_it.ptr());

            vst1q_f32(out_ptr, vmulq_f32(vld1q_f32(in_ptr), vec_normalize_value));
        },
        input_it, output_it);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
} // namespace

NEL2NormalizeLayerKernel::NEL2NormalizeLayerKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr), _axis(0), _epsilon(1e-12)
{
}

void NEL2NormalizeLayerKernel::configure(const ITensor *input, const ITensor *sum, ITensor *output, unsigned int axis, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Normalization axis greater than max number of dimensions");
    ARM_COMPUTE_ERROR_ON_MSG(axis > 0, "Unsupported normalization axis, Supported axis is 0");

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

    unsigned int num_elems_processed_per_iteration     = 16 / data_size_from_type(input->info()->data_type());
    unsigned int num_elems_processed_per_iteration_sum = (axis == 0) ? 1 : num_elems_processed_per_iteration;

    _input   = input;
    _sum     = sum;
    _output  = output;
    _axis    = axis;
    _epsilon = epsilon;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal sum_access(sum->info(), 0, num_elems_processed_per_iteration_sum);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, sum_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEL2NormalizeLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_axis)
    {
        case 0:
            l2_normalize_X(_input, _sum, _output, _epsilon, window);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported normalization axis");
    }
}
