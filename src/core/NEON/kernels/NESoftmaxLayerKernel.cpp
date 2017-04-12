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
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <cfloat>

using namespace arm_compute;

NELogits1DMaxKernel::NELogits1DMaxKernel()
    : _border_size()
{
}

BorderSize NELogits1DMaxKernel::border_size() const
{
    return _border_size;
}

void NELogits1DMaxKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    const int              input_width                       = input->info()->valid_region().shape.x();
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    _input       = input;
    _output      = output;
    _border_size = BorderSize(0, input_width % num_elems_processed_per_iteration, 0, 0);

    // Configure kernel window
    constexpr unsigned int num_elems_written_per_row = 1;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_row, 1.f / input_width);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NELogits1DMaxKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window in_slice = window.first_slice_window_1D();

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window max_slice = window_max.first_slice_window_1D();

    do
    {
        Iterator in(_input, in_slice);
        Iterator out(_output, max_slice);

        float32x4_t vec_max = vdupq_n_f32(-FLT_MAX);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto        in_ptr        = reinterpret_cast<const float *>(in.ptr());
            const float32x4_t current_value = vld1q_f32(in_ptr);
            vec_max                         = vmaxq_f32(vec_max, current_value);
        },
        in);

        float32x2_t carry_max = vpmax_f32(vget_high_f32(vec_max), vget_low_f32(vec_max));
        carry_max             = vpmax_f32(carry_max, carry_max);

        *(reinterpret_cast<float *>(out.ptr())) = vget_lane_f32(carry_max, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}

NELogits1DShiftExpSumKernel::NELogits1DShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr), _border_size(0)
{
}

BorderSize NELogits1DShiftExpSumKernel::border_size() const
{
    return _border_size;
}
void NELogits1DShiftExpSumKernel::configure(const ITensor *input, const ITensor *max, ITensor *output, ITensor *sum)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(max, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, max);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(max, sum);

    const int              input_width                       = input->info()->valid_region().shape.x();
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    _input       = input;
    _max         = max;
    _output      = output;
    _sum         = sum;
    _border_size = BorderSize(0, input_width % num_elems_processed_per_iteration, 0, 0);

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal max_access(max->info(), 0, 1);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal sum_access(sum->info(), 0, 1);

    update_window_and_padding(win, input_access, max_access, output_access, sum_access);

    output_access.set_valid_region(win, input->info()->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NELogits1DShiftExpSumKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window max_slice = window_max.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    do
    {
        Iterator in(_input, in_slice);
        Iterator exp(_output, in_slice);
        Iterator max(_max, max_slice);
        Iterator sum(_sum, max_slice);

        float32x4_t vec_sum_value = vdupq_n_f32(0.0f);

        const auto  max_ptr = reinterpret_cast<const float *>(max.ptr());
        float32x4_t vec_max = vdupq_n_f32(*max_ptr);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const float *>(in.ptr());
            const auto exp_ptr = reinterpret_cast<float *>(exp.ptr());

            float32x4_t vec_elements = vld1q_f32(in_ptr);
            vec_elements             = vsubq_f32(vec_elements, vec_max);
            vec_elements             = vexp_f32(vec_elements);

            vst1q_f32(exp_ptr, vec_elements);

            vec_sum_value = vaddq_f32(vec_elements, vec_sum_value);
        },
        in, exp);

        float32x2_t carry_addition = vpadd_f32(vget_high_f32(vec_sum_value), vget_low_f32(vec_sum_value));
        carry_addition             = vpadd_f32(carry_addition, carry_addition);

        *(reinterpret_cast<float *>(sum.ptr())) = vget_lane_f32(carry_addition, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}

NELogits1DNormKernel::NELogits1DNormKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void NELogits1DNormKernel::configure(const ITensor *input, const ITensor *sum, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, sum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

    _input  = input;
    _sum    = sum;
    _output = output;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     sum_access(sum->info(), 0, 0, 1, sum->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, sum_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NELogits1DNormKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window sum_slice = window_sum.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    do
    {
        Iterator in(_input, in_slice);
        Iterator sum(_sum, sum_slice);
        Iterator out(_output, in_slice);

        float             sum_value        = *reinterpret_cast<const float *>(sum.ptr());
        const float32x4_t vec_sum_inversed = vdupq_n_f32(1.0f / sum_value);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const float *>(in.ptr());
            const auto out_ptr = reinterpret_cast<float *>(out.ptr());

            const float32x4_t vec_in           = vld1q_f32(in_ptr);
            const float32x4_t normalized_value = vmulq_f32(vec_in, vec_sum_inversed);

            vst1q_f32(out_ptr, normalized_value);
        },
        in, out);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
