/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEMinMaxLayerKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <algorithm>
#include <arm_neon.h>
#include <climits>
#include <cstddef>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() < 3);

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        TensorShape output_shape = compute_min_max_shape(input);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    TensorShape output_shape = compute_min_max_shape(input);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, output_shape, 1, input->data_type());

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, 2);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win);
}
} // namespace

NEMinMaxLayerKernel::NEMinMaxLayerKernel()
    : _input(nullptr), _output(nullptr), _mtx()
{
}

void NEMinMaxLayerKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    auto win_config = validate_and_configure_window(input->info(), output->info());

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEMinMaxLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

    return Status{};
}

void NEMinMaxLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int x_start = window.x().start();
    const int x_end   = window.x().end();

    Window window_output;
    window_output.use_tensor_dimensions(_output->info()->tensor_shape());
    window_output.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Handle X dimension manually to split into two loops
    // First one will use vector operations, second one processes the left over pixels
    Window window_input(window);
    window_input.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_input.set(3, Window::Dimension(0, 1, 1));

    Iterator input(_input, window_input);
    Iterator output(_output, window_output);

    execute_window_loop(window_output, [&](const Coordinates & id_batch)
    {
        float32x2_t carry_min = vdup_n_f32(std::numeric_limits<float>::max());
        float32x2_t carry_max = vdup_n_f32(std::numeric_limits<float>::lowest());

        float carry_min_scalar = std::numeric_limits<float>::max();
        float carry_max_scalar = std::numeric_limits<float>::lowest();

        execute_window_loop(window_input, [&](const Coordinates &)
        {
            int        x      = x_start;
            const auto in_ptr = reinterpret_cast<const float *>(input.ptr() + id_batch[1] * _input->info()->strides_in_bytes()[3]);

            // Vector loop
            for(; x <= x_end - 8; x += 8)
            {
                const float32x4x2_t pixels   = vld2q_f32(in_ptr + x);
                const float32x4_t   tmp_min1 = vminq_f32(pixels.val[0], pixels.val[1]);
                const float32x4_t   tmp_max1 = vmaxq_f32(pixels.val[0], pixels.val[1]);
                const float32x2_t   tmp_min2 = vmin_f32(vget_high_f32(tmp_min1), vget_low_f32(tmp_min1));
                const float32x2_t   tmp_max2 = vmax_f32(vget_high_f32(tmp_max1), vget_low_f32(tmp_max1));
                carry_min                    = vmin_f32(tmp_min2, carry_min);
                carry_max                    = vmax_f32(tmp_max2, carry_max);
            }

            // Process leftover pixels
            for(; x < x_end; ++x)
            {
                const float pixel = in_ptr[x];
                carry_min_scalar  = std::min(pixel, carry_min_scalar);
                carry_max_scalar  = std::max(pixel, carry_max_scalar);
            }
        },
        input);

        // Reduce result
        carry_min = vpmin_f32(carry_min, carry_min);
        carry_max = vpmax_f32(carry_max, carry_max);
        carry_min = vpmin_f32(carry_min, carry_min);
        carry_max = vpmax_f32(carry_max, carry_max);

        // Extract max/min values
        const float min_i = std::min(vget_lane_f32(carry_min, 0), carry_min_scalar);
        const float max_i = std::max(vget_lane_f32(carry_max, 0), carry_max_scalar);

        auto out_ptr = reinterpret_cast<float *>(output.ptr());

        // Perform reduction of local min/max values
        update_min_max(out_ptr, min_i, max_i);
    },
    output);
}

void NEMinMaxLayerKernel::reset()
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    float32x2_t reset_values = vdup_n_f32(0.0f);
    reset_values             = vset_lane_f32(std::numeric_limits<float>::max(), reset_values, 0);
    reset_values             = vset_lane_f32(std::numeric_limits<float>::lowest(), reset_values, 1);

    Window window_output;
    window_output.use_tensor_dimensions(_output->info()->tensor_shape());
    window_output.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator output(_output, window_output);

    execute_window_loop(window_output, [&](const Coordinates &)
    {
        vst1_f32(reinterpret_cast<float *>(output.ptr()), reset_values);
    },
    output);
}

void NEMinMaxLayerKernel::update_min_max(float *out_ptr, float min, float max)
{
    std::lock_guard<Mutex> lock(_mtx);

    const float32x2_t old_min = vld1_dup_f32(out_ptr);
    const float32x2_t old_max = vld1_dup_f32(out_ptr + 1);
    const float32x2_t new_min = vmin_f32(vdup_n_f32(min), old_min);
    const float32x2_t new_max = vmax_f32(vdup_n_f32(max), old_max);

    vst1_f32(out_ptr, vzip_f32(new_min, new_max).val[0]);
}
} // namespace arm_compute
