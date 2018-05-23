/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDequantizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *min_max)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, min_max);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() < 3);

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *min_max)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, DataType::F32, 0);

    constexpr unsigned int num_elems_processed_per_iteration = 8;

    // Configure window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     min_max_access(min_max, 0, 0, 2, min_max->dimension(1));

    // Update window and padding
    bool window_changed = update_window_and_padding(win, input_access, output_access, min_max_access);

    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win);
}
} // namespace

NEDequantizationLayerKernel::NEDequantizationLayerKernel()
    : _input(nullptr), _output(nullptr), _min_max(nullptr)
{
}

void NEDequantizationLayerKernel::configure(const ITensor *input, ITensor *output, const ITensor *min_max)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, min_max);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), min_max->info()));

    _input   = input;
    _output  = output;
    _min_max = min_max;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), min_max->info());

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEDequantizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *min_max)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, min_max));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), min_max->clone().get())));

    return Status{};
}

void NEDequantizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_input_output(window);
    window_input_output.set(3, Window::Dimension(0, 1, 1));

    Window window_min_max;
    window_min_max.use_tensor_dimensions(_min_max->info()->tensor_shape());
    window_min_max.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, window_input_output);
    Iterator output(_output, window_input_output);
    Iterator min_max(_min_max, window_min_max);

    execute_window_loop(window_min_max, [&](const Coordinates & id_batch)
    {
        // Get the min and max
        const float min = *(reinterpret_cast<const float *>(min_max.ptr()) + 0);
        const float max = *(reinterpret_cast<const float *>(min_max.ptr()) + 1);

        const float32x4_t vmin    = vdupq_n_f32(min);
        const float       range   = max - min;
        const float32x4_t scaling = vdupq_n_f32(range / 255.0f);

        // Uniformly map values to range 8bit integers, i.e. [min, max] -> [0, 255]
        execute_window_loop(window_input_output, [&](const Coordinates & id)
        {
            // Get the input values
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr() + id_batch[1] * _input->info()->strides_in_bytes()[3]);

            const uint8x8_t  val_u8       = vld1_u8(input_ptr);
            const uint16x8_t val_u16      = vmovl_u8(val_u8);
            const uint32x4_t val_u32_low  = vmovl_u16(vget_low_u16(val_u16));
            const uint32x4_t val_u32_high = vmovl_u16(vget_high_u16(val_u16));
            float32x4_t      val_low      = vcvtq_f32_u32(val_u32_low);
            float32x4_t      val_high     = vcvtq_f32_u32(val_u32_high);

            // Dequantize -> (q / 255.0 * range) + min
            val_low  = vmulq_f32(val_low, scaling);
            val_high = vmulq_f32(val_high, scaling);
            val_low  = vaddq_f32(val_low, vmin);
            val_high = vaddq_f32(val_high, vmin);

            const float32x4x2_t dequantized = vuzpq_f32(val_low, val_high);

            // Store the dequantized values
            auto output_ptr = reinterpret_cast<float *>(output.ptr() + id_batch[1] * _output->info()->strides_in_bytes()[3]);
            vst2q_f32(output_ptr, dequantized);
        },
        input, output);
    },
    min_max);
}