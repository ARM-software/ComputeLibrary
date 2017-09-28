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
#include "arm_compute/core/NEON/kernels/NEQuantizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;

NEQuantizationLayerKernel::NEQuantizationLayerKernel()
    : _input(nullptr), _output(nullptr), _min_max(nullptr)
{
}

void NEQuantizationLayerKernel::configure(const ITensor *input, ITensor *output, const ITensor *min_max)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() < 3);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, DataType::U8, 0);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

    _input   = input;
    _output  = output;
    _min_max = min_max;

    constexpr unsigned int num_elems_processed_per_iteration = 8;

    // Configure window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     min_max_access(min_max->info(), 0, 0, 2, min_max->info()->dimension(1));

    // Update window and padding
    update_window_and_padding(win, input_access, output_access, min_max_access);
    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEQuantizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_input_output(window);
    window_input_output.collapse_if_possible(INEKernel::window(), 3);
    window_input_output.set(3, Window::Dimension(0, 1, 1));

    Window window_min_max;
    window_min_max.use_tensor_dimensions(_min_max->info()->tensor_shape());
    window_min_max.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_min_max.collapse_if_possible(INEKernel::window(), 1);

    Iterator input(_input, window_input_output);
    Iterator output(_output, window_input_output);
    Iterator min_max(_min_max, window_min_max);

    execute_window_loop(window_min_max, [&](const Coordinates & id_batch)
    {
        // Get the min and max
        float min = *(reinterpret_cast<const float *>(min_max.ptr()) + 0);
        float max = *(reinterpret_cast<const float *>(min_max.ptr()) + 1);

        // Saturate the result if min = max
        if(min == max)
        {
            min = 0.0f;
            max = 1.0f;
        }

        const float32x4_t vmin             = vdupq_n_f32(min);
        const float32x4_t inv_range        = vdupq_n_f32(1.0f / (max - min));
        const float32x4_t quantization_max = vdupq_n_f32(255.0f);
        const float32x4_t quantization_mul = vdupq_n_f32(256.0f);

        // Uniformly map values to range 8bit integers, i.e. [min, max] -> [0, 255]
        execute_window_loop(window_input_output, [&](const Coordinates & id)
        {
            // Get the input values
            const auto    input_ptr = reinterpret_cast<const float *>(input.ptr() + id_batch[1] * _input->info()->strides_in_bytes()[3]);
            float32x4x2_t val       = vld2q_f32(input_ptr);

            // Map float values to range [0.0, 1.0]
            val.val[0] = vsubq_f32(val.val[0], vmin);
            val.val[1] = vsubq_f32(val.val[1], vmin);
            val.val[0] = vmulq_f32(val.val[0], inv_range);
            val.val[1] = vmulq_f32(val.val[1], inv_range);

            // Quantize
            val.val[0] = vmulq_f32(val.val[0], quantization_mul);
            val.val[1] = vmulq_f32(val.val[1], quantization_mul);
            val.val[0] = vminq_f32(val.val[0], quantization_max);
            val.val[1] = vminq_f32(val.val[1], quantization_max);

            const uint32x4_t   val_u32_low  = vcvtq_u32_f32(val.val[0]);
            const uint32x4_t   val_u32_high = vcvtq_u32_f32(val.val[1]);
            const uint16x4x2_t val_u16      = vzip_u16(vmovn_u32(val_u32_low), vmovn_u32(val_u32_high));

            const uint8x8_t quantized = vmovn_u16(vcombine_u16(val_u16.val[0], val_u16.val[1]));

            // Store the quantized values
            auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr() + id_batch[1] * _output->info()->strides_in_bytes()[3]);
            vst1_u8(output_ptr, quantized);
        },
        input, output);
    },
    min_max);
}
