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
#ifndef SRC_CORE_NEON_KERNELS_CROP_IMPL_H
#define SRC_CORE_NEON_KERNELS_CROP_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/common/Registrars.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/crop/generic/neon/crop_helper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void in_bounds_crop_window(const ITensor *input,
                           const ITensor *output,
                           float         *output_ptr,
                           Coordinates    input_offset,
                           int32_t        window_step_x,
                           int32_t        output_width_start,
                           int32_t        output_width_limit,
                           bool           input_has_single_channel,
                           bool           is_width_flipped)
{
    // Reverse elements if width flipped.
    if (is_width_flipped)
    {
        // Collapse first dimension if possible.
        if (input_has_single_channel)
        {
            int32_t     x = output_width_start;
            Coordinates negative_offset(input_offset);
            negative_offset.set(1, negative_offset[1] - window_step_x + 1);
            for (; x <= output_width_limit - window_step_x; x += window_step_x, negative_offset[1] -= window_step_x)
            {
                auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(negative_offset)));

                in = wrapper::vrev64(in);
                in = wrapper::vcombine(wrapper::vgethigh(in), wrapper::vgetlow(in));

                wrapper::vstore(output_ptr + x, in);
            }
            input_offset[1] = negative_offset[1] + window_step_x - 1;
            for (; x < output_width_limit; ++x, --input_offset[1])
            {
                *(output_ptr + x) = static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
            }
        }
        else
        {
            for (int32_t x = output_width_start; x < output_width_limit; ++x, --input_offset[1])
            {
                input_offset.set(0, 0);
                int32_t c = 0;
                for (; c <= static_cast<int32_t>(input->info()->dimension(0)) - window_step_x;
                     c += window_step_x, input_offset[0] += window_step_x)
                {
                    auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                    wrapper::vstore(output_ptr + x * output->info()->dimension(0) + c, in);
                }
                for (; c < static_cast<int32_t>(input->info()->dimension(0)); ++c, ++input_offset[0])
                {
                    *(output_ptr + x * output->info()->dimension(0) + c) =
                        static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                }
            }
        }
    }
    else
    {
        // Use memcpy if the elements don't need converting to float.
        if (std::is_same<T, float>::value)
        {
            memcpy(static_cast<void *>(output_ptr + output_width_start * output->info()->dimension(0)),
                   reinterpret_cast<const void *>(input->ptr_to_element(input_offset)),
                   (output_width_limit - output_width_start) * output->info()->dimension(0) *
                       output->info()->element_size());
        }
        else
        {
            int32_t x = 0;
            int32_t limit =
                (output_width_limit - output_width_start) * static_cast<int32_t>(output->info()->dimension(0));
            float *output_start_ptr = output_ptr + output_width_start * output->info()->dimension(0);
            for (; x <= limit - window_step_x; x += window_step_x, input_offset[0] += window_step_x)
            {
                auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                wrapper::vstore(output_start_ptr + x, in);
            }
            for (; x < limit; ++x, ++input_offset[0])
            {
                *(output_start_ptr + x) =
                    static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
            }
        }
    }
}
} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_CROP_IMPL_H
