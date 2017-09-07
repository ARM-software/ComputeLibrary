/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEAbsoluteDifferenceKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
void abs_diff_U8_U8_U8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t input1_val = vld1q_u8(input1.ptr());
        const uint8x16_t input2_val = vld1q_u8(input2.ptr());

        vst1q_u8(output.ptr(), vabdq_u8(input1_val, input2_val));
    },
    input1, input2, output);
}

inline int16x8x2_t vqabd2q_s16(const int16x8x2_t &v1, const int16x8x2_t &v2)
{
    const int16x8x2_t res =
    {
        {
            vqabsq_s16(vqsubq_s16(v1.val[0], v2.val[0])),
            vqabsq_s16(vqsubq_s16(v1.val[1], v2.val[1]))
        }
    };

    return res;
}

void abs_diff_S16_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        int16x8x2_t input1_val = vld2q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        int16x8x2_t input2_val = vld2q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));
        vst2q_s16(reinterpret_cast<int16_t *>(output.ptr()), vqabd2q_s16(input1_val, input2_val));
    },
    input1, input2, output);
}

void abs_diff_U8_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t  input1_val = vld1q_u8(input1.ptr());
        const int16x8x2_t input2_val =
        {
            {
                vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr())),
                vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr()) + 8)
            }
        };

        const int16x8x2_t out_val =
        {
            {
                vqabsq_s16(vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input1_val))), input2_val.val[0])),
                vqabsq_s16(vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input1_val))), input2_val.val[1]))
            }
        };

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), out_val.val[0]);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, out_val.val[1]);

    },
    input1, input2, output);
}

void abs_diff_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    abs_diff_U8_S16_S16(in2, in1, out, window);
}
} // namespace

NEAbsoluteDifferenceKernel::NEAbsoluteDifferenceKernel()
    : _func(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void NEAbsoluteDifferenceKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    set_shape_if_empty(*output->info(), input1->info()->tensor_shape());

    if(input1->info()->data_type() == DataType::S16 || input2->info()->data_type() == DataType::S16)
    {
        set_format_if_unknown(*output->info(), Format::S16);
    }
    else if(input1->info()->data_type() == DataType::F32 || input2->info()->data_type() == DataType::F32)
    {
        set_format_if_unknown(*output->info(), Format::U8);
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input1, input2, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->data_type() == DataType::U8 && (input1->info()->data_type() != DataType::U8 || input2->info()->data_type() != DataType::U8),
                             "The output image can only be U8 if both input images are U8");

    _input1 = input1;
    _input2 = input2;
    _output = output;

    const DataType input1_data_type = input1->info()->data_type();
    const DataType input2_data_type = input2->info()->data_type();

    if(input1_data_type == input2_data_type)
    {
        if(input1_data_type == DataType::U8)
        {
            _func = &abs_diff_U8_U8_U8;
        }
        else
        {
            _func = &abs_diff_S16_S16_S16;
        }
    }
    else
    {
        if(input1_data_type == DataType::U8)
        {
            _func = &abs_diff_U8_S16_S16;
        }
        else
        {
            _func = &abs_diff_S16_U8_S16;
        }
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window                 win = calculate_max_window(*input1->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input1->info(), 0, num_elems_processed_per_iteration),
                              AccessWindowHorizontal(input2->info(), 0, num_elems_processed_per_iteration),
                              output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->info()->valid_region(),
                                                       input2->info()->valid_region());

    output_access.set_valid_region(win, valid_region);

    INEKernel::configure(win);
}

void NEAbsoluteDifferenceKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    _func(_input1, _input2, _output, window);
}
