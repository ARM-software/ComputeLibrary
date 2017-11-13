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
#include "arm_compute/core/NEON/kernels/NEAccumulateKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

/* Max S16 value used for saturation purposes. */
const static uint16x8_t max_int_u16 = vdupq_n_u16(static_cast<uint16_t>(INT16_MAX));

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace fp16
{
inline float16x8x2_t convert_u8x16_to_f16x8x2(uint8x16_t input)
{
    const float16x8x2_t out =
    {
        {
            vcvtq_f16_u16(vmovl_u8(vget_low_u8(input))),
            vcvtq_f16_u16(vmovl_u8(vget_high_u8(input)))
        }
    };

    return out;
}

inline uint8x16_t convert_f16x8x2_to_u8x16(const float16x8x2_t &input)
{
    return vcombine_u8(vmovn_u16(vcvtq_u16_f16(input.val[0])),
                       vmovn_u16(vcvtq_u16_f16(input.val[1])));
}

inline float16x8x2_t vector_accumulate_weighted(const float16x8x2_t &vec0, const float16x8x2_t &vec1, float16x8_t scale_val, float16x8_t scale_val2)
{
    const float16x8x2_t res =
    {
        {
            vfmaq_f16(vmulq_f16(vec1.val[0], scale_val), vec0.val[0], scale_val2),
            vfmaq_f16(vmulq_f16(vec1.val[1], scale_val), vec0.val[1], scale_val2)
        }
    };

    return res;
}

void acc_we_v16_u8(const void *__restrict input, void *__restrict accum, float16x8_t scale_val, float16x8_t scale_val2)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == accum);

    const auto input_ptr = static_cast<const uint8_t *__restrict>(input);
    const auto accum_ptr = static_cast<uint8_t *__restrict>(accum);

    const uint8x16x4_t input_buffer = vld4q_u8(input_ptr);
    uint8x16x4_t       accum_buffer = vld4q_u8(accum_ptr);

    const float16x8x2_t f16_input_0 = convert_u8x16_to_f16x8x2(input_buffer.val[0]);
    const float16x8x2_t f16_input_1 = convert_u8x16_to_f16x8x2(input_buffer.val[1]);
    const float16x8x2_t f16_input_2 = convert_u8x16_to_f16x8x2(input_buffer.val[2]);
    const float16x8x2_t f16_input_3 = convert_u8x16_to_f16x8x2(input_buffer.val[3]);

    float16x8x2_t f16_accum_0 = convert_u8x16_to_f16x8x2(accum_buffer.val[0]);
    float16x8x2_t f16_accum_1 = convert_u8x16_to_f16x8x2(accum_buffer.val[1]);
    float16x8x2_t f16_accum_2 = convert_u8x16_to_f16x8x2(accum_buffer.val[2]);
    float16x8x2_t f16_accum_3 = convert_u8x16_to_f16x8x2(accum_buffer.val[3]);

    f16_accum_0 = vector_accumulate_weighted(f16_input_0, f16_accum_0, scale_val, scale_val2);
    f16_accum_1 = vector_accumulate_weighted(f16_input_1, f16_accum_1, scale_val, scale_val2);
    f16_accum_2 = vector_accumulate_weighted(f16_input_2, f16_accum_2, scale_val, scale_val2);
    f16_accum_3 = vector_accumulate_weighted(f16_input_3, f16_accum_3, scale_val, scale_val2);

    accum_buffer = { {
            convert_f16x8x2_to_u8x16(f16_accum_0),
            convert_f16x8x2_to_u8x16(f16_accum_1),
            convert_f16x8x2_to_u8x16(f16_accum_2),
            convert_f16x8x2_to_u8x16(f16_accum_3)
        }
    };

    vst4q_u8(accum_ptr, accum_buffer);
}
} // namespace fp16

void NEAccumulateWeightedFP16Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator accum(_output, window);

    const float16x8_t scale_val  = vdupq_n_f16(1.f - _alpha);
    const float16x8_t scale_val2 = vdupq_n_f16(_alpha);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        fp16::acc_we_v16_u8(input.ptr(), accum.ptr(), scale_val, scale_val2);
    },
    input, accum);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

namespace
{
inline void acc_v16_u8(const void *__restrict input, void *__restrict accum)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == accum);

    const auto in  = static_cast<const uint8_t *__restrict>(input);
    const auto out = static_cast<int16_t *__restrict>(accum);

    uint8x16_t ta1 = vld1q_u8(in);
    int16x8_t  ta2 = vld1q_s16(out);
    int16x8_t  ta3 = vld1q_s16(out + 8);

    ta2 = vqaddq_s16(ta2, vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ta1))));
    ta3 = vqaddq_s16(ta3, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ta1))));

    vst1q_s16(out, ta2);
    vst1q_s16(out + 8, ta3);
}

inline float32x4x4_t convert_u8x16_to_f32x4x4(uint8x16_t input)
{
    const uint16x8_t u16_output_low = vmovl_u8(vget_low_u8(input));
    const uint16x8_t u16_output_hi  = vmovl_u8(vget_high_u8(input));

    const float32x4x4_t res =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_output_low))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_output_low))),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_output_hi))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_output_hi)))
        }
    };

    return res;
}

inline uint8x16_t convert_f32x4x4_to_u8x16(const float32x4x4_t &input)
{
    return vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(input.val[0])),
                                              vmovn_u32(vcvtq_u32_f32(input.val[1])))),
                       vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(input.val[2])),
                                              vmovn_u32(vcvtq_u32_f32(input.val[3])))));
}

inline float32x4x4_t vector_accumulate_weighted(const float32x4x4_t &vector_input, float32x4x4_t vector_output, float32x4_t scale_val, float32x4_t scale_val2)
{
    vector_output.val[0] = vmulq_f32(vector_output.val[0], scale_val);
    vector_output.val[1] = vmulq_f32(vector_output.val[1], scale_val);
    vector_output.val[2] = vmulq_f32(vector_output.val[2], scale_val);
    vector_output.val[3] = vmulq_f32(vector_output.val[3], scale_val);

    vector_output.val[0] = vmlaq_f32(vector_output.val[0], vector_input.val[0], scale_val2);
    vector_output.val[1] = vmlaq_f32(vector_output.val[1], vector_input.val[1], scale_val2);
    vector_output.val[2] = vmlaq_f32(vector_output.val[2], vector_input.val[2], scale_val2);
    vector_output.val[3] = vmlaq_f32(vector_output.val[3], vector_input.val[3], scale_val2);

    return vector_output;
}

inline void acc_we_v16_u8(const void *__restrict input, void *__restrict accum, const float32x4_t scale_val, const float32x4_t scale_val2)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == accum);

    const auto input_ptr = static_cast<const uint8_t *__restrict>(input);
    const auto accum_ptr = static_cast<uint8_t *__restrict>(accum);

    const uint8x16_t input_buffer = vld1q_u8(input_ptr);
    const uint8x16_t accum_buffer = vld1q_u8(accum_ptr);

    const float32x4x4_t f32_input_0  = convert_u8x16_to_f32x4x4(input_buffer);
    const float32x4x4_t f32_output_0 = convert_u8x16_to_f32x4x4(accum_buffer);

    const float32x4x4_t f32_res_0 = vector_accumulate_weighted(f32_input_0, f32_output_0, scale_val, scale_val2);

    vst1q_u8(accum_ptr, convert_f32x4x4_to_u8x16(f32_res_0));
}

void acc_sq_v16_u8(const void *__restrict input, uint32_t shift, void *__restrict accum)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == accum);
    ARM_COMPUTE_ERROR_ON(shift > 15);

    const auto input_buffer = static_cast<const uint8_t *__restrict>(input);
    const auto accum_buffer = static_cast<int16_t *__restrict>(accum);

    const uint8x16_t ta1 = vld1q_u8(input_buffer);
    uint16x8_t       ta2 = vreinterpretq_u16_s16(vld1q_s16(accum_buffer));
    uint16x8_t       ta3 = vreinterpretq_u16_s16(vld1q_s16(accum_buffer + 8));

    const int16x8_t vector_shift = vdupq_n_s16(-static_cast<int16_t>(shift));

    uint16x8_t linput = vmovl_u8(vget_low_u8(ta1));
    uint16x8_t hinput = vmovl_u8(vget_high_u8(ta1));

    linput = vmulq_u16(linput, linput);
    hinput = vmulq_u16(hinput, hinput);

    linput = vqshlq_u16(linput, vector_shift);
    hinput = vqshlq_u16(hinput, vector_shift);

    ta2 = vqaddq_u16(ta2, linput);
    ta3 = vqaddq_u16(ta3, hinput);

    vst1q_s16(accum_buffer, vreinterpretq_s16_u16(vminq_u16(max_int_u16, ta2)));
    vst1q_s16(accum_buffer + 8, vreinterpretq_s16_u16(vminq_u16(max_int_u16, ta3)));
}
} // namespace

void NEAccumulateKernel::configure(const ITensor *input, ITensor *accum)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, accum);

    set_shape_if_empty(*accum->info(), input->info()->tensor_shape());

    set_format_if_unknown(*accum->info(), Format::S16);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, accum);

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    INESimpleKernel::configure(input, accum, num_elems_processed_per_iteration);
}

void NEAccumulateKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    Iterator input(_input, window);
    Iterator accum(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        acc_v16_u8(input.ptr(), accum.ptr());
    },
    input, accum);
}

NEAccumulateWeightedKernel::NEAccumulateWeightedKernel()
    : _alpha(0.0f)
{
}

void NEAccumulateWeightedKernel::configure(const ITensor *input, float alpha, ITensor *accum)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, accum);

    set_shape_if_empty(*accum->info(), input->info()->tensor_shape());

    set_format_if_unknown(*accum->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, accum);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(alpha < 0.0 || alpha > 1.0);

    _alpha = alpha;

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    INESimpleKernel::configure(input, accum, num_elems_processed_per_iteration);
}

void NEAccumulateWeightedKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator accum(_output, window);

    const float32x4_t scale_val  = vdupq_n_f32(1.f - _alpha);
    const float32x4_t scale_val2 = vdupq_n_f32(_alpha);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        acc_we_v16_u8(input.ptr(), accum.ptr(), scale_val, scale_val2);
    },
    input, accum);
}

NEAccumulateSquaredKernel::NEAccumulateSquaredKernel()
    : _shift(0)
{
}

void NEAccumulateSquaredKernel::configure(const ITensor *input, uint32_t shift, ITensor *accum)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, accum);

    set_shape_if_empty(*accum->info(), input->info()->tensor_shape());

    set_format_if_unknown(*accum->info(), Format::S16);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, accum);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(shift > 15);

    _shift = shift;

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    INESimpleKernel::configure(input, accum, num_elems_processed_per_iteration);
}

void NEAccumulateSquaredKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    Iterator input(_input, window);
    Iterator accum(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        acc_sq_v16_u8(input.ptr(), _shift, accum.ptr());
    },
    input, accum);
}
