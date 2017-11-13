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
#include "arm_compute/core/NEON/kernels/NENonMaximaSuppression3x3Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace fp16
{
inline void mask_top(const float16x8_t &vc, const float16x8_t &in0, const float16x8_t &in1, uint16x8_t &mask)
{
    // vc > nc.val[0], vc > nc.val[1], vc > nc.val[2]
    mask = vandq_u16(mask, vcgeq_f16(vc, in0));
    mask = vandq_u16(mask, vcgeq_f16(vc, vextq_f16(in0, in1, 1)));
    mask = vandq_u16(mask, vcgeq_f16(vc, vextq_f16(in0, in1, 2)));
}

inline void mask_middle(const float16x8_t &vc, const float16x8_t &in0, const float16x8_t &in1, uint16x8_t &mask)
{
    // vc >= nc.val[0], vc > nc.val[2]
    mask = vandq_u16(mask, vcgeq_f16(vc, in0));
    mask = vandq_u16(mask, vcgtq_f16(vc, vextq_f16(in0, in1, 2)));
}

inline void mask_bottom(const float16x8_t &vc, const float16x8_t &in0, const float16x8_t &in1, uint16x8_t &mask)
{
    // vc > nc.val[0], vc > nc.val[1], vc > nc.val[2]
    mask = vandq_u16(mask, vcgtq_f16(vc, in0));
    mask = vandq_u16(mask, vcgtq_f16(vc, vextq_f16(in0, in1, 1)));
    mask = vandq_u16(mask, vcgtq_f16(vc, vextq_f16(in0, in1, 2)));
}

inline void non_maxima_suppression3x3_F32_F32(const void *__restrict in_ptr, void *__restrict out_ptr, const uint32_t in_stride)
{
    auto       in  = static_cast<const float *__restrict>(in_ptr) - 1;
    const auto out = static_cast<float *__restrict>(out_ptr);

    // Get centre scores
    const float16x8x2_t vc =
    {
        vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 1)), vcvt_f16_f32(vld1q_f32(in + 5))),
        vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 9)), vcvt_f16_f32(vld1q_f32(in + 13)))
    };

    // Neighboring pixels
    in -= in_stride;

    static const float16x4_t  zero_f16x4 = vdup_n_f16(0);
    static const uint16x8_t   zero_u16   = vdupq_n_u16(0);
    static const uint16x8_t   true_mask  = vceqq_u16(zero_u16, zero_u16);
    static const uint16x8x2_t true_mask_x2 =
    {
        true_mask,
        true_mask
    };

    uint16x8x2_t mask = true_mask_x2;

    // Top row
    const float16x8_t tmp_top0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in)), vcvt_f16_f32(vld1q_f32(in + 4)));
    const float16x8_t tmp_top1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 8)), vcvt_f16_f32(vld1q_f32(in + 12)));
    const float16x8_t tmp_top2 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 16)), zero_f16x4);

    // vc >= nc.val[0], vc >= nc.val[1], vc >= nc.val[2]
    mask_top(vc.val[0], tmp_top0, tmp_top1, mask.val[0]);
    mask_top(vc.val[1], tmp_top1, tmp_top2, mask.val[1]);

    in += in_stride;

    // Middle row
    const float16x8_t tmp_mid0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in)), vcvt_f16_f32(vld1q_f32(in + 4)));
    const float16x8_t tmp_mid1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 8)), vcvt_f16_f32(vld1q_f32(in + 12)));
    const float16x8_t tmp_mid2 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 16)), zero_f16x4);

    // vc >= nc.val[0], vc > nc.val[2]
    mask_middle(vc.val[0], tmp_mid0, tmp_mid1, mask.val[0]);
    mask_middle(vc.val[1], tmp_mid1, tmp_mid2, mask.val[1]);

    in += in_stride;

    // Bottom row
    const float16x8_t tmp_bot0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in)), vcvt_f16_f32(vld1q_f32(in + 4)));
    const float16x8_t tmp_bot1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 8)), vcvt_f16_f32(vld1q_f32(in + 12)));
    const float16x8_t tmp_bot2 = vcombine_f16(vcvt_f16_f32(vld1q_f32(in + 16)), zero_f16x4);

    // vc > nc.val[0], vc > nc.val[1], vc > nc.val[2]
    mask_bottom(vc.val[0], tmp_bot0, tmp_bot1, mask.val[0]);
    mask_bottom(vc.val[1], tmp_bot1, tmp_bot2, mask.val[1]);

    // Store
    static const float16x8_t zero_f16x8 = vdupq_n_f16(0);

    const float16x8_t suppressed0 = vbslq_f16(mask.val[0], vc.val[0], zero_f16x8);
    vst1q_f32(out + 0, vcvt_f32_f16(vget_low_f16(suppressed0)));
    vst1q_f32(out + 4, vcvt_f32_f16(vget_high_f16(suppressed0)));

    const float16x8_t suppressed1 = vbslq_f16(mask.val[1], vc.val[1], zero_f16x8);
    vst1q_f32(out + 8, vcvt_f32_f16(vget_low_f16(suppressed1)));
    vst1q_f32(out + 12, vcvt_f32_f16(vget_high_f16(suppressed1)));
}

inline void non_maxima_suppression3x3_U8_U8(const void *__restrict in_ptr, void *__restrict out_ptr, const uint32_t in_stride)
{
    auto       in  = static_cast<const uint8_t *__restrict>(in_ptr) - 1;
    const auto out = static_cast<uint8_t *__restrict>(out_ptr);

    // Get centre scores
    const uint8x16_t vc = vld1q_u8(in + 1);

    // Neighboring pixels
    in -= in_stride;

    // Top row
    const uint8x16_t l_nc_0 = vld1q_u8(in);
    const uint8x16_t m_nc_0 = vld1q_u8(in + 1);
    const uint8x16_t r_nc_0 = vld1q_u8(in + 2);

    // Keep center scores if ...
    // vc >= l_nc_0, vc >= m_nc_0, vc >= r_nc_0
    uint8x16_t mask = vcgeq_u8(vc, l_nc_0);
    mask            = vandq_u8(mask, vcgeq_u8(vc, m_nc_0));
    mask            = vandq_u8(mask, vcgeq_u8(vc, r_nc_0));

    in += in_stride;

    // Middle row
    const uint8x16_t l_nc_1 = vld1q_u8(in);
    const uint8x16_t r_nc_1 = vld1q_u8(in + 2);

    // ... and ...
    // vc >= l_nc_1, vc > r_nc_1
    mask = vandq_u8(mask, vcgeq_u8(vc, l_nc_1));
    mask = vandq_u8(mask, vcgtq_u8(vc, r_nc_1));

    in += in_stride;

    // Bottom row
    const uint8x16_t l_nc_2 = vld1q_u8(in);
    const uint8x16_t m_nc_2 = vld1q_u8(in + 1);
    const uint8x16_t r_nc_2 = vld1q_u8(in + 2);

    // ... and ...
    // vc > l_nc_2, vc > m_nc_2, vc > r_nc_2
    mask = vandq_u8(mask, vcgtq_u8(vc, l_nc_2));
    mask = vandq_u8(mask, vcgtq_u8(vc, m_nc_2));
    mask = vandq_u8(mask, vcgtq_u8(vc, r_nc_2));

    // Store
    static const uint8x16_t zero = vdupq_n_u8(0);
    vst1q_u8(out, vbslq_u8(mask, vc, zero));
}
} // namespace fp16

void NENonMaximaSuppression3x3FP16Kernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    switch(input->info()->data_type())
    {
        case DataType::U8:
            _func = &fp16::non_maxima_suppression3x3_U8_U8;
            break;
        default:
            _func = &fp16::non_maxima_suppression3x3_F32_F32;
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    const unsigned int     num_elems_read_per_iteration      = 16 + 2 * border_size().left + (input->info()->data_type() == DataType::U8 ? 0 : 3);
    constexpr unsigned int num_elems_written_per_iteration   = 16;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

namespace
{
inline void non_maxima_suppression3x3_FLOAT_FLOAT(const void *__restrict input_ptr, void *__restrict output_ptr, const uint32_t input_stride)
{
    auto       input  = static_cast<const float *__restrict>(input_ptr) - 1;
    const auto output = static_cast<float *__restrict>(output_ptr);

    // Get centre scores
    const float32x4x4_t vc =
    {
        {
            vld1q_f32(input + 1),
            vld1q_f32(input + 5),
            vld1q_f32(input + 9),
            vld1q_f32(input + 13)
        }
    };

    // Neighboring pixels
    float32x4x4_t l_nc{ {} };
    float32x4x4_t m_nc{ {} };
    float32x4x4_t r_nc{ {} };

    input -= input_stride;

    // Row0 - Low part
    float32x4_t tmp_low   = vld1q_f32(input);
    float32x4_t tmp_high  = vld1q_f32(input + 4);
    float32x4_t tmp_high1 = vld1q_f32(input + 8);

    l_nc.val[0] = tmp_low;
    m_nc.val[0] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[0] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[1] = tmp_low;
    m_nc.val[1] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[1] = vextq_f32(tmp_low, tmp_high, 2);

    // Row0 - High part
    tmp_low   = tmp_high1;
    tmp_high  = vld1q_f32(input + 12);
    tmp_high1 = vld1q_f32(input + 16);

    l_nc.val[2] = tmp_low;
    m_nc.val[2] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[2] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[3] = tmp_low;
    m_nc.val[3] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[3] = vextq_f32(tmp_low, tmp_high, 2);

    // mc >= nc.val[0], mc >= nc.val[1], mc >= nc.val[2]
    uint32x4x4_t mask{ {} };
    mask.val[0] = vcgeq_f32(vc.val[0], l_nc.val[0]);
    mask.val[0] = vandq_u32(mask.val[0], vcgeq_f32(vc.val[0], m_nc.val[0]));
    mask.val[0] = vandq_u32(mask.val[0], vcgeq_f32(vc.val[0], r_nc.val[0]));
    mask.val[1] = vcgeq_f32(vc.val[1], l_nc.val[1]);
    mask.val[1] = vandq_u32(mask.val[1], vcgeq_f32(vc.val[1], m_nc.val[1]));
    mask.val[1] = vandq_u32(mask.val[1], vcgeq_f32(vc.val[1], r_nc.val[1]));
    mask.val[2] = vcgeq_f32(vc.val[2], l_nc.val[2]);
    mask.val[2] = vandq_u32(mask.val[2], vcgeq_f32(vc.val[2], m_nc.val[2]));
    mask.val[2] = vandq_u32(mask.val[2], vcgeq_f32(vc.val[2], r_nc.val[2]));
    mask.val[3] = vcgeq_f32(vc.val[3], l_nc.val[3]);
    mask.val[3] = vandq_u32(mask.val[3], vcgeq_f32(vc.val[3], m_nc.val[3]));
    mask.val[3] = vandq_u32(mask.val[3], vcgeq_f32(vc.val[3], r_nc.val[3]));

    input += input_stride;

    // Row1 - Low part
    tmp_low   = vld1q_f32(input);
    tmp_high  = vld1q_f32(input + 4);
    tmp_high1 = vld1q_f32(input + 8);

    l_nc.val[0] = tmp_low;
    r_nc.val[0] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[1] = tmp_low;
    r_nc.val[1] = vextq_f32(tmp_low, tmp_high, 2);

    // Row1 - High part
    tmp_low   = tmp_high1;
    tmp_high  = vld1q_f32(input + 12);
    tmp_high1 = vld1q_f32(input + 16);

    l_nc.val[2] = tmp_low;
    r_nc.val[2] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[3] = tmp_low;
    r_nc.val[3] = vextq_f32(tmp_low, tmp_high, 2);

    // mc >= nc.val[0], mc > nc.val[2]
    mask.val[0] = vandq_u32(mask.val[0], vcgeq_f32(vc.val[0], l_nc.val[0]));
    mask.val[0] = vandq_u32(mask.val[0], vcgtq_f32(vc.val[0], r_nc.val[0]));
    mask.val[1] = vandq_u32(mask.val[1], vcgeq_f32(vc.val[1], l_nc.val[1]));
    mask.val[1] = vandq_u32(mask.val[1], vcgtq_f32(vc.val[1], r_nc.val[1]));
    mask.val[2] = vandq_u32(mask.val[2], vcgeq_f32(vc.val[2], l_nc.val[2]));
    mask.val[2] = vandq_u32(mask.val[2], vcgtq_f32(vc.val[2], r_nc.val[2]));
    mask.val[3] = vandq_u32(mask.val[3], vcgeq_f32(vc.val[3], l_nc.val[3]));
    mask.val[3] = vandq_u32(mask.val[3], vcgtq_f32(vc.val[3], r_nc.val[3]));

    input += input_stride;

    // Row2 - Low part
    tmp_low   = vld1q_f32(input);
    tmp_high  = vld1q_f32(input + 4);
    tmp_high1 = vld1q_f32(input + 8);

    l_nc.val[0] = tmp_low;
    m_nc.val[0] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[0] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[1] = tmp_low;
    m_nc.val[1] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[1] = vextq_f32(tmp_low, tmp_high, 2);

    // Row2 - High part
    tmp_low   = tmp_high1;
    tmp_high  = vld1q_f32(input + 12);
    tmp_high1 = vld1q_f32(input + 16);

    l_nc.val[2] = tmp_low;
    m_nc.val[2] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[2] = vextq_f32(tmp_low, tmp_high, 2);

    tmp_low  = tmp_high;
    tmp_high = tmp_high1;

    l_nc.val[3] = tmp_low;
    m_nc.val[3] = vextq_f32(tmp_low, tmp_high, 1);
    r_nc.val[3] = vextq_f32(tmp_low, tmp_high, 2);

    // mc > nc.val[0], mc > nc.val[1], mc > nc.val[2]
    mask.val[0] = vandq_u32(mask.val[0], vcgtq_f32(vc.val[0], l_nc.val[0]));
    mask.val[0] = vandq_u32(mask.val[0], vcgtq_f32(vc.val[0], m_nc.val[0]));
    mask.val[0] = vandq_u32(mask.val[0], vcgtq_f32(vc.val[0], r_nc.val[0]));
    mask.val[1] = vandq_u32(mask.val[1], vcgtq_f32(vc.val[1], l_nc.val[1]));
    mask.val[1] = vandq_u32(mask.val[1], vcgtq_f32(vc.val[1], m_nc.val[1]));
    mask.val[1] = vandq_u32(mask.val[1], vcgtq_f32(vc.val[1], r_nc.val[1]));
    mask.val[2] = vandq_u32(mask.val[2], vcgtq_f32(vc.val[2], l_nc.val[2]));
    mask.val[2] = vandq_u32(mask.val[2], vcgtq_f32(vc.val[2], m_nc.val[2]));
    mask.val[2] = vandq_u32(mask.val[2], vcgtq_f32(vc.val[2], r_nc.val[2]));
    mask.val[3] = vandq_u32(mask.val[3], vcgtq_f32(vc.val[3], l_nc.val[3]));
    mask.val[3] = vandq_u32(mask.val[3], vcgtq_f32(vc.val[3], m_nc.val[3]));
    mask.val[3] = vandq_u32(mask.val[3], vcgtq_f32(vc.val[3], r_nc.val[3]));

    static const float32x4_t zero = vdupq_n_f32(0.f);

    // Store
    vst1q_f32(output + 0, vbslq_f32(mask.val[0], vc.val[0], zero));
    vst1q_f32(output + 4, vbslq_f32(mask.val[1], vc.val[1], zero));
    vst1q_f32(output + 8, vbslq_f32(mask.val[2], vc.val[2], zero));
    vst1q_f32(output + 12, vbslq_f32(mask.val[3], vc.val[3], zero));
}

inline void non_maxima_suppression3x3_U8_U8(const void *__restrict input_ptr, void *__restrict output_ptr, const uint32_t input_stride)
{
    auto       input  = static_cast<const uint8_t *__restrict>(input_ptr) - 1;
    const auto output = static_cast<uint8_t *__restrict>(output_ptr);

    // Get centre scores
    const uint8x16_t vc = vld1q_u8(input + 1);

    // Neighboring pixels
    uint8x16_t l_nc{};
    uint8x16_t m_nc{};
    uint8x16_t r_nc{};

    input -= input_stride;

    // Row0
    l_nc = vld1q_u8(input);
    m_nc = vld1q_u8(input + 1);
    r_nc = vld1q_u8(input + 2);

    // mc >= l_nc, mc >= m_nc, mc >= r_nc
    uint8x16_t mask = vcgeq_u8(vc, l_nc);
    mask            = vandq_u8(mask, vcgeq_u8(vc, m_nc));
    mask            = vandq_u8(mask, vcgeq_u8(vc, r_nc));

    input += input_stride;

    // Row1
    l_nc = vld1q_u8(input);
    r_nc = vld1q_u8(input + 2);

    // mc >= l_nc, mc > r_nc
    mask = vandq_u8(mask, vcgeq_u8(vc, l_nc));
    mask = vandq_u8(mask, vcgtq_u8(vc, r_nc));

    input += input_stride;

    // Row2
    l_nc = vld1q_u8(input);
    m_nc = vld1q_u8(input + 1);
    r_nc = vld1q_u8(input + 2);

    // mc > l_nc, mc > m_nc, mc > r_nc
    mask = vandq_u8(mask, vcgtq_u8(vc, l_nc));
    mask = vandq_u8(mask, vcgtq_u8(vc, m_nc));
    mask = vandq_u8(mask, vcgtq_u8(vc, r_nc));

    static const uint8x16_t zero = vdupq_n_u8(0);

    // Store
    vst1q_u8(output, vbslq_u8(mask, vc, zero));
}
} // namespace

NENonMaximaSuppression3x3Kernel::NENonMaximaSuppression3x3Kernel()
    : _func(nullptr), _input(nullptr), _output(nullptr)
{
}

BorderSize NENonMaximaSuppression3x3Kernel::border_size() const
{
    return BorderSize(1);
}

void NENonMaximaSuppression3x3Kernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    if(input->info()->data_type() == DataType::U8)
    {
        _func = &non_maxima_suppression3x3_U8_U8;
    }
    else
    {
        _func = &non_maxima_suppression3x3_FLOAT_FLOAT;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    const unsigned int     num_elems_read_per_iteration      = 16 + 2 * border_size().left + (input->info()->data_type() == DataType::U8 ? 0 : 3);
    constexpr unsigned int num_elems_written_per_iteration   = 16;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NENonMaximaSuppression3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    Iterator input(_input, window);
    Iterator output(_output, window);

    const size_t input_stride = _input->info()->strides_in_bytes()[1] / element_size_from_data_type(_input->info()->data_type());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        _func(input.ptr(), output.ptr(), input_stride);
    },
    input, output);
}
