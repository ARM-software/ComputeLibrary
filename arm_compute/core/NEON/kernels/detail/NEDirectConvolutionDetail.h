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

#ifndef __ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H__
#define __ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H__

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace detail
{
/** Loads a 3x3 matrix as a row  (float).
 *
 * @param[in] ptr            Pointer to a float 3x3 matrix.
 * @param[in] weights_offset (Optional) Weights quantization offset.
 *
 * @return The loaded matrix.
 */
inline float32x4x3_t load_matrix_row(const float *ptr, int weights_offset = 0)
{
    ARM_COMPUTE_UNUSED(weights_offset);
    const float32x4x3_t r =
    {
        {
            vld1q_dup_f32(ptr),
            vld1q_dup_f32(1 + ptr),
            vld1q_dup_f32(2 + ptr)
        }
    };
    return r;
}

/** Loads a 3x3 matrix as a row  (uint8_t).
 *
 * @param[in] ptr            Pointer to a uint8_t 3x3 matrix.
 * @param[in] weights_offset (Optional) Weights quantization offset.
 *
 * @return The loaded matrix.
 */
inline int32x4x3_t load_matrix_row(const uint8_t *ptr, int weights_offset = 0)
{
    const int32x4_t v_weights_offset = vdupq_n_s32(weights_offset);

    /* ptr is a pointer to a row in a 3x3 matrix, the function returns 3 vectors holding exactly the same value in all lanes:
       r.val[0] contains the first element, r.val[1] the second element and r.val[2] the third element (in all lanes) */
    int32x4x3_t r =
    {
        {
            vaddq_s32(v_weights_offset, vdupq_n_s32(*ptr)),
            vaddq_s32(v_weights_offset, vdupq_n_s32(*(ptr + 1))),
            vaddq_s32(v_weights_offset, vdupq_n_s32(*(ptr + 2)))
        }
    };
    return r;
}

/** Perform a 3x3 convolution for 4 consecutive elements on float32 when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
inline float32x4_t single_convolve_3x3_dilation(const float *in_top, const float *in_mid, const float *in_low,
                                                const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                                const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    const float32x4x3_t vtop =
    {
        {
            vld1q_f32(in_top),
            vld1q_f32(in_top + dilation_x),
            vld1q_f32(in_top + 2 * dilation_x)
        }
    };
    const float32x4x3_t vmid =
    {
        {
            vld1q_f32(in_mid),
            vld1q_f32(in_mid + dilation_x),
            vld1q_f32(in_mid + 2 * dilation_x)
        }
    };
    const float32x4x3_t vlow =
    {
        {
            vld1q_f32(in_low),
            vld1q_f32(in_low + dilation_x),
            vld1q_f32(in_low + 2 * dilation_x)
        }
    };
    float32x4_t out = vmulq_f32(vtop.val[0], m0.val[0]);
    out             = vmlaq_f32(out, vtop.val[1], m0.val[1]);
    out             = vmlaq_f32(out, vtop.val[2], m0.val[2]);

    out = vmlaq_f32(out, vmid.val[0], m1.val[0]);
    out = vmlaq_f32(out, vmid.val[1], m1.val[1]);
    out = vmlaq_f32(out, vmid.val[2], m1.val[2]);

    out = vmlaq_f32(out, vlow.val[0], m2.val[0]);
    out = vmlaq_f32(out, vlow.val[1], m2.val[1]);
    out = vmlaq_f32(out, vlow.val[2], m2.val[2]);

    return out;
}

/** Perform a 3x3 convolution for 8 consecutive elements on float32 when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
template <unsigned int stridex>
float32x4x2_t convolve_3x3_dilation(const float *in_top, const float *in_mid, const float *in_low,
                                    const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                    const size_t dilation_x, int input_offset = 0);

template <>
inline float32x4x2_t convolve_3x3_dilation<1>(const float *in_top, const float *in_mid, const float *in_low,
                                              const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    const float32x4x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 4, in_mid + 4, in_low + 4, m0, m1, m2, dilation_x, input_offset)
        }
    };

    return out;
}

template <>
inline float32x4x2_t convolve_3x3_dilation<2>(const float *in_top, const float *in_mid, const float *in_low,
                                              const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    float32x4x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline float32x4x2_t convolve_3x3_dilation<3>(const float *in_top, const float *in_mid, const float *in_low,
                                              const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    float32x4x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);
    ;
    out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    return out;
}

/** Perform a convolve3x3 on float32.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
template <unsigned int stridex>
float32x4x2_t convolve_3x3(const float *in_top, const float *in_mid, const float *in_low,
                           const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                           int input_offset = 0);

template <>
inline float32x4x2_t convolve_3x3<1>(const float *in_top, const float *in_mid, const float *in_low,
                                     const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    const float32x4x3_t vtop =
    {
        {
            vld1q_f32(in_top),
            vld1q_f32(in_top + 4),
            vld1q_f32(in_top + 8)
        }
    };
    const float32x4x3_t vmid =
    {
        {
            vld1q_f32(in_mid),
            vld1q_f32(in_mid + 4),
            vld1q_f32(in_mid + 8)
        }
    };
    const float32x4x3_t vlow =
    {
        {
            vld1q_f32(in_low),
            vld1q_f32(in_low + 4),
            vld1q_f32(in_low + 8)
        }
    };
    float32x4x2_t out =
    {
        {
            vmulq_f32(vtop.val[0], m0.val[0]),
            vmulq_f32(vtop.val[1], m0.val[0])
        }
    };
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vtop.val[0], vtop.val[1], 1), m0.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vtop.val[0], vtop.val[1], 2), m0.val[2]);

    out.val[0] = vmlaq_f32(out.val[0], vmid.val[0], m1.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vmid.val[0], vmid.val[1], 1), m1.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vmid.val[0], vmid.val[1], 2), m1.val[2]);

    out.val[0] = vmlaq_f32(out.val[0], vlow.val[0], m2.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vlow.val[0], vlow.val[1], 1), m2.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vlow.val[0], vlow.val[1], 2), m2.val[2]);

    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vtop.val[1], vtop.val[2], 1), m0.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vtop.val[1], vtop.val[2], 2), m0.val[2]);

    out.val[1] = vmlaq_f32(out.val[1], vmid.val[1], m1.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vmid.val[1], vmid.val[2], 1), m1.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vmid.val[1], vmid.val[2], 2), m1.val[2]);

    out.val[1] = vmlaq_f32(out.val[1], vlow.val[1], m2.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vlow.val[1], vlow.val[2], 1), m2.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vlow.val[1], vlow.val[2], 2), m2.val[2]);
    return out;
}

template <>
inline float32x4x2_t convolve_3x3<2>(const float *in_top, const float *in_mid, const float *in_low,
                                     const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, input_offset);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline float32x4x2_t convolve_3x3<3>(const float *in_top, const float *in_mid, const float *in_low,
                                     const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);

    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, input_offset);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    return out;
}

/** Perform a 3x3 convolution for 4 consecutive elements on uint8_t when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset Input quantization offset.
 *
 */
inline int32x4_t single_convolve_3x3_dilation(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                              const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                              size_t dilation_x, int input_offset)
{
    const int32x4_t v_input_offset = vdupq_n_s32(input_offset);

    const uint8x8x3_t vtop =
    {
        {
            vld1_u8(in_top),
            vld1_u8(in_top + dilation_x),
            vld1_u8(in_top + 2 * dilation_x)
        }
    };
    const uint8x8x3_t vmid =
    {
        {
            vld1_u8(in_mid),
            vld1_u8(in_mid + dilation_x),
            vld1_u8(in_mid + 2 * dilation_x)
        }
    };
    const uint8x8x3_t vlow =
    {
        {
            vld1_u8(in_low),
            vld1_u8(in_low + dilation_x),
            vld1_u8(in_low + 2 * dilation_x)
        }
    };

    const int32x4x3_t vtop_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vtop.val[0])))), //convert from uint8x8 to uint16x8, to uint16x4(lower or bottom half) to int16x4 to int32x4
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vtop.val[1])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vtop.val[2])))),
        }
    };
    const int32x4x3_t vmid_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vmid.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vmid.val[1])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vmid.val[2])))),
        }
    };
    const int32x4x3_t vlow_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vlow.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vlow.val[1])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vlow.val[2])))),
        }
    };

    int32x4_t out = vmulq_s32(vtop_s32.val[0], m0.val[0]);
    out           = vmlaq_s32(out, vtop_s32.val[1], m0.val[1]);
    out           = vmlaq_s32(out, vtop_s32.val[2], m0.val[2]);

    out = vmlaq_s32(out, vmid_s32.val[0], m1.val[0]);
    out = vmlaq_s32(out, vmid_s32.val[1], m1.val[1]);
    out = vmlaq_s32(out, vmid_s32.val[2], m1.val[2]);

    out = vmlaq_s32(out, vlow_s32.val[0], m2.val[0]);
    out = vmlaq_s32(out, vlow_s32.val[1], m2.val[1]);
    out = vmlaq_s32(out, vlow_s32.val[2], m2.val[2]);

    return out;
}

/** Perform a 3x3 convolution for 4 consecutive elements on uint8_t when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset Input quantization offset.
 *
 */
template <unsigned int stridex>
int32x4x2_t convolve_3x3_dilation(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                  const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                  const size_t dilation_x, int input_offset);

template <>
inline int32x4x2_t convolve_3x3_dilation<1>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low, const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                            const size_t dilation_x, int input_offset)
{
    const int32x4x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 4, in_mid + 4, in_low + 4, m0, m1, m2, dilation_x, input_offset)
        }
    };
    return out;
}

template <>
inline int32x4x2_t convolve_3x3_dilation<2>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                            const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                            const size_t dilation_x, int input_offset)
{
    int32x4x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);

    out.val[0] = vsetq_lane_s32(vgetq_lane_s32(out.val[0], 2), out.val[0], 1);
    out.val[0] = vsetq_lane_s32(vgetq_lane_s32(out.val[1], 0), out.val[0], 2);
    out.val[0] = vsetq_lane_s32(vgetq_lane_s32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline int32x4x2_t convolve_3x3_dilation<3>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                            const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                            const size_t dilation_x, int input_offset)
{
    int32x4x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);
    out.val[0]      = vsetq_lane_s32(vgetq_lane_s32(out.val[0], 3), out.val[0], 1);
    return out;
}

/** Perform a convolve3x3 on uint8_t
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
template <unsigned int stridex>
int32x4x2_t convolve_3x3(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                         const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                         int input_offset);

template <>
inline int32x4x2_t convolve_3x3<1>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low, const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                   int input_offset)
{
    const int32x4_t v_input_offset = vdupq_n_s32(input_offset);

    const uint8x8x2_t vtop =
    {
        {
            vld1_u8(in_top),
            vld1_u8(in_top + 8)
        }
    };
    const uint8x8x2_t vmid =
    {
        {
            vld1_u8(in_mid),
            vld1_u8(in_mid + 8)
        }
    };
    const uint8x8x2_t vlow =
    {
        {
            vld1_u8(in_low),
            vld1_u8(in_low + 8)
        }
    };

    const int32x4x3_t vtop_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vtop.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vtop.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vtop.val[1])))),
        }
    };
    const int32x4x3_t vmid_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vmid.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vmid.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vmid.val[1])))),
        }
    };
    const int32x4x3_t vlow_s32 =
    {
        {
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vlow.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vlow.val[0])))),
            vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vlow.val[1])))),
        }
    };

    int32x4x2_t out
    {
        {
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        }
    };

    // 0
    out.val[0] = vmlaq_s32(out.val[0], vtop_s32.val[0], m0.val[0]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vtop_s32.val[0], vtop_s32.val[1], 1), m0.val[1]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vtop_s32.val[0], vtop_s32.val[1], 2), m0.val[2]);

    out.val[0] = vmlaq_s32(out.val[0], vmid_s32.val[0], m1.val[0]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vmid_s32.val[0], vmid_s32.val[1], 1), m1.val[1]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vmid_s32.val[0], vmid_s32.val[1], 2), m1.val[2]);

    out.val[0] = vmlaq_s32(out.val[0], vlow_s32.val[0], m2.val[0]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vlow_s32.val[0], vlow_s32.val[1], 1), m2.val[1]);
    out.val[0] = vmlaq_s32(out.val[0], vextq_s32(vlow_s32.val[0], vlow_s32.val[1], 2), m2.val[2]);

    // 1
    out.val[1] = vmlaq_s32(out.val[1], vtop_s32.val[1], m0.val[0]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vtop_s32.val[1], vtop_s32.val[2], 1), m0.val[1]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vtop_s32.val[1], vtop_s32.val[2], 2), m0.val[2]);

    out.val[1] = vmlaq_s32(out.val[1], vmid_s32.val[1], m1.val[0]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vmid_s32.val[1], vmid_s32.val[2], 1), m1.val[1]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vmid_s32.val[1], vmid_s32.val[2], 2), m1.val[2]);

    out.val[1] = vmlaq_s32(out.val[1], vlow_s32.val[1], m2.val[0]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vlow_s32.val[1], vlow_s32.val[2], 1), m2.val[1]);
    out.val[1] = vmlaq_s32(out.val[1], vextq_s32(vlow_s32.val[1], vlow_s32.val[2], 2), m2.val[2]);

    return out;
}

template <>
inline int32x4x2_t convolve_3x3<2>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                   const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                   int input_offset)
{
    int32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, input_offset);
    out.val[0]      = vsetq_lane_s32(vgetq_lane_s32(out.val[0], 2), out.val[0], 1);
    out.val[0]      = vsetq_lane_s32(vgetq_lane_s32(out.val[1], 0), out.val[0], 2);
    out.val[0]      = vsetq_lane_s32(vgetq_lane_s32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline int32x4x2_t convolve_3x3<3>(const uint8_t *in_top, const uint8_t *in_mid, const uint8_t *in_low,
                                   const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                   int input_offset)
{
    int32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, input_offset);
    out.val[0]      = vsetq_lane_s32(vgetq_lane_s32(out.val[0], 3), out.val[0], 1);
    return out;
}

/** Stores a float32x4x2_t array into a memory location.
 *
 * @param[in] buffer Pointer to the memory location where the values will be stored.
 * @param[in] values Values that will be stored.
 *
 */
template <unsigned int stridex>
void store_results(float *buffer, const float32x4x2_t &values);

template <>
inline void store_results<1>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, values.val[0]);
    vst1q_f32(buffer + 4, values.val[1]);
}

template <>
inline void store_results<2>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, values.val[0]);
}

template <>
inline void store_results<3>(float *buffer, const float32x4x2_t &values)
{
    vst1_f32(buffer, vget_low_f32(values.val[0]));
}

/** Stores a uint32_t array into a memory location.
 *
 * @param[in] buffer Pointer to the memory location where the values will be stored.
 * @param[in] values Values that will be stored.
 *
 */
template <unsigned int stridex>
void store_results(int32_t *buffer, const int32x4x2_t &values);

template <>
inline void store_results<1>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1q_s32(buffer, values.val[0]);
    vst1q_s32(buffer + 4, values.val[1]);
}

template <>
inline void store_results<2>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1q_s32(buffer, values.val[0]);
}

template <>
inline void store_results<3>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1_s32(buffer, vget_low_s32(values.val[0]));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/** Loads a 3x3 matrix as a row (float16_t).
 *
 * @param[in] ptr Pointer to a float 3x3 matrix.
 *
 * @return The loaded matrix.
 */
inline float16x8x3_t load_matrix_row(const float16_t *ptr, int weights_offset = 0)
{
    ARM_COMPUTE_UNUSED(weights_offset);
    /* ptr is a pointer to a row in a 3x3 matrix, the function returns 3 vectors holding exactly the same value in all lanes:
       r.val[0] contains the first element, r.val[1] the second element and r.val[2] the third element (in all lanes) */
    const float16x8x3_t r =
    {
        {
            vld1q_dup_f16(ptr),
            vld1q_dup_f16(1 + ptr),
            vld1q_dup_f16(2 + ptr)
        }
    };
    return r;
}

/** Perform a 3x3 convolution for 8 consecutive elements on float16 when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset (Optional)Input quantization offset.
 *
 */
inline float16x8_t single_convolve_3x3_dilation(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                                const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                                const size_t dilation_x, int input_offset = 0)
{
    ARM_COMPUTE_UNUSED(input_offset);
    const float16x8x3_t vtop =
    {
        {
            vld1q_f16(in_top),
            vld1q_f16(in_top + dilation_x),
            vld1q_f16(in_top + 2 * dilation_x)
        }
    };
    const float16x8x3_t vmid =
    {
        {
            vld1q_f16(in_mid),
            vld1q_f16(in_mid + dilation_x),
            vld1q_f16(in_mid + 2 * dilation_x)
        }
    };
    const float16x8x3_t vlow =
    {
        {
            vld1q_f16(in_low),
            vld1q_f16(in_low + dilation_x),
            vld1q_f16(in_low + 2 * dilation_x)
        }
    };
    float16x8_t out = vmulq_f16(vtop.val[0], m0.val[0]);
    out             = vaddq_f16(out, vmulq_f16(vtop.val[1], m0.val[1]));
    out             = vaddq_f16(out, vmulq_f16(vtop.val[2], m0.val[2]));

    out = vaddq_f16(out, vmulq_f16(vmid.val[0], m1.val[0]));
    out = vaddq_f16(out, vmulq_f16(vmid.val[1], m1.val[1]));
    out = vaddq_f16(out, vmulq_f16(vmid.val[2], m1.val[2]));

    out = vaddq_f16(out, vmulq_f16(vlow.val[0], m2.val[0]));
    out = vaddq_f16(out, vmulq_f16(vlow.val[1], m2.val[1]));
    out = vaddq_f16(out, vmulq_f16(vlow.val[2], m2.val[2]));

    return out;
}

/** Perform a 3x3 convolution for 16 consecutive elements on float16 when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] input_offset (Optional)Input quantization offset.
 *
 */
template <unsigned int stridex>
float16x8x2_t convolve_3x3_dilation(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                    const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                    const size_t dilation_x, int input_offset = 0);

template <>
inline float16x8x2_t convolve_3x3_dilation<1>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                              const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    const float16x8x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 8, in_mid + 8, in_low + 8, m0, m1, m2, dilation_x, input_offset)
        }
    };
    return out;
}

template <>
inline float16x8x2_t convolve_3x3_dilation<2>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                              const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    float16x8x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 4), out.val[0], 2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 3);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 0), out.val[0], 4);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 2), out.val[0], 5);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 4), out.val[0], 6);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 6), out.val[0], 7);
    return out;
}

template <>
inline float16x8x2_t convolve_3x3_dilation<3>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                              const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                              const size_t dilation_x, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    float16x8x2_t out = convolve_3x3_dilation<1>(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 3), out.val[0], 1);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 1), out.val[0], 3);
    return out;
}

/** Perform a convolve3x3 on float16.
 *
 * @param[in] in_top Pointer to the first row of the input.
 * @param[in] in_mid Pointer to the second row of the input.
 * @param[in] in_low Pointer to the third row of the input.
 * @param[in] m0     First row of the filter.
 * @param[in] m1     Second row of the filter.
 * @param[in] m2     Third row of the filter.
 *
 */
template <unsigned int stridex>
float16x8x2_t convolve_3x3(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                           const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                           int input_offset = 0);

template <>
inline float16x8x2_t convolve_3x3<1>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                     const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    const float16x8x3_t vtop =
    {
        {
            vld1q_f16(in_top),
            vld1q_f16(in_top + 8),
            vld1q_f16(in_top + 16)
        }
    };
    const float16x8x3_t vmid =
    {
        {
            vld1q_f16(in_mid),
            vld1q_f16(in_mid + 8),
            vld1q_f16(in_mid + 16)
        }
    };
    const float16x8x3_t vlow =
    {
        {
            vld1q_f16(in_low),
            vld1q_f16(in_low + 8),
            vld1q_f16(in_low + 16)
        }
    };
    float16x8x2_t out =
    {
        {
            vmulq_f16(vtop.val[0], m0.val[0]),
            vmulq_f16(vtop.val[1], m0.val[0])
        }
    };
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vtop.val[0], vtop.val[1], 1), m0.val[1]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vtop.val[0], vtop.val[1], 2), m0.val[2]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vmid.val[0], m1.val[0]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vmid.val[0], vmid.val[1], 1), m1.val[1]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vmid.val[0], vmid.val[1], 2), m1.val[2]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vlow.val[0], m2.val[0]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vlow.val[0], vlow.val[1], 1), m2.val[1]));
    out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vlow.val[0], vlow.val[1], 2), m2.val[2]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vtop.val[1], vtop.val[2], 1), m0.val[1]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vtop.val[1], vtop.val[2], 2), m0.val[2]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vmid.val[1], m1.val[0]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vmid.val[1], vmid.val[2], 1), m1.val[1]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vmid.val[1], vmid.val[2], 2), m1.val[2]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vlow.val[1], m2.val[0]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vlow.val[1], vlow.val[2], 1), m2.val[1]));
    out.val[1] = vaddq_f16(out.val[1], vmulq_f16(vextq_f16(vlow.val[1], vlow.val[2], 2), m2.val[2]));
    return out;
}

template <>
inline float16x8x2_t convolve_3x3<2>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                     const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    float16x8x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 4), out.val[0], 2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 3);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 0), out.val[0], 4);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 2), out.val[0], 5);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 4), out.val[0], 6);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 6), out.val[0], 7);
    return out;
}

template <>
inline float16x8x2_t convolve_3x3<3>(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                     const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                     int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    float16x8x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 3), out.val[0], 1);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 2);
    out.val[0]        = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 1), out.val[0], 3);
    return out;
}

/** Stores a float16x8x2_t array into a memory location.
 *
 * @param[in] buffer Pointer to the memory location where the values will be stored.
 * @param[in] values Values that will be stored.
 *
 */
template <unsigned int stridex>
void store_results(float16_t *buffer, const float16x8x2_t &values);

template <>
inline void store_results<1>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, values.val[0]);
    vst1q_f16(buffer + 8, values.val[1]);
}

template <>
inline void store_results<2>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, values.val[0]);
}

template <>
inline void store_results<3>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1_f16(buffer, vget_low_f16(values.val[0]));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

/** Get the number of elements processed on 3x3 convolution.
 *
 * @param[in] num_elems_written_per_iteration Number of elements written per iteration on 3x3 convolution.
 *
 * @return The number of elements processed.
 */
template <unsigned int stridex>
int get_input_num_elems_processed(unsigned int num_elems_written_per_iteration);

template <>
inline int get_input_num_elems_processed<1>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration;
}

template <>
inline int get_input_num_elems_processed<2>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration << 1;
}

template <>
inline int get_input_num_elems_processed<3>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration * 3;
}
inline int get_input_num_elems_processed(unsigned int num_elems_written_per_iteration, unsigned int stridex)
{
    switch(stridex)
    {
        case 1:
            return get_input_num_elems_processed<1>(num_elems_written_per_iteration);
        case 2:
            return get_input_num_elems_processed<2>(num_elems_written_per_iteration);
        case 3:
            return get_input_num_elems_processed<3>(num_elems_written_per_iteration);
        default:
            ARM_COMPUTE_ERROR("stridex not supported");
            return 0;
    }
}
}
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H__ */
