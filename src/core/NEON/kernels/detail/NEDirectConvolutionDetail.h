/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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

#ifndef ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H
#define ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H

#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "support/AclRequires.h"

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

/** Loads a 3x3 matrix as a row (uint8_t/int8_t).
 *
 * @param[in] ptr            Pointer to a uint8_t/int8_t 3x3 matrix.
 * @param[in] weights_offset (Optional) Weights quantization offset.
 *
 * @return The loaded matrix.
 */
template < typename T, ARM_COMPUTE_REQUIRES_TA(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) >
inline int32x4x3_t load_matrix_row(const T *ptr, int weights_offset = 0)
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

template <unsigned int stridex>
inline void accumulate_results(float *buffer, const float32x4x2_t &values);

template <>
inline void accumulate_results<1>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, vaddq_f32(vld1q_f32(buffer), values.val[0]));
    vst1q_f32(buffer + 4, vaddq_f32(vld1q_f32(buffer + 4), values.val[1]));
}

template <>
inline void accumulate_results<2>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, vaddq_f32(vld1q_f32(buffer), values.val[0]));
}

template <>
inline void accumulate_results<3>(float *buffer, const float32x4x2_t &values)
{
    vst1_f32(buffer, vadd_f32(vld1_f32(buffer), vget_low_f32(values.val[0])));
}

template <unsigned int stridex>
void accumulate_results(int32_t *buffer, const int32x4x2_t &values);

template <>
inline void accumulate_results<1>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1q_s32(buffer, vaddq_s32(vld1q_s32(buffer), values.val[0]));
    vst1q_s32(buffer + 4, vaddq_s32(vld1q_s32(buffer + 4), values.val[1]));
}

template <>
inline void accumulate_results<2>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1q_s32(buffer, vaddq_s32(vld1q_s32(buffer), values.val[0]));
}

template <>
inline void accumulate_results<3>(int32_t *buffer, const int32x4x2_t &values)
{
    vst1_s32(buffer, vadd_s32(vld1_s32(buffer), vget_low_s32(values.val[0])));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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

template <unsigned int stridex>
inline void accumulate_results(float16_t *buffer, const float16x8x2_t &values);

template <>
inline void accumulate_results<1>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, vaddq_f16(vld1q_f16(buffer), values.val[0]));
    vst1q_f16(buffer + 8, vaddq_f16(vld1q_f16(buffer + 8), values.val[1]));
}

template <>
inline void accumulate_results<2>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, vaddq_f16(vld1q_f16(buffer), values.val[0]));
}

template <>
inline void accumulate_results<3>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1_f16(buffer, vadd_f16(vld1_f16(buffer), vget_low_f16(values.val[0])));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

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
 * @param[in] stridex      Stride value in elements across x.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
inline float32x4x2_t convolve_3x3_dilation(const float *in_top, const float *in_mid, const float *in_low,
                                           const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                                           const size_t dilation_x, unsigned int stridex, int input_offset = 0)
{
    ARM_COMPUTE_ERROR_ON(stridex > 3);
    float32x4x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 4, in_mid + 4, in_low + 4, m0, m1, m2, dilation_x, input_offset)
        }
    };

    if(stridex == 2)
    {
        out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
        out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
        out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    }
    else if(stridex == 3)
    {
        out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    }

    return out;
}

/** Perform a convolve3x3 on float32.
 *
 * @param[in]  in_top       Pointer to the first row of the input.
 * @param[in]  in_mid       Pointer to the second row of the input.
 * @param[in]  in_low       Pointer to the third row of the input.
 * @param[out] out_ptr      Pointer to the output.
 * @param[in]  m0           First row of the filter.
 * @param[in]  m1           Second row of the filter.
 * @param[in]  m2           Third row of the filter.
 * @param[in]  stridex      Stride value in elements across x.
 * @param[in]  input_offset (Optional) Input quantization offset.
 *
 */
template <bool accumulate>
void convolve_3x3(const float *in_top, const float *in_mid, const float *in_low, float *out_ptr,
                  const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                  unsigned int stridex, int input_offset = 0);

template <bool accumulate>
inline void convolve_3x3(const float *in_top, const float *in_mid, const float *in_low, float *out_ptr,
                         const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2,
                         unsigned int stridex, int input_offset)
{
    ARM_COMPUTE_UNUSED(input_offset);
    ARM_COMPUTE_ERROR_ON(stridex > 3);

    float32x4x2_t out =
    {
        {
            vdupq_n_f32(0.f),
            vdupq_n_f32(0.f)
        }
    };
    if(stridex == 2)
    {
        const float32x4x2_t vtop     = vld2q_f32(in_top);
        const float32x4x2_t vmid     = vld2q_f32(in_mid);
        const float32x4x2_t vlow     = vld2q_f32(in_low);
        const float32x4_t   vtop_end = vld1q_f32(in_top + 8);
        const float32x4_t   vmid_end = vld1q_f32(in_mid + 8);
        const float32x4_t   vlow_end = vld1q_f32(in_low + 8);

        out.val[0] = vmulq_f32(vtop.val[0], m0.val[0]);

        out.val[0] = vmlaq_f32(out.val[0], vtop.val[1], m0.val[1]);
        out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vtop.val[0], vtop_end, 1), m0.val[2]);

        out.val[0] = vmlaq_f32(out.val[0], vmid.val[0], m1.val[0]);
        out.val[0] = vmlaq_f32(out.val[0], vmid.val[1], m1.val[1]);
        out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vmid.val[0], vmid_end, 1), m1.val[2]);

        out.val[0] = vmlaq_f32(out.val[0], vlow.val[0], m2.val[0]);
        out.val[0] = vmlaq_f32(out.val[0], vlow.val[1], m2.val[1]);
        out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vlow.val[0], vlow_end, 1), m2.val[2]);

        accumulate ? accumulate_results<2>(out_ptr, out) : store_results<2>(out_ptr, out);
    }
    else
    {
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
        out.val[0] = vmulq_f32(vtop.val[0], m0.val[0]);
        out.val[1] = vmulq_f32(vtop.val[1], m0.val[0]);

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

        if(stridex == 3)
        {
            out.val[0] = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
            accumulate ? accumulate_results<3>(out_ptr, out) : store_results<3>(out_ptr, out);
        }
        else
        {
            accumulate ? accumulate_results<1>(out_ptr, out) : store_results<1>(out_ptr, out);
        }
    }
}

/** Perform a 3x3 convolution for 4 consecutive 8-bit elements when dilation.x() or dilation.y() is not 1.
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
template < typename T, ARM_COMPUTE_REQUIRES_TA(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) >
inline int32x4_t single_convolve_3x3_dilation(const T *in_top, const T *in_mid, const T *in_low,
                                              const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                              size_t dilation_x, int32_t input_offset)
{
    using VectorType    = typename std::conditional<std::is_same<T, uint8_t>::value, uint8x8x3_t, int8x8x3_t>::type;
    using OutputTagType = typename wrapper::traits::neon_bitvector_tag_t<int32_t, wrapper::traits::BitWidth::W128>;

    const int32x4_t v_input_offset = wrapper::vdup_n(input_offset, OutputTagType{});

    const VectorType vtop =
    {
        {
            wrapper::vload(in_top),
            wrapper::vload(in_top + dilation_x),
            wrapper::vload(in_top + 2 * dilation_x)
        }
    };
    const VectorType vmid =
    {
        {
            wrapper::vload(in_mid),
            wrapper::vload(in_mid + dilation_x),
            wrapper::vload(in_mid + 2 * dilation_x)
        }
    };
    const VectorType vlow =
    {
        {
            wrapper::vload(in_low),
            wrapper::vload(in_low + dilation_x),
            wrapper::vload(in_low + 2 * dilation_x)
        }
    };

    const int32x4x3_t vtop_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vtop.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vtop.val[1])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vtop.val[2])))),
        }
    };
    const int32x4x3_t vmid_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vmid.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vmid.val[1])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vmid.val[2])))),
        }
    };
    const int32x4x3_t vlow_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vlow.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vlow.val[1])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vlow.val[2])))),
        }
    };

    int32x4_t out = wrapper::vmul(vtop_s32.val[0], m0.val[0]);
    out           = wrapper::vmla(out, vtop_s32.val[1], m0.val[1]);
    out           = wrapper::vmla(out, vtop_s32.val[2], m0.val[2]);

    out = wrapper::vmla(out, vmid_s32.val[0], m1.val[0]);
    out = wrapper::vmla(out, vmid_s32.val[1], m1.val[1]);
    out = wrapper::vmla(out, vmid_s32.val[2], m1.val[2]);

    out = wrapper::vmla(out, vlow_s32.val[0], m2.val[0]);
    out = wrapper::vmla(out, vlow_s32.val[1], m2.val[1]);
    out = wrapper::vmla(out, vlow_s32.val[2], m2.val[2]);

    return out;
}

/** Perform a 3x3 convolution for 4 consecutive 8-bit elements when dilation.x() or dilation.y() is not 1.
 *
 * @param[in] in_top       Pointer to the first row of the input.
 * @param[in] in_mid       Pointer to the second row of the input.
 * @param[in] in_low       Pointer to the third row of the input.
 * @param[in] m0           First row of the filter.
 * @param[in] m1           Second row of the filter.
 * @param[in] m2           Third row of the filter.
 * @param[in] dilation_x   Dilation, in elements across x.
 * @param[in] stridex      Stride value in elements across x.
 * @param[in] input_offset Input quantization offset.
 *
 */
template < typename T, ARM_COMPUTE_REQUIRES_TA(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) >
inline int32x4x2_t convolve_3x3_dilation(const T *in_top, const T *in_mid, const T *in_low, const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                                         const size_t dilation_x, unsigned int stridex, int input_offset)
{
    ARM_COMPUTE_ERROR_ON(stridex > 3);
    int32x4x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 4, in_mid + 4, in_low + 4, m0, m1, m2, dilation_x, input_offset)
        }
    };

    if(stridex == 2)
    {
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[0], 2), out.val[0], 1);
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[1], 0), out.val[0], 2);
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[1], 2), out.val[0], 3);
    }
    else if(stridex == 3)
    {
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[0], 3), out.val[0], 1);
    }
    return out;
}

/** Perform a convolve3x3 on 8-bit elements
 *
 * @param[in]  in_top       Pointer to the first row of the input.
 * @param[in]  in_mid       Pointer to the second row of the input.
 * @param[in]  in_low       Pointer to the third row of the input.
 * @param[out] out_ptr      Pointer to the output.
 * @param[in]  m0           First row of the filter.
 * @param[in]  m1           Second row of the filter.
 * @param[in]  m2           Third row of the filter.
 * @param[in]  stridex      Stride value in elements across x.
 * @param[in]  input_offset Input quantization offset.
 *
 */
template < bool accumulate, typename T1, typename T2, ARM_COMPUTE_REQUIRES_TA(std::is_same<T1, uint8_t>::value || std::is_same<T1, int8_t>::value) >
void convolve_3x3(const T1 *in_top, const T1 *in_mid, const T1 *in_low, T2 *out_ptr,
                  const int32x4x3_t &m0, const int32x4x3_t &m1, const int32x4x3_t &m2,
                  unsigned int stridex, int32_t input_offset)
{
    ARM_COMPUTE_ERROR_ON(stridex > 3);
    using VectorType    = typename std::conditional<std::is_same<T1, uint8_t>::value, uint8x8x2_t, int8x8x2_t>::type;
    using OutputTagType = typename wrapper::traits::neon_bitvector_tag_t<int32_t, wrapper::traits::BitWidth::W128>;

    const int32x4_t v_input_offset = wrapper::vdup_n(input_offset, OutputTagType{});

    const VectorType vtop =
    {
        {
            wrapper::vload(in_top),
            wrapper::vload(in_top + 8)
        }
    };
    const VectorType vmid =
    {
        {
            wrapper::vload(in_mid),
            wrapper::vload(in_mid + 8)
        }
    };
    const VectorType vlow =
    {
        {
            wrapper::vload(in_low),
            wrapper::vload(in_low + 8)
        }
    };

    const int32x4x3_t vtop_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vtop.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgethigh(wrapper::vmovl(vtop.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vtop.val[1])))),
        }
    };
    const int32x4x3_t vmid_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vmid.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgethigh(wrapper::vmovl(vmid.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vmid.val[1])))),
        }
    };
    const int32x4x3_t vlow_s32 =
    {
        {
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vlow.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgethigh(wrapper::vmovl(vlow.val[0])))),
            wrapper::vaddw(v_input_offset, wrapper::vreinterpret(wrapper::vgetlow(wrapper::vmovl(vlow.val[1])))),
        }
    };

    int32x4x2_t out
    {
        {
            wrapper::vdup_n(static_cast<int32_t>(0), OutputTagType{}),
            wrapper::vdup_n(static_cast<int32_t>(0), OutputTagType{}),
        }
    };

    // 0
    out.val[0] = wrapper::vmla(out.val[0], vtop_s32.val[0], m0.val[0]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_1(vtop_s32.val[0], vtop_s32.val[1]), m0.val[1]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_2(vtop_s32.val[0], vtop_s32.val[1]), m0.val[2]);

    out.val[0] = wrapper::vmla(out.val[0], vmid_s32.val[0], m1.val[0]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_1(vmid_s32.val[0], vmid_s32.val[1]), m1.val[1]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_2(vmid_s32.val[0], vmid_s32.val[1]), m1.val[2]);

    out.val[0] = wrapper::vmla(out.val[0], vlow_s32.val[0], m2.val[0]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_1(vlow_s32.val[0], vlow_s32.val[1]), m2.val[1]);
    out.val[0] = wrapper::vmla(out.val[0], wrapper::vext_2(vlow_s32.val[0], vlow_s32.val[1]), m2.val[2]);

    // 1
    out.val[1] = wrapper::vmla(out.val[1], vtop_s32.val[1], m0.val[0]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_1(vtop_s32.val[1], vtop_s32.val[2]), m0.val[1]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_2(vtop_s32.val[1], vtop_s32.val[2]), m0.val[2]);

    out.val[1] = wrapper::vmla(out.val[1], vmid_s32.val[1], m1.val[0]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_1(vmid_s32.val[1], vmid_s32.val[2]), m1.val[1]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_2(vmid_s32.val[1], vmid_s32.val[2]), m1.val[2]);

    out.val[1] = wrapper::vmla(out.val[1], vlow_s32.val[1], m2.val[0]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_1(vlow_s32.val[1], vlow_s32.val[2]), m2.val[1]);
    out.val[1] = wrapper::vmla(out.val[1], wrapper::vext_2(vlow_s32.val[1], vlow_s32.val[2]), m2.val[2]);

    if(stridex == 1)
    {
        accumulate ? accumulate_results<1>(out_ptr, out) : store_results<1>(out_ptr, out);
    }
    else if(stridex == 2)
    {
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[0], 2), out.val[0], 1);
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[1], 0), out.val[0], 2);
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[1], 2), out.val[0], 3);

        accumulate ? accumulate_results<2>(out_ptr, out) : store_results<2>(out_ptr, out);
    }
    else if(stridex == 3)
    {
        out.val[0] = wrapper::vsetlane(wrapper::vgetlane(out.val[0], 3), out.val[0], 1);
        accumulate ? accumulate_results<3>(out_ptr, out) : store_results<3>(out_ptr, out);
    }
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
 * @param[in] stridex      Stride value in elements across x.
 * @param[in] input_offset (Optional) Input quantization offset.
 *
 */
inline float16x8x2_t convolve_3x3_dilation(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low,
                                           const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                                           const size_t dilation_x, unsigned int stridex, int input_offset = 0)
{
    float16x8x2_t out =
    {
        {
            single_convolve_3x3_dilation(in_top, in_mid, in_low, m0, m1, m2, dilation_x, input_offset),
            single_convolve_3x3_dilation(in_top + 8, in_mid + 8, in_low + 8, m0, m1, m2, dilation_x, input_offset)
        }
    };

    if(stridex == 2)
    {
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 2), out.val[0], 1);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 4), out.val[0], 2);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 3);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 0), out.val[0], 4);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 2), out.val[0], 5);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 4), out.val[0], 6);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 6), out.val[0], 7);
    }
    else if(stridex == 3)
    {
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 3), out.val[0], 1);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 2);
        out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 1), out.val[0], 3);
    }

    return out;
}

/** Perform a convolve3x3 on float16.
 *
 * @param[in]  in_top       Pointer to the first row of the input.
 * @param[in]  in_mid       Pointer to the second row of the input.
 * @param[in]  in_low       Pointer to the third row of the input.
 * @param[out] out_ptr      Pointer to the output.
 * @param[in]  m0           First row of the filter.
 * @param[in]  m1           Second row of the filter.
 * @param[in]  m2           Third row of the filter.
 * @param[in]  stridex      Stride value in elements across x.
 * @param[in]  input_offset (Optional) Input quantization offset.
 *
 */
template <bool accumulate>
inline void convolve_3x3(const float16_t *in_top, const float16_t *in_mid, const float16_t *in_low, float16_t *out_ptr,
                         const float16x8x3_t &m0, const float16x8x3_t &m1, const float16x8x3_t &m2,
                         unsigned int stridex, int input_offset = 0)
{
    ARM_COMPUTE_UNUSED(input_offset);

    float16x8x2_t out =
    {
        {
            vdupq_n_f16(0),
            vdupq_n_f16(0)
        }
    };
    if(stridex == 2)
    {
        const float16x8x2_t vtop     = vld2q_f16(in_top);
        const float16x8x2_t vmid     = vld2q_f16(in_mid);
        const float16x8x2_t vlow     = vld2q_f16(in_low);
        const float16x8_t   vtop_end = vld1q_f16(in_top + 16);
        const float16x8_t   vmid_end = vld1q_f16(in_mid + 16);
        const float16x8_t   vlow_end = vld1q_f16(in_low + 16);

        out.val[0] = vmulq_f16(vtop.val[0], m0.val[0]);

        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vtop.val[1], m0.val[1]));
        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vtop.val[0], vtop_end, 1), m0.val[2]));

        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vmid.val[0], m1.val[0]));
        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vmid.val[1], m1.val[1]));
        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vmid.val[0], vmid_end, 1), m1.val[2]));

        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vlow.val[0], m2.val[0]));
        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vlow.val[1], m2.val[1]));
        out.val[0] = vaddq_f16(out.val[0], vmulq_f16(vextq_f16(vlow.val[0], vlow_end, 1), m2.val[2]));

        accumulate ? accumulate_results<2>(out_ptr, out) : store_results<2>(out_ptr, out);
    }
    else
    {
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
        out.val[0] = vmulq_f16(vtop.val[0], m0.val[0]);
        out.val[1] = vmulq_f16(vtop.val[1], m0.val[0]);

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

        if(stridex == 3)
        {
            out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 3), out.val[0], 1);
            out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[0], 6), out.val[0], 2);
            out.val[0] = vsetq_lane_f16(vgetq_lane_f16(out.val[1], 1), out.val[0], 3);

            accumulate ? accumulate_results<3>(out_ptr, out) : store_results<3>(out_ptr, out);
        }
        else
        {
            accumulate ? accumulate_results<1>(out_ptr, out) : store_results<1>(out_ptr, out);
        }
    }
}
#endif /** __ARM_FEATURE_FP16_VECTOR_ARITHMETIC **/

/** Get the number of elements processed on 3x3 convolution.
 *
 * @param[in] num_elems_written_per_iteration Number of elements written per iteration on 3x3 convolution.
 * @param[in] stridex                         Stride value in elements across x.
 *
 * @return The number of elements processed.
 */
inline int get_input_num_elems_processed(unsigned int num_elems_written_per_iteration, unsigned int stridex)
{
    switch(stridex)
    {
        case 1:
            return num_elems_written_per_iteration;
        case 2:
            return num_elems_written_per_iteration << 1;
        case 3:
            return num_elems_written_per_iteration * 3;
        default:
            ARM_COMPUTE_ERROR("stridex not supported");
            return 0;
    }
}
}
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEDIRECTCONVOLUTIONDETAIL_H */
