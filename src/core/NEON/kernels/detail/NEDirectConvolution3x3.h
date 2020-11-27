/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#ifndef ARM_COMPUTE_NECONVOLUTIONKERNEL3x3_H
#define ARM_COMPUTE_NECONVOLUTIONKERNEL3x3_H

#include <arm_neon.h>

namespace arm_compute
{
namespace detail
{
inline float32x4x3_t load_matrix_row(const float *ptr)
{
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

template <unsigned int stridex>
float32x4x2_t convolve_3x3(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2);

template <>
inline float32x4x2_t convolve_3x3<1>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2)
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
inline float32x4x2_t convolve_3x3<2>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2)
{
    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline float32x4x2_t convolve_3x3<3>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2)
{
    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    return out;
}

template <unsigned int stridex>
void store_results(float *buffer, const float32x4x2_t &values);

template <>
void store_results<1>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, values.val[0]);
    vst1q_f32(buffer + 4, values.val[1]);
}

template <>
void store_results<2>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, values.val[0]);
}

template <>
void store_results<3>(float *buffer, const float32x4x2_t &values)
{
    vst1_f32(buffer, vget_low_f32(values.val[0]));
}

template <unsigned int stridex>
int get_input_num_elems_processed(unsigned int num_elems_written_per_iteration);

template <>
int get_input_num_elems_processed<1>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration;
}

template <>
int get_input_num_elems_processed<2>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration << 1;
}

template <>
int get_input_num_elems_processed<3>(unsigned int num_elems_written_per_iteration)
{
    return num_elems_written_per_iteration * 3;
}
}
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECONVOLUTIONKERNEL3x3_H */