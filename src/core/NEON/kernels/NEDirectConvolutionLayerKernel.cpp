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
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>

using namespace arm_compute;

namespace
{
template <unsigned int stridex>
float32x4_t internal_vld1q(const float *in);

template <>
float32x4_t internal_vld1q<1>(const float *in)
{
    return vld1q_f32(in);
}

template <>
float32x4_t internal_vld1q<2>(const float *in)
{
    const float32x4x2_t tmp = vld2q_f32(in);
    return tmp.val[0];
}

template <>
float32x4_t internal_vld1q<3>(const float *in)
{
    const float32x4x3_t tmp = vld3q_f32(in);
    return tmp.val[0];
}

template <unsigned int stridex>
qint8x8_t internal_vld1q(const qint8_t *in);

template <>
qint8x8_t internal_vld1q<1>(const qint8_t *in)
{
    return vld1_qs8(in);
}

template <>
qint8x8_t internal_vld1q<2>(const qint8_t *in)
{
    const qint8x8x2_t tmp = vld2_s8(in);
    return tmp.val[0];
}

template <>
qint8x8_t internal_vld1q<3>(const qint8_t *in)
{
    const qint8x8x3_t tmp = vld3_s8(in);
    return tmp.val[0];
}

template <unsigned int stridex>
qint16x8_t internal_vld1q(const qint16_t *in);

template <>
qint16x8_t internal_vld1q<1>(const qint16_t *in)
{
    return vld1q_s16(in);
}

inline float32x4_t internal_vdupq_n(float v)
{
    return vdupq_n_f32(v);
}

inline qint8x8_t internal_vdupq_n(qint8_t v)
{
    return vdup_n_qs8(v);
}

inline void internal_vst1q(float *p, const float32x4_t &v)
{
    vst1q_f32(p, v);
}

inline void internal_vst1q(qint16_t *p, const qint16x8_t &v)
{
    vst1q_qs16(p, v);
}

float32x4_t internal_vmull(const float32x4_t &x, const float32x4_t &y, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vmulq_f32(x, y);
}

qint16x8_t internal_vmull(const qint8x8_t &x, const qint8x8_t &y, int fixed_point_position)
{
    return vmull_qs8(x, y, fixed_point_position);
}

inline float32x4_t internal_vmlal(const float32x4_t &x, const float32x4_t &y, const float32x4_t &z, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vmlaq_f32(x, y, z);
}

inline qint16x8_t internal_vmlal(const qint16x8_t &x, const qint8x8_t &y, const qint8x8_t &z, int fixed_point_position)
{
    return vqmlal_qs8(x, y, z, fixed_point_position);
}

template <typename T1, typename T2, unsigned int stridex>
class convolver_1x1
{
public:
    static void convolve(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
    {
        const int          input_stride_y       = input->info()->strides_in_bytes().y();
        const int          input_stride_z       = input->info()->strides_in_bytes().z();
        const int          output_stride_y      = output->info()->strides_in_bytes().y();
        const int          output_stride_z      = output->info()->strides_in_bytes().z();
        const int          kernel_stride_z      = weights->info()->strides_in_bytes().z();
        const int          kernel_stride_w      = weights->info()->strides_in_bytes()[3];
        const int          output_w             = output->info()->dimension(0);
        const int          output_h             = output->info()->dimension(1);
        const int          range_z              = window.z().end() - window.z().start();
        const int          kernel_depth         = weights->info()->dimension(Window::DimZ);
        const unsigned int conv_stride_y        = std::get<1>(conv_info.stride());
        const int          fixed_point_position = input->info()->fixed_point_position();

        // setup output window for the iterator
        Window window_out = window;
        window_out.set(Window::DimX, Window::Dimension(0, output->info()->dimension(Window::DimX), output->info()->dimension(Window::DimX)));
        window_out.set(Window::DimY, Window::Dimension(0, output->info()->dimension(Window::DimY), output->info()->dimension(Window::DimY)));
        window_out.set(Window::DimZ, Window::Dimension(window.z().start(), window.z().end(), range_z));

        // setup input window for the iterator
        Window window_in = window;
        // we just want execute_window_loop to iterate over the higher dimensions (>3), so we set the first 3 dimensions to 0
        window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimY, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Window window_k = calculate_max_window(*weights->info(), Steps(1u));

        Iterator out(output, window_out);
        Iterator in(input, window_in);
        Iterator k(weights, window_k);

        const uint8_t *k_ptr = k.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            /*
                For a detailed explanation on how the algorithm works refer to template <> class convolver_3x3<1>
            */
            const uint8_t *input_ptr = in.ptr();
            uint8_t       *out_ptr   = out.ptr();
            int            ih        = 0;
            int            oh        = 0;
            for(int oz = 0; oz < range_z; ++oz)
            {
                auto p_out_base = out_ptr + oz * output_stride_z;
                // Step 1
                {
                    const auto k_val = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + (id.z() + oz) * kernel_stride_w);
                    const auto vk    = internal_vdupq_n(*k_val);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        const int offset_xy = ih * input_stride_y;
                        auto      in_val    = reinterpret_cast<const T1 *>(input_ptr + (0 * input_stride_z + offset_xy));
                        auto      p_out     = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration, in_val += num_elems_read_per_iteration, p_out += num_elems_written_per_iteration)
                        {
                            internal_vst1q(p_out, internal_vmull(vk, internal_vld1q<stridex>(in_val), fixed_point_position));
                        }
                    }
                }
                // Step 2
                for(int p = 1; p < kernel_depth; ++p)
                {
                    const auto k_val = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + (id.z() + oz) * kernel_stride_w);
                    const auto vk    = internal_vdupq_n(*k_val);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        const int offset_xy = ih * input_stride_y;
                        auto      in_val    = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + offset_xy);
                        auto      p_out     = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration, in_val += num_elems_read_per_iteration, p_out += num_elems_written_per_iteration)
                        {
                            internal_vst1q(p_out, internal_vmlal(internal_vld1q<1>(p_out), vk, internal_vld1q<stridex>(in_val), fixed_point_position));
                        }
                    }
                }
            }
        },
        in, out);
    }
};

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
inline qint8x8x3_t load_matrix_row(const qint8_t *ptr)
{
    /* ptr is a pointer to a row in a 3x3 matrix, the function returns 3 vectors holding exactly the same value in all lanes:
       r.val[0] contains the first element, r.val[1] the second element and r.val[2] the third element (in all lanes) */
    const qint8x8x3_t r =
    {
        {
            vld1_dup_qs8(ptr),
            vld1_dup_qs8(1 + ptr),
            vld1_dup_qs8(2 + ptr)
        }
    };
    return r;
}

template <unsigned int stridex>
float32x4x2_t convolve_3x3(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2, int fixed_point_position);

template <>
inline float32x4x2_t convolve_3x3<1>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

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
inline float32x4x2_t convolve_3x3<2>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2, int fixed_point_position)
{
    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, fixed_point_position);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline float32x4x2_t convolve_3x3<3>(const float *in_top, const float *in_mid, const float *in_low, const float32x4x3_t &m0, const float32x4x3_t &m1, const float32x4x3_t &m2, int fixed_point_position)
{
    float32x4x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, fixed_point_position);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    return out;
}

template <unsigned int stridex>
qint16x8x2_t convolve_3x3(const qint8_t *in_top, const qint8_t *in_mid, const qint8_t *in_low, const qint8x8x3_t &m0, const qint8x8x3_t &m1, const qint8x8x3_t &m2, int fixed_point_position);

template <>
inline qint16x8x2_t convolve_3x3<1>(const qint8_t *in_top, const qint8_t *in_mid, const qint8_t *in_low, const qint8x8x3_t &m0, const qint8x8x3_t &m1, const qint8x8x3_t &m2, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

    const qint8x8x3_t vtop =
    {
        {
            vld1_qs8(in_top),
            vld1_qs8(in_top + 8),
            vld1_qs8(in_top + 16)
        }
    };
    const qint8x8x3_t vmid =
    {
        {
            vld1_qs8(in_mid),
            vld1_qs8(in_mid + 8),
            vld1_qs8(in_mid + 16)
        }
    };
    const qint8x8x3_t vlow =
    {
        {
            vld1_qs8(in_low),
            vld1_qs8(in_low + 8),
            vld1_qs8(in_low + 16)
        }
    };
    qint16x8x2_t out =
    {
        {
            vmull_qs8(vtop.val[0], m0.val[0], fixed_point_position),
            vmull_qs8(vtop.val[1], m0.val[0], fixed_point_position)
        }
    };
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vtop.val[0], vtop.val[1], 1), m0.val[1], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vtop.val[0], vtop.val[1], 2), m0.val[2], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vmid.val[0], m1.val[0], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vmid.val[0], vmid.val[1], 1), m1.val[1], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vmid.val[0], vmid.val[1], 2), m1.val[2], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vlow.val[0], m2.val[0], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vlow.val[0], vlow.val[1], 1), m2.val[1], fixed_point_position);
    out.val[0] = vqmlal_qs8(out.val[0], vext_s8(vlow.val[0], vlow.val[1], 2), m2.val[2], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vtop.val[1], vtop.val[2], 1), m0.val[1], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vtop.val[1], vtop.val[2], 2), m0.val[2], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vmid.val[1], m1.val[0], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vmid.val[1], vmid.val[2], 1), m1.val[1], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vmid.val[1], vmid.val[2], 2), m1.val[2], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vlow.val[1], m2.val[0], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vlow.val[1], vlow.val[2], 1), m2.val[1], fixed_point_position);
    out.val[1] = vqmlal_qs8(out.val[1], vext_s8(vlow.val[1], vlow.val[2], 2), m2.val[2], fixed_point_position);
    return out;
}

template <>
inline qint16x8x2_t convolve_3x3<2>(const qint8_t *in_top, const qint8_t *in_mid, const qint8_t *in_low, const qint8x8x3_t &m0, const qint8x8x3_t &m1, const qint8x8x3_t &m2, int fixed_point_position)
{
    qint16x8x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, fixed_point_position);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[0], 2), out.val[0], 1);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[0], 4), out.val[0], 2);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[0], 6), out.val[0], 3);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[1], 0), out.val[0], 4);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[1], 2), out.val[0], 5);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[1], 4), out.val[0], 6);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[1], 6), out.val[0], 7);
    return out;
}

template <>
inline qint16x8x2_t convolve_3x3<3>(const qint8_t *in_top, const qint8_t *in_mid, const qint8_t *in_low, const qint8x8x3_t &m0, const qint8x8x3_t &m1, const qint8x8x3_t &m2, int fixed_point_position)
{
    qint16x8x2_t out = convolve_3x3<1>(in_top, in_mid, in_low, m0, m1, m2, fixed_point_position);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[0], 3), out.val[0], 1);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[0], 6), out.val[0], 2);
    out.val[0]       = vsetq_lane_s16(vgetq_lane_s16(out.val[1], 1), out.val[0], 3);
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
void store_results(qint16_t *buffer, const qint16x8x2_t &values);

template <>
void store_results<1>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1q_qs16(buffer, values.val[0]);
    vst1q_qs16(buffer + 8, values.val[1]);
}

template <>
void store_results<2>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1q_qs16(buffer, values.val[0]);
}

template <>
void store_results<3>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1_qs16(buffer, vget_low_s16(values.val[0]));
}

template <unsigned int stridex>
void accumulate_results(float *buffer, const float32x4x2_t &values);

template <>
void accumulate_results<1>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, vaddq_f32(vld1q_f32(buffer), values.val[0]));
    vst1q_f32(buffer + 4, vaddq_f32(vld1q_f32(buffer + 4), values.val[1]));
}

template <>
void accumulate_results<2>(float *buffer, const float32x4x2_t &values)
{
    vst1q_f32(buffer, vaddq_f32(vld1q_f32(buffer), values.val[0]));
}

template <>
void accumulate_results<3>(float *buffer, const float32x4x2_t &values)
{
    vst1_f32(buffer, vadd_f32(vld1_f32(buffer), vget_low_f32(values.val[0])));
}

template <unsigned int stridex>
void accumulate_results(qint16_t *buffer, const qint16x8x2_t &values);

template <>
void accumulate_results<1>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1q_qs16(buffer, vqaddq_qs16(vld1q_qs16(buffer), values.val[0]));
    vst1q_qs16(buffer + 8, vqaddq_qs16(vld1q_qs16(buffer + 8), values.val[1]));
}

template <>
void accumulate_results<2>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1q_qs16(buffer, vqaddq_qs16(vld1q_qs16(buffer), values.val[0]));
}

template <>
void accumulate_results<3>(qint16_t *buffer, const qint16x8x2_t &values)
{
    vst1_qs16(buffer, vqadd_qs16(vld1_qs16(buffer), vget_low_s16(values.val[0])));
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

template <typename T1, typename T2, unsigned int stridex>
class convolver_3x3
{
public:
    static void convolve(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
    {
        ARM_COMPUTE_UNUSED(num_elems_read_per_iteration);
        const int          input_stride_x       = input->info()->strides_in_bytes().x();
        const int          input_stride_y       = input->info()->strides_in_bytes().y();
        const int          input_stride_z       = input->info()->strides_in_bytes().z();
        const int          output_stride_y      = output->info()->strides_in_bytes().y();
        const int          output_stride_z      = output->info()->strides_in_bytes().z();
        const int          kernel_stride_x      = weights->info()->strides_in_bytes().x();
        const int          kernel_stride_y      = weights->info()->strides_in_bytes().y();
        const int          kernel_stride_z      = weights->info()->strides_in_bytes().z();
        const int          kernel_stride_w      = weights->info()->strides_in_bytes()[3];
        const int          output_w             = output->info()->dimension(0);
        const int          output_h             = output->info()->dimension(1);
        const int          num_planes_z         = window.z().end() - window.z().start();
        const int          delta_input          = get_input_num_elems_processed<stridex>(num_elems_written_per_iteration);
        const int          kernel_depth         = weights->info()->dimension(Window::DimZ);
        const unsigned int conv_stride_y        = std::get<1>(conv_info.stride());
        const unsigned int conv_pad_x           = std::get<0>(conv_info.pad());
        const unsigned int conv_pad_y           = std::get<1>(conv_info.pad());
        const int          fixed_point_position = input->info()->fixed_point_position();

        // setup output window for the iterator
        Window window_out = window;
        window_out.set(Window::DimX, Window::Dimension(0, output->info()->dimension(Window::DimX), output->info()->dimension(Window::DimX)));
        window_out.set(Window::DimY, Window::Dimension(0, output->info()->dimension(Window::DimY), output->info()->dimension(Window::DimY)));
        window_out.set(Window::DimZ, Window::Dimension(window.z().start(), window.z().end(), num_planes_z));

        // setup input window for the iterator
        Window window_in = window;
        // we just want execute_window_loop to iterate over the higher dimensions (>3), so we set the first 3 dimensions to 0
        window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimY, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Window window_k = calculate_max_window(*weights->info(), Steps(1u));

        Iterator out(output, window_out);
        Iterator in(input, window_in);
        Iterator k(weights, window_k);

        const uint8_t *k_ptr = k.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            const uint8_t *input_ptr = in.ptr() - conv_pad_x * input_stride_x - conv_pad_y * input_stride_y;
            uint8_t       *out_ptr   = out.ptr();
            int            ih        = 0;
            int            oh        = 0;
            /*
                    Each thread executing this kernel computes one or more output's volume planes.

                    Let's say the 3rd dimension of the output volume is 32, the first thread will compute the output for Z = [0,7], the second thread will compute the output for Z = [8,15],
                    the third thread [16,24] and the fourth thread [25,31].

                    The algorithm outer loop iterates over Z, P, Y, X where P is the depth/3rd dimension of each kernel. This order is not arbitrary, the main benefit of this
                    is that we setup the neon registers containing the kernerl's values only once and then compute each XY using the preloaded registers as opposed as doing this for every XY value.

                    The algorithm does not require allocating any additional memory amd computes the results directly in-place in two stages:
                        1) Convolve plane 0 with kernel 0 and initialize the corresponding output plane with these values.
                        2) Convolve the remaining planes and accumulate the results in the output's plane which has been initialized in step 1.
            */

            for(int oz = 0; oz < num_planes_z; ++oz)
            {
                uint8_t *p_out_base = out_ptr + oz * output_stride_z;
                // Step 1
                {
                    const auto ptr_k_r0 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 0 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r1 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 1 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r2 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 2 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto vk_r0    = load_matrix_row(ptr_k_r0);
                    const auto vk_r1    = load_matrix_row(ptr_k_r1);
                    const auto vk_r2    = load_matrix_row(ptr_k_r2);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        auto in_top = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 0) * input_stride_y);
                        auto in_mid = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 1) * input_stride_y);
                        auto in_low = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 2) * input_stride_y);
                        auto p_out  = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                            in_top += delta_input, in_mid += delta_input, in_low += delta_input, p_out += num_elems_written_per_iteration)
                        {
                            auto vres = convolve_3x3<stridex>(in_top, in_mid, in_low, vk_r0, vk_r1, vk_r2, fixed_point_position);
                            store_results<stridex>(p_out, vres);
                        }
                    }
                }
                // Step 2
                for(int p = 1; p < kernel_depth; ++p)
                {
                    const auto ptr_k_r0 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 0 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r1 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 1 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r2 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + (id.z() + oz) * kernel_stride_w + 2 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto vk_r0    = load_matrix_row(ptr_k_r0);
                    const auto vk_r1    = load_matrix_row(ptr_k_r1);
                    const auto vk_r2    = load_matrix_row(ptr_k_r2);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        auto in_top = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 0) * input_stride_y);
                        auto in_mid = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 1) * input_stride_y);
                        auto in_low = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 2) * input_stride_y);
                        auto p_out  = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                            in_top += delta_input, in_mid += delta_input, in_low += delta_input, p_out += num_elems_written_per_iteration)
                        {
                            auto vres = convolve_3x3<stridex>(in_top, in_mid, in_low, vk_r0, vk_r1, vk_r2, fixed_point_position);
                            accumulate_results<stridex>(p_out, vres);
                        }
                    }
                }
            }
        },
        in, out);
    }
};

template <typename T1, typename T2>
inline void convolve_1x1(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_1x1<T1, T2, 1>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 2:
            convolver_1x1<T1, T2, 2>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 3:
            convolver_1x1<T1, T2, 3>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

template <typename T1, typename T2>
inline void convolve_3x3(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_3x3<T1, T2, 1>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 2:
            convolver_3x3<T1, T2, 2>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 3:
            convolver_3x3<T1, T2, 3>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace

NEDirectConvolutionLayerKernel::NEDirectConvolutionLayerKernel()
    : _input(nullptr), _weights(nullptr), _output(nullptr), _conv_info(), _border_size(0), _kernel_size(0), _num_elems_read_per_iteration(0), _num_elems_written_per_iteration(0)
{
}

BorderSize NEDirectConvolutionLayerKernel::border_size() const
{
    return _border_size;
}

void NEDirectConvolutionLayerKernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MSG(weights->info()->dimension(0) == 1 && (std::get<0>(conv_info.pad()) || std::get<1>(conv_info.pad())),
                             "Pad > 0 not supported for 1x1 weights");
    ARM_COMPUTE_ERROR_ON_MSG(weights->info()->dimension(0) == 3 && (std::get<0>(conv_info.pad()) > 1 || std::get<1>(conv_info.pad()) > 1),
                             "Pad > 1 not supported for 3x3 weights");
    ARM_COMPUTE_ERROR_ON_MSG(std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported.");

    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    const unsigned int conv_pad_x    = std::get<0>(conv_info.pad());
    const unsigned int conv_pad_y    = std::get<1>(conv_info.pad());

    _input       = input;
    _weights     = weights;
    _output      = output;
    _conv_info   = conv_info;
    _kernel_size = weights->info()->dimension(0);
    _border_size = BorderSize(conv_pad_y, conv_pad_x);

    Window win = calculate_max_window(*output->info());

    switch(_kernel_size)
    {
        case 1:
        {
            _num_elems_written_per_iteration = (input->info()->data_type() == DataType::QS8) ? 8 : 4;
            _num_elems_read_per_iteration    = conv_stride_x * _num_elems_written_per_iteration;

            win = calculate_max_window(*output->info(), Steps(_num_elems_written_per_iteration));
            AccessWindowHorizontal input_access(input->info(), 0, _num_elems_read_per_iteration);
            AccessWindowHorizontal output_access(output->info(), 0, _num_elems_written_per_iteration);
            update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
            break;
        }
        case 3:
        {
            if(input->info()->data_type() == DataType::F32)
            {
                _num_elems_read_per_iteration    = 12;
                _num_elems_written_per_iteration = 16 >> conv_stride_x;
            }
            else
            {
                _num_elems_read_per_iteration    = 24;
                _num_elems_written_per_iteration = 32 >> conv_stride_x;
            }

            // Calculate right and bottom border
            const unsigned int conv_stride_y = std::get<1>(_conv_info.stride());
            const int          input_width   = input->info()->dimension(0);
            const int          input_height  = input->info()->dimension(1);
            const int          upper_bound_w = ceil_to_multiple(((output->info()->dimension(0) - 1) * conv_stride_x + _kernel_size), _num_elems_read_per_iteration) - conv_pad_x - input_width;
            const int          upper_bound_h = ((output->info()->dimension(1) - 1) * conv_stride_y - conv_pad_y + _kernel_size) - input_height;
            _border_size.right               = std::max(upper_bound_w, static_cast<int>(_kernel_size));
            _border_size.bottom              = std::max(upper_bound_h, static_cast<int>(_kernel_size));

            // Create window and update padding
            win = calculate_max_window(*output->info(), Steps(_num_elems_written_per_iteration));
            AccessWindowStatic     input_access(input->info(), -conv_pad_x, -conv_pad_y, input_width + _border_size.right, input_height + _border_size.bottom);
            AccessWindowStatic     weights_access(weights->info(), 0, 0, _kernel_size, _kernel_size);
            AccessWindowHorizontal output_access(output->info(), 0, _num_elems_written_per_iteration);
            update_window_and_padding(win, input_access, weights_access, output_access);
            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not implemented");
            break;
        }
    }

    INEKernel::configure(win);
}

void NEDirectConvolutionLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    const int kernel_size = _weights->info()->dimension(0);

    switch(kernel_size)
    {
        case 1:
        {
            if(_input->info()->data_type() == DataType::QS8)
            {
                convolve_1x1<qint8_t, qint16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            }
            else
            {
                convolve_1x1<float, float>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            }
            break;
        }
        case 3:
        {
            if(_input->info()->data_type() == DataType::QS8)
            {
                convolve_3x3<qint8_t, qint16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            }
            else
            {
                convolve_3x3<float, float>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Only kernel sizes 1x1 and 3x3 are supported.");
            break;
        }
    }
}
