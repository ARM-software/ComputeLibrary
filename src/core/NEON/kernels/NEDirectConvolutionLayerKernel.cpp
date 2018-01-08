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
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"
#include "arm_compute/core/NEON/kernels/convolution/NEDirectConvolutionDetail.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>

using namespace arm_compute;
using namespace arm_compute::detail;

namespace
{
template <unsigned int stridex>
qint16x8_t internal_vld1q(const qint16_t *in);

template <>
qint16x8_t internal_vld1q<1>(const qint16_t *in)
{
    return vld1q_qs16(in);
}

template <>
qint16x8_t internal_vld1q<2>(const qint16_t *in)
{
    const int16x8x2_t tmp = vld2q_s16(in);
    return tmp.val[0];
}

template <>
qint16x8_t internal_vld1q<3>(const qint16_t *in)
{
    const int16x8x3_t tmp = vld3q_s16(in);
    return tmp.val[0];
}

inline qint16x8_t internal_vdupq_n(qint16_t v)
{
    return vdupq_n_qs16(v);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <unsigned int stridex>
float16x8_t internal_vld1q(const float16_t *in);

template <>
float16x8_t internal_vld1q<1>(const float16_t *in)
{
    return vld1q_f16(in);
}

template <>
float16x8_t internal_vld1q<2>(const float16_t *in)
{
    const float16x8x2_t tmp = vld2q_f16(in);
    return tmp.val[0];
}

template <>
float16x8_t internal_vld1q<3>(const float16_t *in)
{
    const float16x8x3_t tmp = vld3q_f16(in);
    return tmp.val[0];
}

inline float16x8_t internal_vdupq_n(float16_t v)
{
    return vdupq_n_f16(v);
}

inline void internal_vst1q(float16_t *p, const float16x8_t &v)
{
    vst1q_f16(p, v);
}

float16x8_t internal_vmull(const float16x8_t &x, const float16x8_t &y, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vmulq_f16(x, y);
}

inline float16x8_t internal_vmlal(const float16x8_t &x, const float16x8_t &y, const float16x8_t &z, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vaddq_f16(x, vmulq_f16(y, z));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

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

inline float32x4_t internal_vdupq_n(float v)
{
    return vdupq_n_f32(v);
}

inline void internal_vst1q(float *p, const float32x4_t &v)
{
    vst1q_f32(p, v);
}

float32x4_t internal_vmull(const float32x4_t &x, const float32x4_t &y, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vmulq_f32(x, y);
}

inline float32x4_t internal_vmlal(const float32x4_t &x, const float32x4_t &y, const float32x4_t &z, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    return vmlaq_f32(x, y, z);
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

inline qint8x8_t internal_vdupq_n(qint8_t v)
{
    return vdup_n_qs8(v);
}

inline qint16x8_t internal_vmull(const qint8x8_t &x, const qint8x8_t &y, int fixed_point_position)
{
    return vmull_qs8(x, y, fixed_point_position);
}

inline qint16x8_t internal_vmlal(const qint16x8_t &x, const qint8x8_t &y, const qint8x8_t &z, int fixed_point_position)
{
    return vqmlal_qs8(x, y, z, fixed_point_position);
}

inline void internal_vst1q(qint16_t *p, const qint16x8_t &v)
{
    vst1q_qs16(p, v);
}

inline void internal_vst1q(int32_t *p, const qint32x4x2_t &v)
{
    vst1q_s32(p, v.val[0]);
    vst1q_s32(p + 4, v.val[1]);
}

template <unsigned int stridex>
qint32x4x2_t internal_vld1q(const qint32_t *in);

template <>
qint32x4x2_t internal_vld1q<1>(const qint32_t *in)
{
    const qint32x4x2_t r =
    {
        {
            vld1q_s32(in),
            vld1q_s32(in + 4)
        }
    };
    return r;
}

inline qint32x4x2_t internal_vmull(const qint16x8_t &x, const qint16x8_t &y, int fixed_point_position)
{
    const qint32x4x2_t r =
    {
        {
            vmull_qs16(vget_low_s16(x), vget_low_s16(y), fixed_point_position),
            vmull_qs16(vget_high_s16(x), vget_high_s16(y), fixed_point_position),
        }
    };
    return r;
}

inline qint32x4x2_t internal_vmlal(const qint32x4x2_t &x, const qint16x8_t &y, const qint16x8_t &z, int fixed_point_position)
{
    const qint32x4x2_t r =
    {
        {
            vqmlal_qs16(x.val[0], vget_low_s16(y), vget_low_s16(z), fixed_point_position),
            vqmlal_qs16(x.val[1], vget_high_s16(y), vget_high_s16(z), fixed_point_position)
        }
    };
    return r;
}

constexpr int small_tensor_size_optim = 8;
inline bool run_optim_small_tensor_info(const ITensorInfo *t)
{
    return t->dimension(Window::DimX) <= small_tensor_size_optim && t->dimension(Window::DimY) <= small_tensor_size_optim;
}

inline bool run_optim_small_tensor(const ITensor *t)
{
    return run_optim_small_tensor_info(t->info());
}

// Optimized convolver for 1x1 kernels used only where input width and height are both <= 8
// For big Z as in Input=7x7x832, this implementation is faster than the general code becuase it doesn't need to
// store intermidiate results in memory. Temporary results are stored in NEON registers directly and then written to the output buffer.
template <unsigned int stridex>
class convolver_w1x1_i8x8_f32
{
public:
    static void convolve(const Window &window, const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(Window::DimX) > small_tensor_size_optim);
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(Window::DimY) > small_tensor_size_optim);

        const int          input_stride_y  = input->info()->strides_in_bytes().y();
        const int          input_stride_z  = input->info()->strides_in_bytes().z();
        const int          output_stride_y = output->info()->strides_in_bytes().y();
        const int          output_stride_z = output->info()->strides_in_bytes().z();
        const int          kernel_stride_z = weights->info()->strides_in_bytes().z();
        const int          kernel_stride_w = weights->info()->strides_in_bytes()[3];
        const int          output_h        = output->info()->dimension(1);
        const int          range_z         = window.z().end() - window.z().start();
        const int          kernel_depth    = weights->info()->dimension(Window::DimZ);
        const unsigned int conv_stride_y   = std::get<1>(conv_info.stride());

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

        Window   window_k = calculate_max_window(*weights->info(), Steps(1u));
        Iterator out(output, window_out);
        Iterator in(input, window_in);
        Iterator k(weights, window_k);

        const uint8_t *k_ptr = k.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            const uint8_t *input_ptr                       = in.ptr();
            uint8_t       *out_ptr                         = out.ptr();
            int            ih                              = 0;
            int            oh                              = 0;
            float32x4_t    accum0[small_tensor_size_optim] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t    accum1[small_tensor_size_optim] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            for(int oz = 0; oz < range_z; ++oz)
            {
                accum0[0] = accum0[1] = accum0[2] = accum0[3] = accum0[4] = accum0[5] = accum0[6] = accum0[7] = vdupq_n_f32(0.f);
                accum1[0] = accum1[1] = accum1[2] = accum1[3] = accum1[4] = accum1[5] = accum1[6] = accum1[7] = vdupq_n_f32(0.f);
                auto p_out_base                                                                               = out_ptr + oz * output_stride_z;
                for(int p = 0; p < kernel_depth; ++p)
                {
                    const auto k_val = reinterpret_cast<const float *>(k_ptr + p * kernel_stride_z + (id.z() + oz) * kernel_stride_w);
                    const auto vk0   = internal_vdupq_n(*k_val);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        const int offset_xy = ih * input_stride_y;
                        auto      in_val    = reinterpret_cast<const float *>(input_ptr + p * input_stride_z + offset_xy);
                        auto      v_in0     = internal_vld1q<stridex>(in_val);
                        auto      v_in1     = internal_vld1q<stridex>(in_val + 4);
                        accum0[oh]          = vmlaq_f32(accum0[oh], vk0, v_in0);
                        accum1[oh]          = vmlaq_f32(accum1[oh], vk0, v_in1);
                    }
                }
                for(oh = 0; oh < output_h; ++oh)
                {
                    auto p_out = reinterpret_cast<float *>(p_out_base + oh * output_stride_y);
                    vst1q_f32(p_out, accum0[oh]);
                    vst1q_f32(p_out + 4, accum1[oh]);
                }
            }
        },
        in, out);
    }
};

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

        Window   window_k = calculate_max_window(*weights->info(), Steps(1u));
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

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <unsigned int stridex>
void accumulate_results(float16_t *buffer, const float16x8x2_t &values);

template <>
void accumulate_results<1>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, vaddq_f16(vld1q_f16(buffer), values.val[0]));
    vst1q_f16(buffer + 8, vaddq_f16(vld1q_f16(buffer + 8), values.val[1]));
}

template <>
void accumulate_results<2>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1q_f16(buffer, vaddq_f16(vld1q_f16(buffer), values.val[0]));
}

template <>
void accumulate_results<3>(float16_t *buffer, const float16x8x2_t &values)
{
    vst1_f16(buffer, vadd_f16(vld1_f16(buffer), vget_low_f16(values.val[0])));
}

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <unsigned int stridex>
float32x4x2_t convolve_5x5(const float *in_0, const float *in_1, const float *in_2, const float *in_3, const float *in_4,
                           const float *m0, const float *m1, const float *m2, const float *m3, const float *m4, int fixed_point_position);

inline float32x4x3_t load_matrix_hi(const float *const m0, const float *const m1, const float *const m2)
{
    const float32x4x3_t m00 =
    {
        {
            vld1q_dup_f32(m0),
            vld1q_dup_f32(m1),
            vld1q_dup_f32(m2)
        }
    };
    return m00;
}

inline float32x4x2_t load_matrix_lo(const float *const m3, const float *const m4)
{
    const float32x4x2_t m00 =
    {
        {
            vld1q_dup_f32(m3),
            vld1q_dup_f32(m4)
        }
    };
    return m00;
}

inline float32x4x3_t load_input(const float *const in)
{
    const float32x4x3_t vin =
    {
        {
            vld1q_f32(in),
            vld1q_f32(in + 4),
            vld1q_f32(in + 8)
        }
    };
    return vin;
}

template <>
inline float32x4x2_t convolve_5x5<1>(const float *in_0, const float *in_1, const float *in_2, const float *in_3, const float *in_4,
                                     const float *m0, const float *m1, const float *m2, const float *m3, const float *m4, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    const float32x4x3_t vin0 = load_input(in_0);
    const float32x4x3_t vin1 = load_input(in_1);
    const float32x4x3_t vin2 = load_input(in_2);
    const float32x4x3_t vin3 = load_input(in_3);
    const float32x4x3_t vin4 = load_input(in_4);
    const float32x4x3_t m00  = load_matrix_hi(m0, 1 + m0, 2 + m0);
    const float32x4x2_t m01  = load_matrix_lo(3 + m0, 4 + m0);
    const float32x4x3_t m10  = load_matrix_hi(m1, 1 + m1, 2 + m1);
    const float32x4x2_t m11  = load_matrix_lo(3 + m1, 4 + m1);
    const float32x4x3_t m20  = load_matrix_hi(m2, 1 + m2, 2 + m2);
    const float32x4x2_t m21  = load_matrix_lo(3 + m2, 4 + m2);
    const float32x4x3_t m30  = load_matrix_hi(m3, 1 + m3, 2 + m3);
    const float32x4x2_t m31  = load_matrix_lo(3 + m3, 4 + m3);
    const float32x4x3_t m40  = load_matrix_hi(m4, 1 + m4, 2 + m4);
    const float32x4x2_t m41  = load_matrix_lo(3 + m4, 4 + m4);

    float32x4x2_t out =
    {
        {
            vmulq_f32(vin0.val[0], m00.val[0]),
            vmulq_f32(vin0.val[1], m00.val[0])
        }
    };

    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 1), m00.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 2), m00.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin0.val[0], vin0.val[1], 3), m01.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin0.val[1], m01.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin1.val[0], m10.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 1), m10.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 2), m10.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin1.val[0], vin1.val[1], 3), m11.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin1.val[1], m11.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin2.val[0], m20.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 1), m20.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 2), m20.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin2.val[0], vin2.val[1], 3), m21.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin2.val[1], m21.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin3.val[0], m30.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 1), m30.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 2), m30.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin3.val[0], vin3.val[1], 3), m31.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin3.val[1], m31.val[1]);

    out.val[0] = vmlaq_f32(out.val[0], vin4.val[0], m40.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 1), m40.val[1]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 2), m40.val[2]);
    out.val[0] = vmlaq_f32(out.val[0], vextq_f32(vin4.val[0], vin4.val[1], 3), m41.val[0]);
    out.val[0] = vmlaq_f32(out.val[0], vin4.val[1], m41.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 1), m00.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 2), m00.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin0.val[1], vin0.val[2], 3), m01.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin0.val[2], m01.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin1.val[1], m10.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 1), m10.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 2), m10.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin1.val[1], vin1.val[2], 3), m11.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin1.val[2], m11.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin2.val[1], m20.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 1), m20.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 2), m20.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin2.val[1], vin2.val[2], 3), m21.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin2.val[2], m21.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin3.val[1], m30.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 1), m30.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 2), m30.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin3.val[1], vin3.val[2], 3), m31.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin3.val[2], m31.val[1]);

    out.val[1] = vmlaq_f32(out.val[1], vin4.val[1], m40.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 1), m40.val[1]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 2), m40.val[2]);
    out.val[1] = vmlaq_f32(out.val[1], vextq_f32(vin4.val[1], vin4.val[2], 3), m41.val[0]);
    out.val[1] = vmlaq_f32(out.val[1], vin4.val[2], m41.val[1]);

    return out;
}

template <>
inline float32x4x2_t convolve_5x5<2>(const float *in_0, const float *in_1, const float *in_2, const float *in_3, const float *in_4,
                                     const float *m0, const float *m1, const float *m2, const float *m3, const float *m4, int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);
    float32x4x2_t out = convolve_5x5<1>(in_0, in_1, in_2, in_3, in_4, m0, m1, m2, m3, m4, fixed_point_position);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 2), out.val[0], 1);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 0), out.val[0], 2);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[1], 2), out.val[0], 3);
    return out;
}

template <>
inline float32x4x2_t convolve_5x5<3>(const float *in_0, const float *in_1, const float *in_2, const float *in_3, const float *in_4,
                                     const float *m0, const float *m1, const float *m2, const float *m3, const float *m4, int fixed_point_position)
{
    float32x4x2_t out = convolve_5x5<1>(in_0, in_1, in_2, in_3, in_4, m0, m1, m2, m3, m4, fixed_point_position);
    out.val[0]        = vsetq_lane_f32(vgetq_lane_f32(out.val[0], 3), out.val[0], 1);
    return out;
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
                    is that we setup the neon registers containing the kernel's values only once and then compute each XY using the preloaded registers as opposed as doing this for every XY value.

                    The algorithm does not require allocating any additional memory amd computes the results directly in-place in two stages:
                        1) Convolve plane 0 with kernel 0 and initialize the corresponding output plane with these values.
                        2) Convolve the remaining planes and accumulate the results in the output's plane which has been initialized in step 1.
            */
            for(int oz = 0; oz < num_planes_z; ++oz)
            {
                const int zoffset    = id.z() + oz;
                uint8_t *p_out_base = out_ptr + oz * output_stride_z;
                // Step 1
                {
                    const auto ptr_k_r0 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 0 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r1 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 1 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r2 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 2 * kernel_stride_y + 0 * kernel_stride_x);
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
                    const uint8_t *ptr_k_base = k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w;
                    const uint8_t *input_base = input_ptr + p * input_stride_z;
                    const auto     ptr_k_r0   = reinterpret_cast<const T1 *>(ptr_k_base);
                    const auto     ptr_k_r1   = reinterpret_cast<const T1 *>(ptr_k_base + kernel_stride_y);
                    const auto     ptr_k_r2   = reinterpret_cast<const T1 *>(ptr_k_base + kernel_stride_y * 2);
                    const auto     vk_r0      = load_matrix_row(ptr_k_r0);
                    const auto     vk_r1      = load_matrix_row(ptr_k_r1);
                    const auto     vk_r2      = load_matrix_row(ptr_k_r2);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        auto in_top = reinterpret_cast<const T1 *>(input_base + (ih + 0) * input_stride_y);
                        auto in_mid = reinterpret_cast<const T1 *>(input_base + (ih + 1) * input_stride_y);
                        auto in_low = reinterpret_cast<const T1 *>(input_base + (ih + 2) * input_stride_y);
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

template <typename T1, typename T2, unsigned int stridex>
class convolver_5x5
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
            for(int oz = 0; oz < num_planes_z; ++oz)
            {
                const int zoffset    = id.z() + oz;
                uint8_t *p_out_base = out_ptr + oz * output_stride_z;
                // Step 1
                {
                    const auto ptr_k_r0 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 0 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r1 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 1 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r2 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 2 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r3 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 3 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r4 = reinterpret_cast<const T1 *>(k_ptr + 0 * kernel_stride_z + zoffset * kernel_stride_w + 4 * kernel_stride_y + 0 * kernel_stride_x);
                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        auto in_0  = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 0) * input_stride_y);
                        auto in_1  = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 1) * input_stride_y);
                        auto in_2  = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 2) * input_stride_y);
                        auto in_3  = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 3) * input_stride_y);
                        auto in_4  = reinterpret_cast<const T1 *>(input_ptr + 0 * input_stride_z + (ih + 4) * input_stride_y);
                        auto p_out = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                            in_0 += delta_input, in_1 += delta_input, in_2 += delta_input, in_3 += delta_input, in_4 += delta_input, p_out += num_elems_written_per_iteration)
                        {
                            auto vres = convolve_5x5<stridex>(in_0, in_1, in_2, in_3, in_4, ptr_k_r0, ptr_k_r1, ptr_k_r2, ptr_k_r3, ptr_k_r4, fixed_point_position);
                            store_results<stridex>(p_out, vres);
                        }
                    }
                }
                // Step 2
                for(int p = 1; p < kernel_depth; ++p)
                {
                    const auto ptr_k_r0 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w + 0 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r1 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w + 1 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r2 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w + 2 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r3 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w + 3 * kernel_stride_y + 0 * kernel_stride_x);
                    const auto ptr_k_r4 = reinterpret_cast<const T1 *>(k_ptr + p * kernel_stride_z + zoffset * kernel_stride_w + 4 * kernel_stride_y + 0 * kernel_stride_x);

                    for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
                    {
                        auto in_0  = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 0) * input_stride_y);
                        auto in_1  = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 1) * input_stride_y);
                        auto in_2  = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 2) * input_stride_y);
                        auto in_3  = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 3) * input_stride_y);
                        auto in_4  = reinterpret_cast<const T1 *>(input_ptr + p * input_stride_z + (ih + 4) * input_stride_y);
                        auto p_out = reinterpret_cast<T2 *>(p_out_base + oh * output_stride_y);
                        for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                            in_0 += delta_input, in_1 += delta_input, in_2 += delta_input, in_3 += delta_input, in_4 += delta_input, p_out += num_elems_written_per_iteration)
                        {
                            auto vres = convolve_5x5<stridex>(in_0, in_1, in_2, in_3, in_4, ptr_k_r0, ptr_k_r1, ptr_k_r2, ptr_k_r3, ptr_k_r4, fixed_point_position);
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

template <>
inline void convolve_1x1<float, float>(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                                       const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    if(run_optim_small_tensor(input))
    {
        switch(conv_stride_x)
        {
            case 1:
                convolver_w1x1_i8x8_f32<1>::convolve(window, input, weights, output, conv_info);
                break;
            case 2:
                convolver_w1x1_i8x8_f32<2>::convolve(window, input, weights, output, conv_info);
                break;
            case 3:
                convolver_w1x1_i8x8_f32<3>::convolve(window, input, weights, output, conv_info);
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
    }
    else
    {
        switch(conv_stride_x)
        {
            case 1:
                convolver_1x1<float, float, 1>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
                break;
            case 2:
                convolver_1x1<float, float, 2>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
                break;
            case 3:
                convolver_1x1<float, float, 3>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
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

template <typename T1, typename T2>
inline void convolve_5x5(const Window &window, unsigned int num_elems_read_per_iteration, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_5x5<T1, T2, 1>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 2:
            convolver_5x5<T1, T2, 2>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 3:
            convolver_5x5<T1, T2, 3>::convolve(window, num_elems_read_per_iteration, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

inline TensorShape get_convolved_dimensions(const ITensorInfo *input, const ITensorInfo *weights, const int kernel_size, const PadStrideInfo &conv_info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input->dimension(0), input->dimension(1), kernel_size, kernel_size, conv_info);

    TensorShape output_shape = input->tensor_shape();
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights->dimension(3));

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) == 1 && (std::get<0>(conv_info.pad()) || std::get<1>(conv_info.pad())),
                                    "Pad > 0 not supported for 1x1 weights");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) == 3 && (std::get<0>(conv_info.pad()) > 1 || std::get<1>(conv_info.pad()) > 1),
                                    "Pad > 1 not supported for 3x3 weights");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) == 5 && (std::get<0>(conv_info.pad()) > 2 || std::get<1>(conv_info.pad()) > 2),
                                    "Pad > 2 not supported for 5x5 weights");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported.");
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) != input->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        TensorShape output_shape = get_convolved_dimensions(input, weights, weights->dimension(0), conv_info);

        DataType data_type = input->data_type();
        if(is_data_type_fixed_point(data_type))
        {
            // Promote data type in case of fixed point
            data_type = ((data_type == DataType::QS8) ? DataType::QS16 : DataType::QS32);
        }

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(output->data_type() != data_type);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int &num_weight_elems_read_per_row,
                                                        unsigned int &num_elems_read_per_iteration, unsigned int &num_elems_written_per_iteration, BorderSize &border_size)
{
    // Calculate right and bottom border
    unsigned int       kernel_size   = weights->dimension(0);
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    const unsigned int conv_stride_y = std::get<1>(conv_info.stride());
    const int          input_width   = input->dimension(0);
    const int          input_height  = input->dimension(1);

    switch(kernel_size)
    {
        case 1:
        {
            switch(input->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                case DataType::QS8:
                case DataType::QS16:
                    num_elems_written_per_iteration = 8;
                    break;
                case DataType::F32:
                    if(run_optim_small_tensor_info(input))
                    {
                        num_elems_written_per_iteration = 8;
                    }
                    else
                    {
                        num_elems_written_per_iteration = 4;
                    }
                    break;
                default:
                    ARM_COMPUTE_ERROR("Data type not supported.");
                    break;
            }
            num_weight_elems_read_per_row = kernel_size;
            num_elems_read_per_iteration  = conv_stride_x * num_elems_written_per_iteration;
            break;
        }
        case 3:
        case 5:
        {
            switch(input->data_type())
            {
                case DataType::F32:
                    num_weight_elems_read_per_row   = 4 + kernel_size - 1;
                    num_elems_read_per_iteration    = 12;
                    num_elems_written_per_iteration = 16 >> conv_stride_x;
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                case DataType::QS8:
                case DataType::QS16:
                    num_weight_elems_read_per_row   = 8 + kernel_size - 1;
                    num_elems_read_per_iteration    = 24;
                    num_elems_written_per_iteration = 32 >> conv_stride_x;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Data type not supported.");
                    break;
            }
        }
        break;
        default:
        {
            ARM_COMPUTE_ERROR("Not implemented");
            break;
        }
    }

    // Calculate border
    int upper_bound_w = ceil_to_multiple(((output->dimension(0) - 1) * conv_stride_x + kernel_size), num_elems_read_per_iteration) - conv_info.pad_left() - conv_info.pad_right() - input_width;
    int upper_bound_h = ((output->dimension(1) - 1) * conv_stride_y - conv_info.pad_top() - conv_info.pad_bottom() + kernel_size) - input_height;

    const unsigned int conv_pad_left   = std::max(upper_bound_w - static_cast<int>(conv_info.pad_right()), static_cast<int>(kernel_size) / 2);
    const unsigned int conv_pad_top    = std::max(upper_bound_h - static_cast<int>(conv_info.pad_bottom()), static_cast<int>(kernel_size) / 2);
    const unsigned int conv_pad_right  = std::max(upper_bound_w - static_cast<int>(conv_info.pad_left()), static_cast<int>(kernel_size) / 2);
    const unsigned int conv_pad_bottom = std::max(upper_bound_h - static_cast<int>(conv_info.pad_top()), static_cast<int>(kernel_size) / 2);

    border_size.right  = conv_pad_right;
    border_size.bottom = conv_pad_bottom;
    border_size.left   = conv_pad_left;
    border_size.top    = conv_pad_top;

    Window                 win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));
    AccessWindowStatic     input_access(input, -conv_pad_left, -conv_pad_top, input_width + conv_pad_right, input_height + conv_pad_bottom);
    AccessWindowStatic     weights_access(weights, 0, 0, num_weight_elems_read_per_row, kernel_size);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEDirectConvolutionLayerKernel::NEDirectConvolutionLayerKernel()
    : _input(nullptr), _weights(nullptr), _output(nullptr), _conv_info(), _border_size(0), _kernel_size(0), _num_weight_elems_read_per_row(0), _num_elems_read_per_iteration(0),
      _num_elems_written_per_iteration(0)
{
}

BorderSize NEDirectConvolutionLayerKernel::border_size() const
{
    return _border_size;
}

void NEDirectConvolutionLayerKernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    _input       = input;
    _weights     = weights;
    _output      = output;
    _conv_info   = conv_info;
    _kernel_size = weights->info()->dimension(0);

    const unsigned int conv_pad_left   = conv_info.pad_left();
    const unsigned int conv_pad_top    = conv_info.pad_top();
    const unsigned int conv_pad_right  = conv_info.pad_right();
    const unsigned int conv_pad_bottom = conv_info.pad_bottom();
    _border_size                       = BorderSize(conv_pad_top, conv_pad_right, conv_pad_bottom, conv_pad_left);

    // Get convolved dimensions
    TensorShape output_shape = get_convolved_dimensions(input->info(), weights->info(), _kernel_size, conv_info);

    DataType data_type = input->info()->data_type();

    if(is_data_type_fixed_point(data_type))
    {
        // Promote data type in case of fixed point
        data_type = ((data_type == DataType::QS8) ? DataType::QS16 : DataType::QS32);
    }

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, data_type, input->info()->fixed_point_position());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), output->info(), conv_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), output->info(), conv_info, _num_weight_elems_read_per_row,
                                                    _num_elems_read_per_iteration, _num_elems_written_per_iteration, _border_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEDirectConvolutionLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    unsigned int num_weight_elems_read_per_row   = 0;
    unsigned int num_elems_read_per_iteration    = 0;
    unsigned int num_elems_written_per_iteration = 0;
    BorderSize   border_size(conv_info.pad().first, conv_info.pad().second);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, output, conv_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              weights->clone().get(),
                                                              output->clone().get(),
                                                              conv_info,
                                                              num_weight_elems_read_per_row,
                                                              num_elems_read_per_iteration,
                                                              num_elems_written_per_iteration,
                                                              border_size)
                                .first);

    return Status{};
}

void NEDirectConvolutionLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    const int kernel_size = _weights->info()->dimension(0);

    switch(kernel_size)
    {
        case 1:
        {
            switch(_input->info()->data_type())
            {
                case DataType::QS8:
                    convolve_1x1<qint8_t, qint16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
                case DataType::QS16:
                    convolve_1x1<qint16_t, qint32_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
                case DataType::F32:
                    convolve_1x1<float, float>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    convolve_1x1<float16_t, float16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Data type not supported");
                    break;
            }
            break;
        }
        case 3:
        {
            switch(_input->info()->data_type())
            {
                case DataType::QS8:
                    convolve_3x3<qint8_t, qint16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
                case DataType::F32:
                    convolve_3x3<float, float>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    convolve_3x3<float16_t, float16_t>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Data type not supported");
                    break;
            }
            break;
        }
        case 5:
        {
            switch(_input->info()->data_type())
            {
                case DataType::F32:
                    convolve_5x5<float, float>(window, _num_elems_read_per_iteration, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Data type not supported");
                    break;
            }
            break;
        }

        default:
        {
            ARM_COMPUTE_ERROR("Only kernel sizes 1x1, 3x3 and 5x5 are supported.");
            break;
        }
    }
}
