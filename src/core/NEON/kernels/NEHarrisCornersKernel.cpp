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
#include "arm_compute/core/NEON/kernels/NEHarrisCornersKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstddef>

using namespace arm_compute;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template class arm_compute::NEHarrisScoreFP16Kernel<3>;
template class arm_compute::NEHarrisScoreFP16Kernel<5>;
template class arm_compute::NEHarrisScoreFP16Kernel<7>;

namespace fp16
{
inline float16x8_t harris_score(float16x8_t gx2, float16x8_t gy2, float16x8_t gxgy, float sensitivity, float strength_thresh)
{
    static const float16x8_t zero = vdupq_n_f16(0.f);

    // Trace^2
    float16x8_t trace2 = vaddq_f16(gx2, gy2);
    trace2             = vmulq_f16(trace2, trace2);

    // Det(A)
    float16x8_t det = vmulq_f16(gx2, gy2);
    det             = vfmsq_f16(det, gxgy, gxgy);

    // Det(A) - sensitivity * trace^2
    const float16x8_t mc = vfmsq_f16(det, vdupq_n_f16(sensitivity), trace2);

    // mc > strength_thresh
    const uint16x8_t mask = vcgtq_f16(mc, vdupq_n_f16(strength_thresh));

    return vbslq_f16(mask, mc, zero);
}

template <size_t block_size>
inline void harris_score1xN_FLOAT_FLOAT_FLOAT(float16x8_t low_gx, float16x8_t low_gy, float16x8_t high_gx, float16x8_t high_gy, float16x8_t &gx2, float16x8_t &gy2, float16x8_t &gxgy,
                                              float norm_factor)
{
    const float16x8_t norm_factor_fp16 = vdupq_n_f16(norm_factor);

    // Normalize
    low_gx  = vmulq_f16(low_gx, norm_factor_fp16);
    low_gy  = vmulq_f16(low_gy, norm_factor_fp16);
    high_gx = vmulq_f16(high_gx, norm_factor_fp16);
    high_gy = vmulq_f16(high_gy, norm_factor_fp16);

    float16x8_t gx = vextq_f16(low_gx, high_gx, 0);
    float16x8_t gy = vextq_f16(low_gy, high_gy, 0);

    gx2  = vfmaq_f16(gx2, gx, gx);
    gy2  = vfmaq_f16(gy2, gy, gy);
    gxgy = vfmaq_f16(gxgy, gx, gy);

    gx = vextq_f16(low_gx, high_gx, 1);
    gy = vextq_f16(low_gy, high_gy, 1);

    gx2  = vfmaq_f16(gx2, gx, gx);
    gy2  = vfmaq_f16(gy2, gy, gy);
    gxgy = vfmaq_f16(gxgy, gx, gy);

    gx = vextq_f16(low_gx, high_gx, 2);
    gy = vextq_f16(low_gy, high_gy, 2);

    gx2  = vfmaq_f16(gx2, gx, gx);
    gy2  = vfmaq_f16(gy2, gy, gy);
    gxgy = vfmaq_f16(gxgy, gx, gy);

    if(block_size > 3)
    {
        gx = vextq_f16(low_gx, high_gx, 3);
        gy = vextq_f16(low_gy, high_gy, 3);

        gx2  = vfmaq_f16(gx2, gx, gx);
        gy2  = vfmaq_f16(gy2, gy, gy);
        gxgy = vfmaq_f16(gxgy, gx, gy);

        gx = vextq_f16(low_gx, high_gx, 4);
        gy = vextq_f16(low_gy, high_gy, 4);

        gx2  = vfmaq_f16(gx2, gx, gx);
        gy2  = vfmaq_f16(gy2, gy, gy);
        gxgy = vfmaq_f16(gxgy, gx, gy);
    }

    if(block_size == 7)
    {
        gx = vextq_f16(low_gx, high_gx, 5);
        gy = vextq_f16(low_gy, high_gy, 5);

        gx2  = vfmaq_f16(gx2, gx, gx);
        gy2  = vfmaq_f16(gy2, gy, gy);
        gxgy = vfmaq_f16(gxgy, gx, gy);

        gx = vextq_f16(low_gx, high_gx, 6);
        gy = vextq_f16(low_gy, high_gy, 6);

        gx2  = vfmaq_f16(gx2, gx, gx);
        gy2  = vfmaq_f16(gy2, gy, gy);
        gxgy = vfmaq_f16(gxgy, gx, gy);
    }
}

template <size_t block_size>
inline void harris_score_S16_S16_FLOAT(const void *__restrict in1_ptr, const void *__restrict in2_ptr, void *__restrict out_ptr, int32_t in_stride, float norm_factor, float sensitivity,
                                       float strength_thresh)
{
    auto           gx_ptr_0 = static_cast<const int16_t *__restrict>(in1_ptr) - (block_size / 2) * (in_stride + 1);
    auto           gy_ptr_0 = static_cast<const int16_t *__restrict>(in2_ptr) - (block_size / 2) * (in_stride + 1);
    const int16_t *gx_ptr_1 = gx_ptr_0 + 8;
    const int16_t *gy_ptr_1 = gy_ptr_0 + 8;
    const auto     output   = static_cast<float *__restrict>(out_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float16x8_t gx2  = vdupq_n_f16(0.0f);
    float16x8_t gy2  = vdupq_n_f16(0.0f);
    float16x8_t gxgy = vdupq_n_f16(0.0f);

    for(size_t i = 0; i < block_size; ++i)
    {
        const float16x8_t low_gx  = vcvtq_f16_s16(vld1q_s16(gx_ptr_0));
        const float16x8_t high_gx = vcvtq_f16_s16(vld1q_s16(gx_ptr_1));
        const float16x8_t low_gy  = vcvtq_f16_s16(vld1q_s16(gy_ptr_0));
        const float16x8_t high_gy = vcvtq_f16_s16(vld1q_s16(gy_ptr_1));
        harris_score1xN_FLOAT_FLOAT_FLOAT<block_size>(low_gx, low_gy, high_gx, high_gy, gx2, gy2, gxgy, norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += in_stride;
        gy_ptr_0 += in_stride;
        gx_ptr_1 += in_stride;
        gy_ptr_1 += in_stride;
    }

    // Calculate harris score
    const float16x8_t mc = harris_score(gx2, gy2, gxgy, sensitivity, strength_thresh);

    // Store score
    vst1q_f32(output + 0, vcvt_f32_f16(vget_low_f16(mc)));
    vst1q_f32(output + 4, vcvt_f32_f16(vget_high_f16(mc)));
}

template <size_t block_size>
inline void harris_score_S32_S32_FLOAT(const void *__restrict in1_ptr, const void *__restrict in2_ptr, void *__restrict out_ptr, int32_t in_stride, float norm_factor, float sensitivity,
                                       float strength_thresh)
{
    static const float16x8_t zero = vdupq_n_f16(0.0f);

    auto           gx_ptr_0 = static_cast<const int32_t *__restrict>(in1_ptr) - (block_size / 2) * (in_stride + 1);
    auto           gy_ptr_0 = static_cast<const int32_t *__restrict>(in2_ptr) - (block_size / 2) * (in_stride + 1);
    const int32_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int32_t *gy_ptr_1 = gy_ptr_0 + 4;
    const int32_t *gx_ptr_2 = gx_ptr_0 + 8;
    const int32_t *gy_ptr_2 = gy_ptr_0 + 8;
    const auto     output   = static_cast<float *__restrict>(out_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float16x8_t gx2  = zero;
    float16x8_t gy2  = zero;
    float16x8_t gxgy = zero;

    for(size_t i = 0; i < block_size; ++i)
    {
        const float16x8_t low_gx = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_0))),
                                                vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_1))));
        const float16x8_t high_gx = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_2))),
                                                 vget_low_f16(zero));
        const float16x8_t low_gy = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_0))),
                                                vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_1))));
        const float16x8_t high_gy = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_2))),
                                                 vget_low_f16(zero));
        harris_score1xN_FLOAT_FLOAT_FLOAT<block_size>(low_gx, low_gy, high_gx, high_gy, gx2, gy2, gxgy, norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += in_stride;
        gy_ptr_0 += in_stride;
        gx_ptr_1 += in_stride;
        gy_ptr_1 += in_stride;
        gx_ptr_2 += in_stride;
        gy_ptr_2 += in_stride;
    }

    // Calculate harris score
    const float16x8_t mc = harris_score(gx2, gy2, gxgy, sensitivity, strength_thresh);

    // Store score
    vst1q_f32(output + 0, vcvt_f32_f16(vget_low_f16(mc)));
    vst1q_f32(output + 4, vcvt_f32_f16(vget_high_f16(mc)));
}

template <>
inline void harris_score_S32_S32_FLOAT<7>(const void *__restrict in1_ptr, const void *__restrict in2_ptr, void *__restrict out_ptr, int32_t in_stride, float norm_factor, float sensitivity,
                                          float strength_thresh)
{
    static const float16x8_t zero = vdupq_n_f16(0.0f);

    auto           gx_ptr_0 = static_cast<const int32_t *__restrict>(in1_ptr) - 3 * (in_stride + 1);
    auto           gy_ptr_0 = static_cast<const int32_t *__restrict>(in2_ptr) - 3 * (in_stride + 1);
    const int32_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int32_t *gy_ptr_1 = gy_ptr_0 + 4;
    const int32_t *gx_ptr_2 = gx_ptr_0 + 8;
    const int32_t *gy_ptr_2 = gy_ptr_0 + 8;
    const int32_t *gx_ptr_3 = gx_ptr_0 + 12;
    const int32_t *gy_ptr_3 = gy_ptr_0 + 12;
    const auto     output   = static_cast<float *__restrict>(out_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float16x8_t gx2  = zero;
    float16x8_t gy2  = zero;
    float16x8_t gxgy = zero;

    for(size_t i = 0; i < 7; ++i)
    {
        const float16x8_t low_gx = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_0))),
                                                vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_1))));
        const float16x8_t high_gx = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_2))),
                                                 vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gx_ptr_3))));
        const float16x8_t low_gy = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_0))),
                                                vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_1))));
        const float16x8_t high_gy = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_2))),
                                                 vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(gy_ptr_3))));
        harris_score1xN_FLOAT_FLOAT_FLOAT<7>(low_gx, low_gy, high_gx, high_gy, gx2, gy2, gxgy, norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += in_stride;
        gy_ptr_0 += in_stride;
        gx_ptr_1 += in_stride;
        gy_ptr_1 += in_stride;
        gx_ptr_2 += in_stride;
        gy_ptr_2 += in_stride;
    }

    // Calculate harris score
    const float16x8_t mc = harris_score(gx2, gy2, gxgy, sensitivity, strength_thresh);

    // Store score
    vst1q_f32(output + 0, vcvt_f32_f16(vget_low_f16(mc)));
    vst1q_f32(output + 4, vcvt_f32_f16(vget_high_f16(mc)));
}

} // namespace fp16

template <int32_t block_size>
BorderSize        NEHarrisScoreFP16Kernel<block_size>::border_size() const
{
    return _border_size;
}

template <int32_t block_size>
NEHarrisScoreFP16Kernel<block_size>::NEHarrisScoreFP16Kernel()
    : INEHarrisScoreKernel(), _func(nullptr)
{
}

template <int32_t block_size>
void NEHarrisScoreFP16Kernel<block_size>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    Iterator input1(_input1, window);
    Iterator input2(_input2, window);
    Iterator output(_output, window);

    const size_t input_stride = _input1->info()->strides_in_bytes()[1] / element_size_from_data_type(_input1->info()->data_type());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        (*_func)(input1.ptr(), input2.ptr(), output.ptr(), input_stride, _norm_factor, _sensitivity, _strength_thresh);
    },
    input1, input2, output);
}

template <int32_t block_size>
void NEHarrisScoreFP16Kernel<block_size>::configure(const IImage *input1, const IImage *input2, IImage *output, float norm_factor, float strength_thresh, float sensitivity,
                                                    bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input2);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);
    ARM_COMPUTE_ERROR_ON(0.0f == norm_factor);

    _input1          = input1;
    _input2          = input2;
    _output          = output;
    _sensitivity     = sensitivity;
    _strength_thresh = strength_thresh;
    _norm_factor     = norm_factor;
    _border_size     = BorderSize(block_size / 2);

    if(input1->info()->data_type() == DataType::S16)
    {
        _func = &fp16::harris_score_S16_S16_FLOAT<block_size>;
    }
    else
    {
        _func = &fp16::harris_score_S32_S32_FLOAT<block_size>;
    }

    ARM_COMPUTE_ERROR_ON(nullptr == _func);

    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = block_size;

    // Configure kernel window
    Window                 win = calculate_max_window(*input1->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input1->info(), -_border_size.left, -_border_size.top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              AccessWindowRectangle(input2->info(), -_border_size.left, -_border_size.top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->info()->valid_region(),
                                                       input2->info()->valid_region());

    output_access.set_valid_region(win, valid_region, border_undefined, border_size());

    INEKernel::configure(win);
}

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template class arm_compute::NEHarrisScoreKernel<3>;
template class arm_compute::NEHarrisScoreKernel<5>;
template class arm_compute::NEHarrisScoreKernel<7>;
template arm_compute::NEHarrisScoreKernel<3>::NEHarrisScoreKernel();
template arm_compute::NEHarrisScoreKernel<5>::NEHarrisScoreKernel();
template arm_compute::NEHarrisScoreKernel<7>::NEHarrisScoreKernel();

namespace
{
inline float32x4_t harris_score(float32x4_t gx2, float32x4_t gy2, float32x4_t gxgy, float32x4_t sensitivity, float32x4_t strength_thresh)
{
    // Trace^2
    float32x4_t trace2 = vaddq_f32(gx2, gy2);
    trace2             = vmulq_f32(trace2, trace2);

    // Det(A)
    float32x4_t det = vmulq_f32(gx2, gy2);
    det             = vmlsq_f32(det, gxgy, gxgy);

    // Det(A) - sensitivity * trace^2
    const float32x4_t mc = vmlsq_f32(det, sensitivity, trace2);

    // mc > strength_thresh
    const uint32x4_t mask = vcgtq_f32(mc, strength_thresh);

    return vbslq_f32(mask, mc, vdupq_n_f32(0.0f));
}

inline void harris_score1x3_FLOAT_FLOAT_FLOAT(float32x4_t low_gx, float32x4_t low_gy, float32x4_t high_gx, float32x4_t high_gy, float32x4_t &gx2, float32x4_t &gy2, float32x4_t &gxgy,
                                              float32x4_t norm_factor)
{
    // Normalize
    low_gx  = vmulq_f32(low_gx, norm_factor);
    low_gy  = vmulq_f32(low_gy, norm_factor);
    high_gx = vmulq_f32(high_gx, norm_factor);
    high_gy = vmulq_f32(high_gy, norm_factor);

    const float32x4_t l_gx = low_gx;
    const float32x4_t l_gy = low_gy;
    const float32x4_t m_gx = vextq_f32(low_gx, high_gx, 1);
    const float32x4_t m_gy = vextq_f32(low_gy, high_gy, 1);
    const float32x4_t r_gx = vextq_f32(low_gx, high_gx, 2);
    const float32x4_t r_gy = vextq_f32(low_gy, high_gy, 2);

    // Gx*Gx
    gx2 = vmlaq_f32(gx2, l_gx, l_gx);
    gx2 = vmlaq_f32(gx2, m_gx, m_gx);
    gx2 = vmlaq_f32(gx2, r_gx, r_gx);

    // Gy*Gy
    gy2 = vmlaq_f32(gy2, l_gy, l_gy);
    gy2 = vmlaq_f32(gy2, m_gy, m_gy);
    gy2 = vmlaq_f32(gy2, r_gy, r_gy);

    // Gx*Gy
    gxgy = vmlaq_f32(gxgy, l_gx, l_gy);
    gxgy = vmlaq_f32(gxgy, m_gx, m_gy);
    gxgy = vmlaq_f32(gxgy, r_gx, r_gy);
}

inline void harris_score1x5_FLOAT_FLOAT_FLOAT(float32x4_t low_gx, float32x4_t low_gy, float32x4_t high_gx, float32x4_t high_gy, float32x4_t &gx2, float32x4_t &gy2, float32x4_t &gxgy,
                                              float32x4_t norm_factor)
{
    // Normalize
    low_gx  = vmulq_f32(low_gx, norm_factor);
    low_gy  = vmulq_f32(low_gy, norm_factor);
    high_gx = vmulq_f32(high_gx, norm_factor);
    high_gy = vmulq_f32(high_gy, norm_factor);

    // L2 values
    float32x4_t gx = low_gx;
    float32x4_t gy = low_gy;

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // L1 values
    gx = vextq_f32(low_gx, high_gx, 1);
    gy = vextq_f32(low_gy, high_gy, 1);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // M values
    gx = vextq_f32(low_gx, high_gx, 2);
    gy = vextq_f32(low_gy, high_gy, 2);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // R1 values
    gx = vextq_f32(low_gx, high_gx, 3);
    gy = vextq_f32(low_gy, high_gy, 3);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // R2 values
    gx = high_gx;
    gy = high_gy;

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);
}

inline void harris_score1x7_FLOAT_FLOAT_FLOAT(float32x4_t low_gx, float32x4_t low_gy, float32x4_t high_gx, float32x4_t high_gy, float32x4_t high_gx1, float32x4_t high_gy1, float32x4_t &gx2,
                                              float32x4_t &gy2, float32x4_t &gxgy, float32x4_t norm_factor)
{
    // Normalize
    low_gx  = vmulq_f32(low_gx, norm_factor);
    low_gy  = vmulq_f32(low_gy, norm_factor);
    high_gx = vmulq_f32(high_gx, norm_factor);
    high_gy = vmulq_f32(high_gy, norm_factor);

    // L3 values
    float32x4_t gx = low_gx;
    float32x4_t gy = low_gy;

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // L2 values
    gx = vextq_f32(low_gx, high_gx, 1);
    gy = vextq_f32(low_gy, high_gy, 1);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // L1 values
    gx = vextq_f32(low_gx, high_gx, 2);
    gy = vextq_f32(low_gy, high_gy, 2);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // M values
    gx = vextq_f32(low_gx, high_gx, 3);
    gy = vextq_f32(low_gy, high_gy, 3);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // R1 values
    gx = high_gx;
    gy = high_gy;

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // Change tmp_low and tmp_high for calculating R2 and R3 values
    low_gx  = high_gx;
    low_gy  = high_gy;
    high_gx = high_gx1;
    high_gy = high_gy1;

    // Normalize
    high_gx = vmulq_f32(high_gx, norm_factor);
    high_gy = vmulq_f32(high_gy, norm_factor);

    // R2 values
    gx = vextq_f32(low_gx, high_gx, 1);
    gy = vextq_f32(low_gy, high_gy, 1);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);

    // R3 values
    gx = vextq_f32(low_gx, high_gx, 2);
    gy = vextq_f32(low_gy, high_gy, 2);

    // Accumulate
    gx2  = vmlaq_f32(gx2, gx, gx);
    gy2  = vmlaq_f32(gy2, gy, gy);
    gxgy = vmlaq_f32(gxgy, gx, gy);
}

inline void harris_score3x3_S16_S16_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)

{
    const auto     gx_ptr_0 = static_cast<const int16_t *__restrict>(input1_ptr) - 1;
    const auto     gy_ptr_0 = static_cast<const int16_t *__restrict>(input2_ptr) - 1;
    const int16_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int16_t *gy_ptr_1 = gy_ptr_0 + 4;
    const auto     output   = static_cast<float *__restrict>(output_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4x2_t gx2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gy2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gxgy =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };

    // Row0
    int16x8x2_t tmp_gx =
    {
        {
            vld1q_s16(gx_ptr_0 - input_stride),
            vld1q_s16(gx_ptr_1 - input_stride)
        }
    };
    int16x8x2_t tmp_gy =
    {
        {
            vld1q_s16(gy_ptr_0 - input_stride),
            vld1q_s16(gy_ptr_1 - input_stride)
        }
    };
    float32x4_t sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t strength_thresh = vdupq_n_f32(in_strength_thresh);

    float32x4_t low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[0])));
    float32x4_t low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[0])));
    float32x4_t high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[0])));
    float32x4_t high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[0])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[1])));
    low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[1])));
    high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[1])));
    high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[1])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Row1
    tmp_gx.val[0] = vld1q_s16(gx_ptr_0);
    tmp_gy.val[0] = vld1q_s16(gy_ptr_0);
    tmp_gx.val[1] = vld1q_s16(gx_ptr_1);
    tmp_gy.val[1] = vld1q_s16(gy_ptr_1);

    low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[0])));
    low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[0])));
    high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[0])));
    high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[0])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[1])));
    low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[1])));
    high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[1])));
    high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[1])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Row2
    tmp_gx.val[0] = vld1q_s16(gx_ptr_0 + input_stride);
    tmp_gy.val[0] = vld1q_s16(gy_ptr_0 + input_stride);
    tmp_gx.val[1] = vld1q_s16(gx_ptr_1 + input_stride);
    tmp_gy.val[1] = vld1q_s16(gy_ptr_1 + input_stride);

    low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[0])));
    low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[0])));
    high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[0])));
    high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[0])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[1])));
    low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[1])));
    high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[1])));
    high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[1])));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Calculate harris score
    const float32x4x2_t mc =
    {
        {
            harris_score(gx2.val[0], gy2.val[0], gxgy.val[0], sensitivity, strength_thresh),
            harris_score(gx2.val[1], gy2.val[1], gxgy.val[1], sensitivity, strength_thresh)
        }
    };

    // Store score
    vst1q_f32(output + 0, mc.val[0]);
    vst1q_f32(output + 4, mc.val[1]);
}

inline void harris_score3x3_S32_S32_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)
{
    auto           gx_ptr_0        = static_cast<const int32_t *__restrict>(input1_ptr) - 1;
    auto           gy_ptr_0        = static_cast<const int32_t *__restrict>(input2_ptr) - 1;
    const int32_t *gx_ptr_1        = gx_ptr_0 + 4;
    const int32_t *gy_ptr_1        = gy_ptr_0 + 4;
    const int32_t *gx_ptr_2        = gx_ptr_0 + 8;
    const int32_t *gy_ptr_2        = gy_ptr_0 + 8;
    const auto     output          = static_cast<float *__restrict>(output_ptr);
    float32x4_t    sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t    norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t    strength_thresh = vdupq_n_f32(in_strength_thresh);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4x2_t gx2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gy2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gxgy =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };

    // Row0
    float32x4_t low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_0 - input_stride));
    float32x4_t low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_0 - input_stride));
    float32x4_t high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_1 - input_stride));
    float32x4_t high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_1 - input_stride));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_1 - input_stride));
    low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_1 - input_stride));
    high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_2 - input_stride));
    high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_2 - input_stride));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Row1
    low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_0));
    low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_0));
    high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_1));
    high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_1));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_1));
    low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_1));
    high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_2));
    high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_2));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Row2
    low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_0 + input_stride));
    low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_0 + input_stride));
    high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_1 + input_stride));
    high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_1 + input_stride));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

    low_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_1 + input_stride));
    low_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_1 + input_stride));
    high_gx = vcvtq_f32_s32(vld1q_s32(gx_ptr_2 + input_stride));
    high_gy = vcvtq_f32_s32(vld1q_s32(gy_ptr_2 + input_stride));
    harris_score1x3_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

    // Calculate harris score
    const float32x4x2_t mc =
    {
        {
            harris_score(gx2.val[0], gy2.val[0], gxgy.val[0], sensitivity, strength_thresh),
            harris_score(gx2.val[1], gy2.val[1], gxgy.val[1], sensitivity, strength_thresh)
        }
    };

    // Store score
    vst1q_f32(output + 0, mc.val[0]);
    vst1q_f32(output + 4, mc.val[1]);
}

inline void harris_score5x5_S16_S16_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)
{
    auto           gx_ptr_0 = static_cast<const int16_t *__restrict>(input1_ptr) - 2 - 2 * input_stride;
    auto           gy_ptr_0 = static_cast<const int16_t *__restrict>(input2_ptr) - 2 - 2 * input_stride;
    const int16_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int16_t *gy_ptr_1 = gy_ptr_0 + 4;
    const auto     output   = static_cast<float *__restrict>(output_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4x2_t gx2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gy2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gxgy =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4_t sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t strength_thresh = vdupq_n_f32(in_strength_thresh);

    for(int i = 0; i < 5; ++i)
    {
        const int16x8x2_t tmp_gx =
        {
            {
                vld1q_s16(gx_ptr_0),
                vld1q_s16(gx_ptr_1)
            }
        };
        const int16x8x2_t tmp_gy =
        {
            {
                vld1q_s16(gy_ptr_0),
                vld1q_s16(gy_ptr_1)
            }
        };

        float32x4_t low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[0])));
        float32x4_t low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[0])));
        float32x4_t high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[0])));
        float32x4_t high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[0])));
        harris_score1x5_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

        low_gx  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gx.val[1])));
        low_gy  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp_gy.val[1])));
        high_gx = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gx.val[1])));
        high_gy = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp_gy.val[1])));
        harris_score1x5_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += input_stride;
        gy_ptr_0 += input_stride;
        gx_ptr_1 += input_stride;
        gy_ptr_1 += input_stride;
    }

    // Calculate harris score
    const float32x4x2_t mc =
    {
        {
            harris_score(gx2.val[0], gy2.val[0], gxgy.val[0], sensitivity, strength_thresh),
            harris_score(gx2.val[1], gy2.val[1], gxgy.val[1], sensitivity, strength_thresh)
        }
    };

    // Store score
    vst1q_f32(output + 0, mc.val[0]);
    vst1q_f32(output + 4, mc.val[1]);
}

inline void harris_score5x5_S32_S32_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)

{
    auto           gx_ptr_0 = static_cast<const int32_t *__restrict>(input1_ptr) - 2 - 2 * input_stride;
    auto           gy_ptr_0 = static_cast<const int32_t *__restrict>(input2_ptr) - 2 - 2 * input_stride;
    const int32_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int32_t *gy_ptr_1 = gy_ptr_0 + 4;
    const int32_t *gx_ptr_2 = gx_ptr_0 + 8;
    const int32_t *gy_ptr_2 = gy_ptr_0 + 8;
    const auto     output   = static_cast<float *__restrict>(output_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4x2_t gx2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gy2 =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4x2_t gxgy =
    {
        {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        }
    };
    float32x4_t sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t strength_thresh = vdupq_n_f32(in_strength_thresh);

    for(int i = 0; i < 5; ++i)
    {
        const float32x4_t low_gx_0  = vcvtq_f32_s32(vld1q_s32(gx_ptr_0));
        const float32x4_t low_gy_0  = vcvtq_f32_s32(vld1q_s32(gy_ptr_0));
        const float32x4_t high_gx_0 = vcvtq_f32_s32(vld1q_s32(gx_ptr_1));
        const float32x4_t high_gy_0 = vcvtq_f32_s32(vld1q_s32(gy_ptr_1));
        harris_score1x5_FLOAT_FLOAT_FLOAT(low_gx_0, low_gy_0, high_gx_0, high_gy_0, gx2.val[0], gy2.val[0], gxgy.val[0], norm_factor);

        const float32x4_t low_gx_1  = vcvtq_f32_s32(vld1q_s32(gx_ptr_1));
        const float32x4_t low_gy_1  = vcvtq_f32_s32(vld1q_s32(gy_ptr_1));
        const float32x4_t high_gx_1 = vcvtq_f32_s32(vld1q_s32(gx_ptr_2));
        const float32x4_t high_gy_1 = vcvtq_f32_s32(vld1q_s32(gy_ptr_2));
        harris_score1x5_FLOAT_FLOAT_FLOAT(low_gx_1, low_gy_1, high_gx_1, high_gy_1, gx2.val[1], gy2.val[1], gxgy.val[1], norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += input_stride;
        gy_ptr_0 += input_stride;
        gx_ptr_1 += input_stride;
        gy_ptr_1 += input_stride;
        gx_ptr_2 += input_stride;
        gy_ptr_2 += input_stride;
    }

    // Calculate harris score
    const float32x4x2_t mc =
    {
        {
            harris_score(gx2.val[0], gy2.val[0], gxgy.val[0], sensitivity, strength_thresh),
            harris_score(gx2.val[1], gy2.val[1], gxgy.val[1], sensitivity, strength_thresh)
        }
    };

    // Store score
    vst1q_f32(output + 0, mc.val[0]);
    vst1q_f32(output + 4, mc.val[1]);
}

inline void harris_score7x7_S16_S16_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)
{
    auto           gx_ptr_0 = static_cast<const int16_t *__restrict>(input1_ptr) - 3 - 3 * input_stride;
    auto           gy_ptr_0 = static_cast<const int16_t *__restrict>(input2_ptr) - 3 - 3 * input_stride;
    const int16_t *gx_ptr_1 = gx_ptr_0 + 8;
    const int16_t *gy_ptr_1 = gy_ptr_0 + 8;
    const auto     output   = static_cast<float *__restrict>(output_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4_t gx2             = vdupq_n_f32(0.0f);
    float32x4_t gy2             = vdupq_n_f32(0.0f);
    float32x4_t gxgy            = vdupq_n_f32(0.0f);
    float32x4_t sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t strength_thresh = vdupq_n_f32(in_strength_thresh);

    for(int i = 0; i < 7; ++i)
    {
        const int16x8_t tmp0_gx = vld1q_s16(gx_ptr_0);
        const int16x8_t tmp0_gy = vld1q_s16(gy_ptr_0);
        const int16x4_t tmp1_gx = vld1_s16(gx_ptr_1);
        const int16x4_t tmp1_gy = vld1_s16(gy_ptr_1);

        float32x4_t low_gx   = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp0_gx)));
        float32x4_t low_gy   = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp0_gy)));
        float32x4_t high_gx  = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp0_gx)));
        float32x4_t high_gy  = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp0_gy)));
        float32x4_t high_gx1 = vcvtq_f32_s32(vmovl_s16(tmp1_gx));
        float32x4_t high_gy1 = vcvtq_f32_s32(vmovl_s16(tmp1_gy));
        harris_score1x7_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, high_gx1, high_gy1, gx2, gy2, gxgy, norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += input_stride;
        gy_ptr_0 += input_stride;
        gx_ptr_1 += input_stride;
        gy_ptr_1 += input_stride;
    }

    // Calculate harris score
    const float32x4_t mc = harris_score(gx2, gy2, gxgy, sensitivity, strength_thresh);

    // Store score
    vst1q_f32(output, mc);
}

inline void harris_score7x7_S32_S32_FLOAT(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                          float in_norm_factor, float in_sensitivity, float in_strength_thresh)
{
    auto           gx_ptr_0 = static_cast<const int32_t *__restrict>(input1_ptr) - 3 - 3 * input_stride;
    auto           gy_ptr_0 = static_cast<const int32_t *__restrict>(input2_ptr) - 3 - 3 * input_stride;
    const int32_t *gx_ptr_1 = gx_ptr_0 + 4;
    const int32_t *gy_ptr_1 = gy_ptr_0 + 4;
    const int32_t *gx_ptr_2 = gx_ptr_1 + 4;
    const int32_t *gy_ptr_2 = gy_ptr_1 + 4;
    const auto     output   = static_cast<float *__restrict>(output_ptr);

    // Gx^2, Gy^2 and Gx*Gy
    float32x4_t gx2             = vdupq_n_f32(0.0f);
    float32x4_t gy2             = vdupq_n_f32(0.0f);
    float32x4_t gxgy            = vdupq_n_f32(0.0f);
    float32x4_t sensitivity     = vdupq_n_f32(in_sensitivity);
    float32x4_t norm_factor     = vdupq_n_f32(in_norm_factor);
    float32x4_t strength_thresh = vdupq_n_f32(in_strength_thresh);

    for(int i = 0; i < 7; ++i)
    {
        const float32x4_t low_gx   = vcvtq_f32_s32(vld1q_s32(gx_ptr_0));
        const float32x4_t low_gy   = vcvtq_f32_s32(vld1q_s32(gy_ptr_0));
        const float32x4_t high_gx  = vcvtq_f32_s32(vld1q_s32(gx_ptr_1));
        const float32x4_t high_gy  = vcvtq_f32_s32(vld1q_s32(gy_ptr_1));
        const float32x4_t high_gx1 = vcvtq_f32_s32(vld1q_s32(gx_ptr_2));
        const float32x4_t high_gy1 = vcvtq_f32_s32(vld1q_s32(gy_ptr_2));
        harris_score1x7_FLOAT_FLOAT_FLOAT(low_gx, low_gy, high_gx, high_gy, high_gx1, high_gy1, gx2, gy2, gxgy, norm_factor);

        // Update gx and gy pointer
        gx_ptr_0 += input_stride;
        gy_ptr_0 += input_stride;
        gx_ptr_1 += input_stride;
        gy_ptr_1 += input_stride;
        gx_ptr_2 += input_stride;
        gy_ptr_2 += input_stride;
    }

    // Calculate harris score
    const float32x4_t mc = harris_score(gx2, gy2, gxgy, sensitivity, strength_thresh);

    // Store score
    vst1q_f32(output, mc);
}

} // namespace

INEHarrisScoreKernel::INEHarrisScoreKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr), _sensitivity(0.0f), _strength_thresh(0.0f), _norm_factor(0.0f), _border_size()
{
}

template <int32_t block_size>
NEHarrisScoreKernel<block_size>::NEHarrisScoreKernel()
    : INEHarrisScoreKernel(), _func(nullptr)
{
}

template <int32_t block_size>
void NEHarrisScoreKernel<block_size>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    Iterator input1(_input1, window);
    Iterator input2(_input2, window);
    Iterator output(_output, window);

    const size_t input_stride = _input1->info()->strides_in_bytes()[1] / element_size_from_data_type(_input1->info()->data_type());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        (*_func)(input1.ptr(), input2.ptr(), output.ptr(), input_stride, _norm_factor, _sensitivity, _strength_thresh);
    },
    input1, input2, output);
}

template <int32_t block_size>
BorderSize        NEHarrisScoreKernel<block_size>::border_size() const
{
    return _border_size;
}

template <int32_t block_size>
void NEHarrisScoreKernel<block_size>::configure(const IImage *input1, const IImage *input2, IImage *output, float norm_factor, float strength_thresh, float sensitivity,
                                                bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input2);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);
    ARM_COMPUTE_ERROR_ON(0.0f == norm_factor);

    _input1          = input1;
    _input2          = input2;
    _output          = output;
    _sensitivity     = sensitivity;
    _strength_thresh = strength_thresh;
    _norm_factor     = norm_factor;
    _border_size     = BorderSize(block_size / 2);

    if(input1->info()->data_type() == DataType::S16)
    {
        switch(block_size)
        {
            case 3:
                _func = &harris_score3x3_S16_S16_FLOAT;
                break;
            case 5:
                _func = &harris_score5x5_S16_S16_FLOAT;
                break;
            case 7:
                _func = &harris_score7x7_S16_S16_FLOAT;
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid block size");
                break;
        }
    }
    else
    {
        switch(block_size)
        {
            case 3:
                _func = &harris_score3x3_S32_S32_FLOAT;
                break;
            case 5:
                _func = &harris_score5x5_S32_S32_FLOAT;
                break;
            case 7:
                _func = &harris_score7x7_S32_S32_FLOAT;
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid block size");
                break;
        }
    }

    ARM_COMPUTE_ERROR_ON(nullptr == _func);

    constexpr unsigned int num_elems_processed_per_iteration = block_size != 7 ? 8 : 4;
    constexpr unsigned int num_elems_read_per_iteration      = block_size != 7 ? 16 : 12;
    constexpr unsigned int num_elems_written_per_iteration   = block_size != 7 ? 8 : 4;
    constexpr unsigned int num_rows_read_per_iteration       = block_size;

    // Configure kernel window
    Window                 win = calculate_max_window(*input1->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input1->info(), -_border_size.left, -_border_size.top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              AccessWindowRectangle(input2->info(), -_border_size.left, -_border_size.top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->info()->valid_region(),
                                                       input2->info()->valid_region());

    output_access.set_valid_region(win, valid_region, border_undefined, border_size());

    INEKernel::configure(win);
}
