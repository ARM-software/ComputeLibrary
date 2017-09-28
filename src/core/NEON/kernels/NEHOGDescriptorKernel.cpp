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
#include "arm_compute/core/NEON/kernels/NEHOGDescriptorKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstring>

using namespace arm_compute;

namespace
{
void cell_width_lt8(const int16_t *__restrict mag_row_ptr, const uint8_t *__restrict phase_row_ptr, float *__restrict output_ptr,
                    size_t mag_stride, size_t phase_stride, size_t cell_width, size_t cell_height, size_t num_bins, float phase_scale)
{
    const float32x4_t        scale_f32    = vdupq_n_f32(phase_scale);
    static const float32x4_t one_f32      = vdupq_n_f32(1.0f);
    static const float32x4_t zerofive_f32 = vdupq_n_f32(0.5f);
    static const int32x4_t   zero_s32     = vdupq_n_s32(0);
    static const int32x4_t   one_s32      = vdupq_n_s32(1);
    const int32x4_t          num_bins_s32 = vdupq_n_s32(num_bins);

    memset(output_ptr, 0, sizeof(float) * num_bins);

    for(size_t yc = 0; yc < cell_height; ++yc)
    {
        int32_t xc = 0;

        for(; xc <= static_cast<int32_t>(cell_width) - 4; xc += 4)
        {
            // Load magnitude and phase values
            const uint8x8_t phase_u8 = vld1_u8(phase_row_ptr + xc + yc * phase_stride);
            const int16x4_t mag_s16  = vld1_s16(mag_row_ptr + xc + yc * mag_stride);

            // Convert magnitude and phase to float
            const float32x4_t mag_f32   = vcvtq_f32_s32(vmovl_s16(mag_s16));
            float32x4_t       phase_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(phase_u8))));

            // Scale phase: phase * scale + 0.5f
            phase_f32 = vmlaq_f32(zerofive_f32, phase_f32, scale_f32);

            // Compute histogram index.
            int32x4_t hidx_s32 = vcvtq_s32_f32(phase_f32);

            // Compute magnitude weights (w0 and w1)
            const float32x4_t hidx_f32 = vcvtq_f32_s32(hidx_s32);

            // w1 = phase_f32 - hidx_f32
            const float32x4_t w1_f32 = vsubq_f32(phase_f32, hidx_f32);

            // w0 = 1.0 - w1
            const float32x4_t w0_f32 = vsubq_f32(one_f32, w1_f32);

            // Compute contribute for splitting vote
            const float32x4_t mag_w0_f32 = vmulq_f32(mag_f32, w0_f32);
            const float32x4_t mag_w1_f32 = vmulq_f32(mag_f32, w1_f32);

            // Weighted vote between 2 bins

            // Check if the histogram index is equal to num_bins. If so, replace the index with 0
            uint32x4_t mask = vceqq_s32(hidx_s32, num_bins_s32);
            hidx_s32        = vbslq_s32(mask, zero_s32, hidx_s32);

            // Bin 0
            *(output_ptr + vgetq_lane_s32(hidx_s32, 0)) += vgetq_lane_f32(mag_w0_f32, 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 1)) += vgetq_lane_f32(mag_w0_f32, 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 2)) += vgetq_lane_f32(mag_w0_f32, 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 3)) += vgetq_lane_f32(mag_w0_f32, 3);

            hidx_s32 = vaddq_s32(hidx_s32, one_s32);

            // Check if the histogram index is equal to num_bins
            mask     = vceqq_s32(hidx_s32, num_bins_s32);
            hidx_s32 = vbslq_s32(mask, zero_s32, hidx_s32);

            // Bin1
            *(output_ptr + vgetq_lane_s32(hidx_s32, 0)) += vgetq_lane_f32(mag_w1_f32, 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 1)) += vgetq_lane_f32(mag_w1_f32, 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 2)) += vgetq_lane_f32(mag_w1_f32, 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32, 3)) += vgetq_lane_f32(mag_w1_f32, 3);
        }

        for(; xc < static_cast<int32_t>(cell_width); ++xc)
        {
            const float phase_value = *(phase_row_ptr + xc + yc * phase_stride) * phase_scale + 0.5f;
            const float mag_value   = *(mag_row_ptr + xc + yc * mag_stride);

            const float w1 = phase_value - std::floor(phase_value);

            // The quantised phase is the histogram index [0, num_bins - 1] - Round
            // Check limit of histogram index. If hidx == num_bins, hidx = 0
            const auto hidx = static_cast<size_t>(phase_value) % num_bins;

            // Weighted vote between 2 bins
            *(output_ptr + hidx) += mag_value * (1.0f - w1);
            *(output_ptr + ((hidx + 1) % (num_bins))) += mag_value * w1;
        }
    }
}

void cell_width_ge8(const int16_t *__restrict mag_row_ptr, const uint8_t *__restrict phase_row_ptr, float *__restrict output_ptr, size_t mag_stride, size_t phase_stride, size_t cell_width,
                    size_t cell_height, size_t num_bins, float phase_scale)
{
    const float32x4_t        scale_f32    = vdupq_n_f32(phase_scale);
    static const float32x4_t one_f32      = vdupq_n_f32(1.0f);
    static const float32x4_t zerofive_f32 = vdupq_n_f32(0.5f);
    static const int32x4_t   zero_s32     = vdupq_n_s32(0);
    static const int32x4_t   one_s32      = vdupq_n_s32(1);
    const int32x4_t          num_bins_s32 = vdupq_n_s32(num_bins);

    memset(output_ptr, 0, sizeof(float) * num_bins);

    for(size_t yc = 0; yc < cell_height; ++yc)
    {
        int32_t xc = 0;

        for(; xc <= static_cast<int32_t>(cell_width) - 8; xc += 8)
        {
            // Load magnitude and phase values
            const uint8x8_t phase_u8 = vld1_u8(phase_row_ptr + xc + yc * phase_stride);
            const int16x8_t mag_s16  = vld1q_s16(mag_row_ptr + xc + yc * mag_stride);

            // Convert phase to U16
            const uint16x8_t phase_u16 = vmovl_u8(phase_u8);

            // Convert magnitude to float32
            const float32x4x2_t mag_f32 =
            {
                {
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(mag_s16))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(mag_s16)))
                }
            };

            // Convert phase to float32
            float32x4x2_t phase_f32 =
            {
                {
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(phase_u16))),
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(phase_u16)))
                }
            };

            // Scale phase: phase * scale + 0.5f
            phase_f32.val[0] = vmlaq_f32(zerofive_f32, phase_f32.val[0], scale_f32);
            phase_f32.val[1] = vmlaq_f32(zerofive_f32, phase_f32.val[1], scale_f32);

            // Compute histogram index.
            int32x4x2_t hidx_s32 =
            {
                {
                    vcvtq_s32_f32(phase_f32.val[0]),
                    vcvtq_s32_f32(phase_f32.val[1])
                }
            };

            // Compute magnitude weights (w0 and w1)
            const float32x4x2_t hidx_f32 =
            {
                {
                    vcvtq_f32_s32(hidx_s32.val[0]),
                    vcvtq_f32_s32(hidx_s32.val[1])
                }
            };

            float32x4x2_t w1_f32 =
            {
                {
                    vsubq_f32(phase_f32.val[0], hidx_f32.val[0]),
                    vsubq_f32(phase_f32.val[1], hidx_f32.val[1])
                }
            };

            float32x4x2_t w0_f32 =
            {
                {
                    vsubq_f32(one_f32, w1_f32.val[0]),
                    vsubq_f32(one_f32, w1_f32.val[1])
                }
            };

            // Compute contribute for splitting vote
            const float32x4x2_t mag_w0_f32 =
            {
                {
                    vmulq_f32(mag_f32.val[0], w0_f32.val[0]),
                    vmulq_f32(mag_f32.val[1], w0_f32.val[1])
                }
            };

            const float32x4x2_t mag_w1_f32 =
            {
                {
                    vmulq_f32(mag_f32.val[0], w1_f32.val[0]),
                    vmulq_f32(mag_f32.val[1], w1_f32.val[1])
                }
            };

            // Weighted vote between 2 bins

            // Check if the histogram index is equal to num_bins
            uint32x4x2_t mask =
            {
                {
                    vceqq_s32(hidx_s32.val[0], num_bins_s32),
                    vceqq_s32(hidx_s32.val[1], num_bins_s32)
                }
            };

            hidx_s32.val[0] = vbslq_s32(mask.val[0], zero_s32, hidx_s32.val[0]);
            hidx_s32.val[1] = vbslq_s32(mask.val[1], zero_s32, hidx_s32.val[1]);

            // First bin - Low
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 0)) += vgetq_lane_f32(mag_w0_f32.val[0], 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 1)) += vgetq_lane_f32(mag_w0_f32.val[0], 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 2)) += vgetq_lane_f32(mag_w0_f32.val[0], 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 3)) += vgetq_lane_f32(mag_w0_f32.val[0], 3);

            // First bin - high
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 0)) += vgetq_lane_f32(mag_w0_f32.val[1], 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 1)) += vgetq_lane_f32(mag_w0_f32.val[1], 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 2)) += vgetq_lane_f32(mag_w0_f32.val[1], 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 3)) += vgetq_lane_f32(mag_w0_f32.val[1], 3);

            hidx_s32.val[0] = vaddq_s32(hidx_s32.val[0], one_s32);
            hidx_s32.val[1] = vaddq_s32(hidx_s32.val[1], one_s32);

            // Check if the histogram index is equal to num_bins
            mask.val[0] = vceqq_s32(hidx_s32.val[0], num_bins_s32);
            mask.val[1] = vceqq_s32(hidx_s32.val[1], num_bins_s32);

            hidx_s32.val[0] = vbslq_s32(mask.val[0], zero_s32, hidx_s32.val[0]);
            hidx_s32.val[1] = vbslq_s32(mask.val[1], zero_s32, hidx_s32.val[1]);

            // Second bin - Low
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 0)) += vgetq_lane_f32(mag_w1_f32.val[0], 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 1)) += vgetq_lane_f32(mag_w1_f32.val[0], 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 2)) += vgetq_lane_f32(mag_w1_f32.val[0], 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[0], 3)) += vgetq_lane_f32(mag_w1_f32.val[0], 3);

            // Second bin - high
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 0)) += vgetq_lane_f32(mag_w1_f32.val[1], 0);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 1)) += vgetq_lane_f32(mag_w1_f32.val[1], 1);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 2)) += vgetq_lane_f32(mag_w1_f32.val[1], 2);
            *(output_ptr + vgetq_lane_s32(hidx_s32.val[1], 3)) += vgetq_lane_f32(mag_w1_f32.val[1], 3);
        }

        for(; xc < static_cast<int32_t>(cell_width); xc++)
        {
            const float phase_value = *(phase_row_ptr + xc + yc * phase_stride) * phase_scale + 0.5f;
            const float mag_value   = *(mag_row_ptr + xc + yc * mag_stride);

            const float w1 = phase_value - std::floor(phase_value);

            // The quantised phase is the histogram index [0, num_bins - 1] - Round
            // Check limit of histogram index. If hidx == num_bins, hidx = 0
            const size_t hidx = static_cast<size_t>(phase_value) % num_bins;

            // Weighted vote between 2 bins
            *(output_ptr + hidx) += mag_value * (1.0f - w1);
            *(output_ptr + ((hidx + 1) % (num_bins))) += mag_value * w1;
        }
    }
}

void l2_norm(const float *__restrict input_row_ptr, float *__restrict output_ptr, size_t input_stride,
             size_t num_cells_per_block_height, size_t num_bins_block_x, size_t num_bins_block, float l2_hyst_threshold)
{
    ARM_COMPUTE_UNUSED(l2_hyst_threshold);

    float       sum     = 0.0f;
    float32x4_t sum_f32 = vdupq_n_f32(0.0f);

    // Compute L2-Norm
    for(size_t yc = 0; yc < num_cells_per_block_height; ++yc)
    {
        const float *const hist_ptr = input_row_ptr + yc * input_stride;

        int32_t xc = 0;

        for(; xc <= static_cast<int32_t>(num_bins_block_x) - 16; xc += 16)
        {
            const float32x4x4_t input_value =
            {
                {
                    vld1q_f32(hist_ptr + xc + 0),
                    vld1q_f32(hist_ptr + xc + 4),
                    vld1q_f32(hist_ptr + xc + 8),
                    vld1q_f32(hist_ptr + xc + 12)
                }
            };

            // Compute input_value^2
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[0], input_value.val[0]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[1], input_value.val[1]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[2], input_value.val[2]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[3], input_value.val[3]);

            vst1q_f32(&output_ptr[xc + 0 + yc * num_bins_block_x], input_value.val[0]);
            vst1q_f32(&output_ptr[xc + 4 + yc * num_bins_block_x], input_value.val[1]);
            vst1q_f32(&output_ptr[xc + 8 + yc * num_bins_block_x], input_value.val[2]);
            vst1q_f32(&output_ptr[xc + 12 + yc * num_bins_block_x], input_value.val[3]);
        }

        // Compute left over
        for(; xc < static_cast<int32_t>(num_bins_block_x); xc++)
        {
            const float input_value = hist_ptr[xc];

            sum += input_value * input_value;

            output_ptr[xc + yc * num_bins_block_x] = input_value;
        }
    }

    sum += vgetq_lane_f32(sum_f32, 0);
    sum += vgetq_lane_f32(sum_f32, 1);
    sum += vgetq_lane_f32(sum_f32, 2);
    sum += vgetq_lane_f32(sum_f32, 3);

    const float       scale     = 1.0f / (std::sqrt(sum) + num_bins_block * 0.1f);
    const float32x4_t scale_f32 = vdupq_n_f32(scale);

    int32_t i = 0;

    for(; i <= static_cast<int32_t>(num_bins_block) - 16; i += 16)
    {
        float32x4x4_t input_value =
        {
            {
                vld1q_f32(&output_ptr[i + 0]),
                vld1q_f32(&output_ptr[i + 4]),
                vld1q_f32(&output_ptr[i + 8]),
                vld1q_f32(&output_ptr[i + 12])
            }
        };

        // Scale input_value
        input_value.val[0] = vmulq_f32(input_value.val[0], scale_f32);
        input_value.val[1] = vmulq_f32(input_value.val[1], scale_f32);
        input_value.val[2] = vmulq_f32(input_value.val[2], scale_f32);
        input_value.val[3] = vmulq_f32(input_value.val[3], scale_f32);

        vst1q_f32(&output_ptr[i + 0], input_value.val[0]);
        vst1q_f32(&output_ptr[i + 4], input_value.val[1]);
        vst1q_f32(&output_ptr[i + 8], input_value.val[2]);
        vst1q_f32(&output_ptr[i + 12], input_value.val[3]);
    }

    for(; i < static_cast<int32_t>(num_bins_block); ++i)
    {
        output_ptr[i] *= scale;
    }
}

void l2hys_norm(const float *__restrict input_row_ptr, float *__restrict output_ptr, size_t input_stride, size_t num_cells_per_block_height, size_t num_bins_block_x, size_t num_bins_block,
                float l2_hyst_threshold)
{
    float       sum     = 0.0f;
    float32x4_t sum_f32 = vdupq_n_f32(0.0f);

    // Compute L2-Hys
    for(size_t yc = 0; yc < num_cells_per_block_height; ++yc)
    {
        const float *const hist_ptr = input_row_ptr + yc * input_stride;

        int32_t xc = 0;

        for(; xc <= static_cast<int32_t>(num_bins_block_x) - 16; xc += 16)
        {
            const float32x4x4_t input_value =
            {
                {
                    vld1q_f32(hist_ptr + xc + 0),
                    vld1q_f32(hist_ptr + xc + 4),
                    vld1q_f32(hist_ptr + xc + 8),
                    vld1q_f32(hist_ptr + xc + 12)
                }
            };

            // Compute input_value^2
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[0], input_value.val[0]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[1], input_value.val[1]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[2], input_value.val[2]);
            sum_f32 = vmlaq_f32(sum_f32, input_value.val[3], input_value.val[3]);

            vst1q_f32(&output_ptr[xc + 0 + yc * num_bins_block_x], input_value.val[0]);
            vst1q_f32(&output_ptr[xc + 4 + yc * num_bins_block_x], input_value.val[1]);
            vst1q_f32(&output_ptr[xc + 8 + yc * num_bins_block_x], input_value.val[2]);
            vst1q_f32(&output_ptr[xc + 12 + yc * num_bins_block_x], input_value.val[3]);
        }

        // Compute left over
        for(; xc < static_cast<int32_t>(num_bins_block_x); ++xc)
        {
            const float input_value = hist_ptr[xc];

            sum += input_value * input_value;

            output_ptr[xc + yc * num_bins_block_x] = input_value;
        }
    }

    sum += vgetq_lane_f32(sum_f32, 0);
    sum += vgetq_lane_f32(sum_f32, 1);
    sum += vgetq_lane_f32(sum_f32, 2);
    sum += vgetq_lane_f32(sum_f32, 3);

    float             scale                 = 1.0f / (std::sqrt(sum) + num_bins_block * 0.1f);
    float32x4_t       scale_f32             = vdupq_n_f32(scale);
    const float32x4_t l2_hyst_threshold_f32 = vdupq_n_f32(l2_hyst_threshold);

    // Reset sum
    sum_f32 = vdupq_n_f32(0.0f);
    sum     = 0.0f;

    int32_t i = 0;

    for(; i <= static_cast<int32_t>(num_bins_block) - 16; i += 16)
    {
        float32x4x4_t input_value =
        {
            {
                vld1q_f32(&output_ptr[i + 0]),
                vld1q_f32(&output_ptr[i + 4]),
                vld1q_f32(&output_ptr[i + 8]),
                vld1q_f32(&output_ptr[i + 12])
            }
        };

        // Scale input_value
        input_value.val[0] = vmulq_f32(input_value.val[0], scale_f32);
        input_value.val[1] = vmulq_f32(input_value.val[1], scale_f32);
        input_value.val[2] = vmulq_f32(input_value.val[2], scale_f32);
        input_value.val[3] = vmulq_f32(input_value.val[3], scale_f32);

        // Clip input_value if over _threshold_l2hys
        input_value.val[0] = vminq_f32(input_value.val[0], l2_hyst_threshold_f32);
        input_value.val[1] = vminq_f32(input_value.val[1], l2_hyst_threshold_f32);
        input_value.val[2] = vminq_f32(input_value.val[2], l2_hyst_threshold_f32);
        input_value.val[3] = vminq_f32(input_value.val[3], l2_hyst_threshold_f32);

        // Compute input_value^2
        sum_f32 = vmlaq_f32(sum_f32, input_value.val[0], input_value.val[0]);
        sum_f32 = vmlaq_f32(sum_f32, input_value.val[1], input_value.val[1]);
        sum_f32 = vmlaq_f32(sum_f32, input_value.val[2], input_value.val[2]);
        sum_f32 = vmlaq_f32(sum_f32, input_value.val[3], input_value.val[3]);

        vst1q_f32(&output_ptr[i + 0], input_value.val[0]);
        vst1q_f32(&output_ptr[i + 4], input_value.val[1]);
        vst1q_f32(&output_ptr[i + 8], input_value.val[2]);
        vst1q_f32(&output_ptr[i + 12], input_value.val[3]);
    }

    sum += vgetq_lane_f32(sum_f32, 0);
    sum += vgetq_lane_f32(sum_f32, 1);
    sum += vgetq_lane_f32(sum_f32, 2);
    sum += vgetq_lane_f32(sum_f32, 3);

    for(; i < static_cast<int32_t>(num_bins_block); ++i)
    {
        float input_value = output_ptr[i] * scale;

        // Clip scaled input_value if over _threshold_L2hys
        input_value = std::min(input_value, l2_hyst_threshold);

        sum += input_value * input_value;

        output_ptr[i] = input_value;
    }

    // We use the same constants of OpenCV
    scale     = 1.0f / (std::sqrt(sum) + 1e-3f);
    scale_f32 = vdupq_n_f32(scale);

    // Rescale
    i = 0;

    for(; i <= static_cast<int32_t>(num_bins_block) - 16; i += 16)
    {
        float32x4x4_t input_value =
        {
            {
                vld1q_f32(&output_ptr[i + 0]),
                vld1q_f32(&output_ptr[i + 4]),
                vld1q_f32(&output_ptr[i + 8]),
                vld1q_f32(&output_ptr[i + 12])
            }
        };

        // Scale input_value
        input_value.val[0] = vmulq_f32(input_value.val[0], scale_f32);
        input_value.val[1] = vmulq_f32(input_value.val[1], scale_f32);
        input_value.val[2] = vmulq_f32(input_value.val[2], scale_f32);
        input_value.val[3] = vmulq_f32(input_value.val[3], scale_f32);

        vst1q_f32(&output_ptr[i + 0], input_value.val[0]);
        vst1q_f32(&output_ptr[i + 4], input_value.val[1]);
        vst1q_f32(&output_ptr[i + 8], input_value.val[2]);
        vst1q_f32(&output_ptr[i + 12], input_value.val[3]);
    }

    for(; i < static_cast<int32_t>(num_bins_block); ++i)
    {
        // Store result
        output_ptr[i] *= scale;
    }
}

void l1_norm(const float *__restrict input_row_ptr, float *__restrict output_ptr, size_t input_stride, size_t num_cells_per_block_height, size_t num_bins_block_x, size_t num_bins_block,
             float l2_hyst_threshold)
{
    ARM_COMPUTE_UNUSED(l2_hyst_threshold);

    float       sum     = 0.0f;
    float32x4_t sum_f32 = vdupq_n_f32(0.0f);

    // Compute L1-Norm
    for(size_t yc = 0; yc < num_cells_per_block_height; ++yc)
    {
        const float *const hist_ptr = input_row_ptr + yc * input_stride;

        int32_t xc = 0;

        for(; xc <= static_cast<int32_t>(num_bins_block_x) - 16; xc += 16)
        {
            const float32x4x4_t input_value =
            {
                {
                    vld1q_f32(hist_ptr + xc + 0),
                    vld1q_f32(hist_ptr + xc + 4),
                    vld1q_f32(hist_ptr + xc + 8),
                    vld1q_f32(hist_ptr + xc + 12)
                }
            };

            // Compute |input_value|
            sum_f32 += vabsq_f32(input_value.val[0]);
            sum_f32 += vabsq_f32(input_value.val[1]);
            sum_f32 += vabsq_f32(input_value.val[2]);
            sum_f32 += vabsq_f32(input_value.val[3]);

            vst1q_f32(&output_ptr[xc + 0 + yc * num_bins_block_x], input_value.val[0]);
            vst1q_f32(&output_ptr[xc + 4 + yc * num_bins_block_x], input_value.val[1]);
            vst1q_f32(&output_ptr[xc + 8 + yc * num_bins_block_x], input_value.val[2]);
            vst1q_f32(&output_ptr[xc + 12 + yc * num_bins_block_x], input_value.val[3]);
        }

        for(; xc < static_cast<int32_t>(num_bins_block_x); xc++)
        {
            const float input_value = hist_ptr[xc];

            sum += std::abs(input_value);

            output_ptr[xc + yc * num_bins_block_x] = input_value;
        }
    }

    sum += vgetq_lane_f32(sum_f32, 0);
    sum += vgetq_lane_f32(sum_f32, 1);
    sum += vgetq_lane_f32(sum_f32, 2);
    sum += vgetq_lane_f32(sum_f32, 3);

    const float       scale     = 1.0f / (std::sqrt(sum) + num_bins_block * 0.1f);
    const float32x4_t scale_f32 = vdupq_n_f32(scale);

    int32_t i = 0;

    for(; i <= static_cast<int32_t>(num_bins_block) - 16; i += 16)
    {
        float32x4x4_t input_value =
        {
            {
                vld1q_f32(&output_ptr[i + 0]),
                vld1q_f32(&output_ptr[i + 4]),
                vld1q_f32(&output_ptr[i + 8]),
                vld1q_f32(&output_ptr[i + 12])
            }
        };

        // Scale input_value
        input_value.val[0] = vmulq_f32(input_value.val[0], scale_f32);
        input_value.val[1] = vmulq_f32(input_value.val[1], scale_f32);
        input_value.val[2] = vmulq_f32(input_value.val[2], scale_f32);
        input_value.val[3] = vmulq_f32(input_value.val[3], scale_f32);

        vst1q_f32(&output_ptr[i + 0], input_value.val[0]);
        vst1q_f32(&output_ptr[i + 4], input_value.val[1]);
        vst1q_f32(&output_ptr[i + 8], input_value.val[2]);
        vst1q_f32(&output_ptr[i + 12], input_value.val[3]);
    }

    for(; i < static_cast<int32_t>(num_bins_block); ++i)
    {
        output_ptr[i] *= scale;
    }
}
} // namespace

NEHOGOrientationBinningKernel::NEHOGOrientationBinningKernel()
    : _func(nullptr), _input_magnitude(nullptr), _input_phase(nullptr), _output(nullptr), _cell_width(0), _cell_height(0), _num_bins(0), _phase_scale(0)
{
}

void NEHOGOrientationBinningKernel::configure(const ITensor *input_magnitude, const ITensor *input_phase, ITensor *output, const HOGInfo *hog_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_magnitude, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(hog_info == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, hog_info->num_bins(), DataType::F32);
    ARM_COMPUTE_ERROR_ON(input_magnitude->info()->dimension(Window::DimX) != input_phase->info()->dimension(Window::DimX));
    ARM_COMPUTE_ERROR_ON(input_magnitude->info()->dimension(Window::DimY) != input_phase->info()->dimension(Window::DimY));

    _input_magnitude = input_magnitude;
    _input_phase     = input_phase;
    _output          = output;
    _cell_width      = hog_info->cell_size().width;
    _cell_height     = hog_info->cell_size().height;
    _num_bins        = hog_info->num_bins();
    _phase_scale     = (PhaseType::SIGNED == hog_info->phase_type() ? _num_bins / 360.0f : _num_bins / 180.0f);
    _phase_scale *= (PhaseType::SIGNED == hog_info->phase_type() ? 360.0f / 255.0f : 1.0f);

    if(_cell_width < 8)
    {
        _func = &cell_width_lt8;
    }
    else
    {
        _func = &cell_width_ge8;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    const unsigned int     num_elems_read_per_iteration      = 1;
    const unsigned int     num_rows_read_per_iteration       = _cell_height;
    const unsigned int     num_elems_written_per_iteration   = 1;

    // Configure kernel window
    Window                 win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input_magnitude->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              AccessWindowRectangle(input_phase->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEHOGOrientationBinningKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const size_t mag_stride   = _input_magnitude->info()->strides_in_bytes()[Window::DimY] / pixel_size_from_format(_input_magnitude->info()->format());
    const size_t phase_stride = _input_phase->info()->strides_in_bytes()[Window::DimY] / pixel_size_from_format(_input_phase->info()->format());

    Window win_mag(window);
    win_mag.set(Window::DimX, Window::Dimension(window.x().start() * _cell_width, window.x().start() * _cell_width, _cell_width));
    win_mag.set(Window::DimY, Window::Dimension(window.y().start() * _cell_height, window.y().start() * _cell_height, _cell_height));

    Window win_phase(win_mag);

    Iterator mag(_input_magnitude, win_mag);
    Iterator phase(_input_phase, win_phase);
    Iterator out(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto mag_row_ptr   = reinterpret_cast<const int16_t *>(mag.ptr());
        const auto phase_row_ptr = reinterpret_cast<const uint8_t *>(phase.ptr());
        const auto out_row_ptr   = reinterpret_cast<float *>(out.ptr());

        (*_func)(mag_row_ptr, phase_row_ptr, out_row_ptr, mag_stride, phase_stride, _cell_width, _cell_height, _num_bins, _phase_scale);
    },
    mag, phase, out);
}

NEHOGBlockNormalizationKernel::NEHOGBlockNormalizationKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _num_cells_per_block(), _num_cells_per_block_stride(), _num_bins(0), _l2_hyst_threshold(0.0f)
{
}

void NEHOGBlockNormalizationKernel::configure(const ITensor *input, ITensor *output, const HOGInfo *hog_info)
{
    ARM_COMPUTE_ERROR_ON(hog_info == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, hog_info->num_bins(), DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);

    // Number of cells per block
    const Size2D num_cells_per_block(hog_info->block_size().width / hog_info->cell_size().width,
                                     hog_info->block_size().height / hog_info->cell_size().height);

    // Number of cells per block stride
    const Size2D num_cells_per_block_stride(hog_info->block_stride().width / hog_info->cell_size().width,
                                            hog_info->block_stride().height / hog_info->cell_size().height);

    _input                      = input;
    _output                     = output;
    _l2_hyst_threshold          = hog_info->l2_hyst_threshold();
    _num_cells_per_block        = num_cells_per_block;
    _num_cells_per_block_stride = num_cells_per_block_stride;
    _num_bins                   = hog_info->num_bins();

    ARM_COMPUTE_ERROR_ON((output->info()->num_channels() != (_num_bins * num_cells_per_block.width * num_cells_per_block.height)));

    switch(hog_info->normalization_type())
    {
        case HOGNormType::L2_NORM:
            _func = &l2_norm;
            break;
        case HOGNormType::L2HYS_NORM:
            _func = &l2hys_norm;
            break;
        case HOGNormType::L1_NORM:
            _func = &l1_norm;
            break;
        default:
            ARM_COMPUTE_ERROR_ON("Normalisation type not supported");
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    const unsigned int     num_elems_read_per_iteration      = 1;
    const unsigned int     num_rows_read_per_iteration       = _num_cells_per_block.height;
    const unsigned int     num_elems_written_per_iteration   = 1;
    const unsigned int     num_rows_written_per_iteration    = _num_cells_per_block.height;

    // Configure kernel window
    Window                win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, num_rows_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEHOGBlockNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get number of bins per block
    const size_t num_bins_per_block = _output->info()->num_channels();

    // Number of bins on the same row of the block
    const int32_t num_bins_per_block_x = _num_cells_per_block.width * _num_bins;

    const size_t input_stride = _input->info()->strides_in_bytes()[Window::DimY] / data_size_from_type(_input->info()->data_type());

    Window win_in(window);
    win_in.set_dimension_step(Window::DimX, _num_cells_per_block_stride.width);
    win_in.set_dimension_step(Window::DimY, _num_cells_per_block_stride.height);

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    // Normalises blocks
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_row_ptr = reinterpret_cast<const float *>(in.ptr());
        const auto out_row_ptr   = reinterpret_cast<float *>(out.ptr());

        // Execute normalization function
        (*_func)(input_row_ptr, out_row_ptr, input_stride, _num_cells_per_block.height, num_bins_per_block_x, num_bins_per_block, _l2_hyst_threshold);
    },
    in, out);
}
