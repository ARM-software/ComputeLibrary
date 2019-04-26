/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NECannyEdgeKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
constexpr int NO_EDGE = 0;
constexpr int EDGE    = 255;
constexpr int MAYBE   = 127;
} // namespace

namespace
{
inline uint8x8_t phase_quantization(const float32x4x2_t &gx, const float32x4x2_t &gy)
{
    // Constant use for evaluating score1 and score3
    static const float32x4_t const45 = vdupq_n_f32(0.70710678118655f);
    static const float32x4_t zero    = vdupq_n_f32(0.0f);
    static const float32x4_t one     = vdupq_n_f32(1.0f);
    static const float32x4_t two     = vdupq_n_f32(2.0f);
    static const float32x4_t three   = vdupq_n_f32(3.0f);

    // Score0: (1, 0)
    const float32x4x2_t score0 =
    {
        {
            vabsq_f32(gx.val[0]),
            vabsq_f32(gx.val[1])
        }
    };

    // Score2: ( 0, 1 )
    const float32x4x2_t score2 =
    {
        {
            vabsq_f32(gy.val[0]),
            vabsq_f32(gy.val[1])
        }
    };

    // Score1 and Score3: ( sqrt(2) / 2, sqrt(2) / 2 ) - ( -sqrt(2) / 2, sqrt(2) / 2 )
    float32x4x2_t score1 =
    {
        {
            vmulq_f32(gy.val[0], const45),
            vmulq_f32(gy.val[1], const45)
        }
    };

    float32x4x2_t score3 = score1;

    score1.val[0] = vmlaq_f32(score1.val[0], gx.val[0], const45);
    score1.val[1] = vmlaq_f32(score1.val[1], gx.val[1], const45);
    score3.val[0] = vmlsq_f32(score3.val[0], gx.val[0], const45);
    score3.val[1] = vmlsq_f32(score3.val[1], gx.val[1], const45);

    score1.val[0] = vabsq_f32(score1.val[0]);
    score1.val[1] = vabsq_f32(score1.val[1]);
    score3.val[0] = vabsq_f32(score3.val[0]);
    score3.val[1] = vabsq_f32(score3.val[1]);

    float32x4x2_t phase =
    {
        {
            zero,
            zero
        }
    };

    float32x4x2_t old_score = score0;

    // score1 > old_score?
    uint32x4x2_t mask =
    {
        {
            vcgtq_f32(score1.val[0], old_score.val[0]),
            vcgtq_f32(score1.val[1], old_score.val[1])
        }
    };

    phase.val[0]     = vbslq_f32(mask.val[0], one, phase.val[0]);
    phase.val[1]     = vbslq_f32(mask.val[1], one, phase.val[1]);
    old_score.val[0] = vbslq_f32(mask.val[0], score1.val[0], old_score.val[0]);
    old_score.val[1] = vbslq_f32(mask.val[1], score1.val[1], old_score.val[1]);

    // score2 > old_score?
    mask.val[0] = vcgtq_f32(score2.val[0], old_score.val[0]);
    mask.val[1] = vcgtq_f32(score2.val[1], old_score.val[1]);

    phase.val[0]     = vbslq_f32(mask.val[0], two, phase.val[0]);
    phase.val[1]     = vbslq_f32(mask.val[1], two, phase.val[1]);
    old_score.val[0] = vbslq_f32(mask.val[0], score2.val[0], old_score.val[0]);
    old_score.val[1] = vbslq_f32(mask.val[1], score2.val[1], old_score.val[1]);

    // score3 > old_score?
    mask.val[0] = vcgtq_f32(score3.val[0], old_score.val[0]);
    mask.val[1] = vcgtq_f32(score3.val[1], old_score.val[1]);

    phase.val[0]     = vbslq_f32(mask.val[0], three, phase.val[0]);
    phase.val[1]     = vbslq_f32(mask.val[1], three, phase.val[1]);
    old_score.val[0] = vbslq_f32(mask.val[0], score3.val[0], old_score.val[0]);
    old_score.val[1] = vbslq_f32(mask.val[1], score3.val[1], old_score.val[1]);

    // Convert from float32x4_t to uint8x8_t
    return vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(phase.val[0])),
                                  vmovn_u32(vcvtq_u32_f32(phase.val[1]))));
}

/* Computes the gradient phase if gradient_size = 3 or 5. The output is quantized.
 * 0 = 0°, 1 = 45°, 2 = 90°, 3 = 135°
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return quantized phase for 8 pixels
 */
inline uint8x8_t phase_quantization_S16_S16(int16x8_t gx, int16x8_t gy)
{
    // Convert to float
    const float32x4x2_t gx_f32 =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gx))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gx)))
        }
    };

    const float32x4x2_t gy_f32 =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gy))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gy)))
        }
    };

    return phase_quantization(gx_f32, gy_f32);
}

/* Computes the gradient phase if gradient_size = 7. The output is quantized.
 * 0 = 0°, 1 = 45°, 2 = 90°, 3 = 135°
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return quantized phase for 8 pixels
 */
inline uint8x8_t phase_quantization_S32_S32(const int32x4x2_t &gx, const int32x4x2_t &gy)
{
    // Convert to float
    const float32x4x2_t gx_f32 =
    {
        {
            vcvtq_f32_s32(gx.val[0]),
            vcvtq_f32_s32(gx.val[1])
        }
    };

    const float32x4x2_t gy_f32 =
    {
        {
            vcvtq_f32_s32(gy.val[0]),
            vcvtq_f32_s32(gy.val[1])
        }
    };

    return phase_quantization(gx_f32, gy_f32);
}

/* Computes the magnitude using the L1-norm type if gradient_size = 3 or 5
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return magnitude for 8 pixels
 */
inline uint16x8_t mag_l1_S16_S16(int16x8_t gx, int16x8_t gy)
{
    return vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(gx)),
                     vreinterpretq_u16_s16(vabsq_s16(gy)));
}

/* Computes the magnitude using the L1-norm type if gradient_size = 7
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return magnitude for 8 pixels
 */
inline uint32x4x2_t mag_l1_S32_S32(const int32x4x2_t &gx, const int32x4x2_t &gy)
{
    const uint32x4x2_t gx_abs =
    {
        {
            vreinterpretq_u32_s32(vabsq_s32(gx.val[0])),
            vreinterpretq_u32_s32(vabsq_s32(gx.val[1]))
        }
    };

    const uint32x4x2_t gy_abs =
    {
        {
            vreinterpretq_u32_s32(vabsq_s32(gy.val[0])),
            vreinterpretq_u32_s32(vabsq_s32(gy.val[1]))
        }
    };

    const uint32x4x2_t output =
    {
        {
            vaddq_u32(gx_abs.val[0], gy_abs.val[0]),
            vaddq_u32(gx_abs.val[1], gy_abs.val[1])
        }
    };

    return output;
}

inline float32x4x2_t mag_l2(const float32x4x2_t &gx, const float32x4x2_t &gy)
{
    // x^2 ...
    float32x4x2_t magnitude =
    {
        {
            vmulq_f32(gx.val[0], gx.val[0]),
            vmulq_f32(gx.val[1], gx.val[1])
        }
    };

    // ... + y^2
    magnitude.val[0] = vmlaq_f32(magnitude.val[0], gy.val[0], gy.val[0]);
    magnitude.val[1] = vmlaq_f32(magnitude.val[1], gy.val[1], gy.val[1]);

    // sqrt(...)
    magnitude.val[0] = vmulq_f32(vrsqrteq_f32(magnitude.val[0]), magnitude.val[0]);
    magnitude.val[1] = vmulq_f32(vrsqrteq_f32(magnitude.val[1]), magnitude.val[1]);

    return magnitude;
}

/* Computes the magnitude using L2-norm if gradient_size = 3 or 5
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return magnitude for 8 pixels
 */
inline uint16x8_t mag_l2_S16_S16(int16x8_t gx, int16x8_t gy)
{
    // Compute magnitude using L2 normalization
    const float32x4x2_t gx2 =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gx))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gx)))
        }
    };

    const float32x4x2_t gy2 =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gy))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gy)))
        }
    };

    const float32x4x2_t magnitude = mag_l2(gx2, gy2);

    // Store magnitude - Convert to uint16x8
    return vcombine_u16(vmovn_u32(vcvtq_u32_f32(magnitude.val[0])),
                        vmovn_u32(vcvtq_u32_f32(magnitude.val[1])));
}

/* Computes the magnitude using L2-norm if gradient_size = 7
 *
 * @param[in] gx Gx component
 * @param[in] gy Gy component
 *
 * @return magnitude for 8 pixels
 */
inline uint32x4x2_t mag_l2_S32_S32(const int32x4x2_t &gx, const int32x4x2_t &gy)
{
    // Compute magnitude using L2 normalization
    float32x4x2_t gx2 =
    {
        {
            vcvtq_f32_s32(gx.val[0]),
            vcvtq_f32_s32(gx.val[1])
        }
    };

    float32x4x2_t gy2 =
    {
        {
            vcvtq_f32_s32(gy.val[0]),
            vcvtq_f32_s32(gy.val[1])
        }
    };

    const float32x4x2_t magnitude = mag_l2(gx2, gy2);
    const uint32x4x2_t  mag32 =
    {
        {
            vcvtq_u32_f32(magnitude.val[0]),
            vcvtq_u32_f32(magnitude.val[1])
        }
    };

    return mag32;
}

/* Gradient function used when the gradient size = 3 or 5 and when the norm_type = L1-norm
 *
 * @param[in]  gx_ptr        Pointer to source image. Gx image. Data type supported S16
 * @param[in]  gy_ptr        Pointer to source image. Gy image. Data type supported S16
 * @param[out] magnitude_ptr Pointer to destination image. Magnitude. Data type supported U16
 * @param[out] phase_ptr     Pointer to destination image. Quantized phase. Data type supported U8
 */
void mag_phase_l1norm_S16_S16_U16_U8(const void *__restrict gx_ptr, const void *__restrict gy_ptr, void *__restrict magnitude_ptr, void *__restrict phase_ptr)
{
    const auto gx        = static_cast<const int16_t *__restrict>(gx_ptr);
    const auto gy        = static_cast<const int16_t *__restrict>(gy_ptr);
    const auto magnitude = static_cast<uint16_t *__restrict>(magnitude_ptr);
    const auto phase     = static_cast<uint8_t *__restrict>(phase_ptr);

    const int16x8x4_t gx_val =
    {
        {
            vld1q_s16(gx),
            vld1q_s16(gx + 8),
            vld1q_s16(gx + 16),
            vld1q_s16(gx + 24)
        }
    };

    const int16x8x4_t gy_val =
    {
        {
            vld1q_s16(gy),
            vld1q_s16(gy + 8),
            vld1q_s16(gy + 16),
            vld1q_s16(gy + 24)
        }
    };

    // Compute and store phase
    vst1_u8(phase + 0, phase_quantization_S16_S16(gx_val.val[0], gy_val.val[0]));
    vst1_u8(phase + 8, phase_quantization_S16_S16(gx_val.val[1], gy_val.val[1]));
    vst1_u8(phase + 16, phase_quantization_S16_S16(gx_val.val[2], gy_val.val[2]));
    vst1_u8(phase + 24, phase_quantization_S16_S16(gx_val.val[3], gy_val.val[3]));

    // Compute ans store magnitude using L1 normalization
    vst1q_u16(magnitude + 0, mag_l1_S16_S16(gx_val.val[0], gy_val.val[0]));
    vst1q_u16(magnitude + 8, mag_l1_S16_S16(gx_val.val[1], gy_val.val[1]));
    vst1q_u16(magnitude + 16, mag_l1_S16_S16(gx_val.val[2], gy_val.val[2]));
    vst1q_u16(magnitude + 24, mag_l1_S16_S16(gx_val.val[3], gy_val.val[3]));
}

/* Gradient function used when the gradient size = 3 or 5 and when the norm_type = L2-norm
 *
 * @param[in]  gx_ptr        Pointer to source image. Gx image. Data type supported S16
 * @param[in]  gy_ptr        Pointer to source image. Gy image. Data type supported S16
 * @param[out] magnitude_ptr Pointer to destination image. Magnitude. Data type supported U16
 * @param[out] phase_ptr     Pointer to destination image. Quantized phase. Data type supported U8
 */
void mag_phase_l2norm_S16_S16_U16_U8(const void *__restrict gx_ptr, const void *__restrict gy_ptr, void *__restrict magnitude_ptr, void *__restrict phase_ptr)
{
    const auto gx        = static_cast<const int16_t *__restrict>(gx_ptr);
    const auto gy        = static_cast<const int16_t *__restrict>(gy_ptr);
    const auto magnitude = static_cast<uint16_t *__restrict>(magnitude_ptr);
    const auto phase     = static_cast<uint8_t *__restrict>(phase_ptr);

    const int16x8x4_t gx_val =
    {
        {
            vld1q_s16(gx),
            vld1q_s16(gx + 8),
            vld1q_s16(gx + 16),
            vld1q_s16(gx + 24)
        }
    };

    const int16x8x4_t gy_val =
    {
        {
            vld1q_s16(gy),
            vld1q_s16(gy + 8),
            vld1q_s16(gy + 16),
            vld1q_s16(gy + 24)
        }
    };

    // Compute and store phase
    vst1_u8(phase + 0, phase_quantization_S16_S16(gx_val.val[0], gy_val.val[0]));
    vst1_u8(phase + 8, phase_quantization_S16_S16(gx_val.val[1], gy_val.val[1]));
    vst1_u8(phase + 16, phase_quantization_S16_S16(gx_val.val[2], gy_val.val[2]));
    vst1_u8(phase + 24, phase_quantization_S16_S16(gx_val.val[3], gy_val.val[3]));

    // Compute and store magnitude using L2 normalization
    vst1q_u16(magnitude + 0, mag_l2_S16_S16(gx_val.val[0], gy_val.val[0]));
    vst1q_u16(magnitude + 8, mag_l2_S16_S16(gx_val.val[1], gy_val.val[1]));
    vst1q_u16(magnitude + 16, mag_l2_S16_S16(gx_val.val[2], gy_val.val[2]));
    vst1q_u16(magnitude + 24, mag_l2_S16_S16(gx_val.val[3], gy_val.val[3]));
}

/* Gradient function used when the gradient size = 7 and when the norm_type = L1-norm
 *
 * @param[in]  gx_ptr        Pointer to source image. Gx image. Data type supported S32
 * @param[in]  gy_ptr        Pointer to source image. Gy image. Data type supported S32
 * @param[out] magnitude_ptr Pointer to destination image. Magnitude. Data type supported U32
 * @param[out] phase_ptr     Pointer to destination image. Quantized phase. Data type support U8
 */
void mag_phase_l1norm_S32_S32_U32_U8(const void *__restrict gx_ptr, const void *__restrict gy_ptr, void *__restrict magnitude_ptr, void *__restrict phase_ptr)
{
    auto gx        = static_cast<const int32_t *__restrict>(gx_ptr);
    auto gy        = static_cast<const int32_t *__restrict>(gy_ptr);
    auto magnitude = static_cast<uint32_t *__restrict>(magnitude_ptr);
    auto phase     = static_cast<uint8_t *__restrict>(phase_ptr);

    // Process low and high part
    for(size_t i = 0; i < 2; ++i, gx += 16, gy += 16, magnitude += 16, phase += 16)
    {
        const int32x4x2_t gx0 =
        {
            {
                vld1q_s32(gx + 0),
                vld1q_s32(gx + 4)
            }
        };

        const int32x4x2_t gx1 =
        {
            {
                vld1q_s32(gx + 8),
                vld1q_s32(gx + 12)
            }
        };

        const int32x4x2_t gy0 =
        {
            {
                vld1q_s32(gy + 0),
                vld1q_s32(gy + 4)
            }
        };

        const int32x4x2_t gy1 =
        {
            {
                vld1q_s32(gy + 8),
                vld1q_s32(gy + 12)
            }
        };

        // Compute and store phase
        vst1_u8(phase + 0, phase_quantization_S32_S32(gx0, gy0));
        vst1_u8(phase + 8, phase_quantization_S32_S32(gx1, gy1));

        // Compute magnitude using L1 normalization
        const uint32x4x2_t mag0 = mag_l1_S32_S32(gx0, gy0);
        const uint32x4x2_t mag1 = mag_l1_S32_S32(gx1, gy1);

        // Store magnitude
        vst1q_u32(magnitude + 0, mag0.val[0]);
        vst1q_u32(magnitude + 4, mag0.val[1]);
        vst1q_u32(magnitude + 8, mag1.val[0]);
        vst1q_u32(magnitude + 12, mag1.val[1]);
    }
}

/* Gradient function used when the gradient size = 7 and when the norm_type = L2-norm
 *
 * @param[in]  gx_ptr        Pointer to source image. Gx image. Data type supported S32
 * @param[in]  gy_ptr        Pointer to source image. Gy image. Data type supported S32
 * @param[out] magnitude_ptr Pointer to destination image. Magnitude. Data type supported U32
 * @param[out] phase_ptr     Pointer to destination image. Quantized phase. Data type supported U8
 */
void mag_phase_l2norm_S32_S32_U32_U8(const void *__restrict gx_ptr, const void *__restrict gy_ptr, void *__restrict magnitude_ptr, void *__restrict phase_ptr)
{
    auto gx        = static_cast<const int32_t *__restrict>(gx_ptr);
    auto gy        = static_cast<const int32_t *__restrict>(gy_ptr);
    auto magnitude = static_cast<uint32_t *__restrict>(magnitude_ptr);
    auto phase     = static_cast<uint8_t *__restrict>(phase_ptr);

    // Process low and high part
    for(size_t i = 0; i < 2; ++i, gx += 16, gy += 16, magnitude += 16, phase += 16)
    {
        const int32x4x2_t gx0 =
        {
            {
                vld1q_s32(gx + 0),
                vld1q_s32(gx + 4)
            }
        };

        const int32x4x2_t gx1 =
        {
            {
                vld1q_s32(gx + 8),
                vld1q_s32(gx + 12)
            }
        };

        const int32x4x2_t gy0 =
        {
            {
                vld1q_s32(gy + 0),
                vld1q_s32(gy + 4)
            }
        };

        const int32x4x2_t gy1 =
        {
            {
                vld1q_s32(gy + 8),
                vld1q_s32(gy + 12)
            }
        };

        // Compute and store phase
        vst1_u8(phase + 0, phase_quantization_S32_S32(gx0, gy0));
        vst1_u8(phase + 8, phase_quantization_S32_S32(gx1, gy1));

        // Compute magnitude using L2 normalization
        const uint32x4x2_t mag0 = mag_l2_S32_S32(gx0, gy0);
        const uint32x4x2_t mag1 = mag_l2_S32_S32(gx1, gy1);

        // Store magnitude
        vst1q_u32(magnitude + 0, mag0.val[0]);
        vst1q_u32(magnitude + 4, mag0.val[1]);
        vst1q_u32(magnitude + 8, mag1.val[0]);
        vst1q_u32(magnitude + 12, mag1.val[1]);
    }
}

/* Computes non-maxima suppression and hysteresis when the gradient size = 3 or 5
 *
 * @param[in]  magnitude_ptr Pointer to source image. Magnitude. Data type supported U16
 * @param[in]  phase_ptr     Pointer to source image. Quantized phase. Data type supported U8
 * @param[out] output_ptr    Pointer to output image. Data type supported U8
 * @param[in]  stride_mag    Stride of magnitude image
 * @param[in]  lower_thr     Lower threshold used for the hysteresis
 * @param[in]  upper_thr     Upper threshold used for the hysteresis
 */
void non_max_suppression_U16_U8_U8(const void *__restrict magnitude_ptr, const void *__restrict phase_ptr, void *__restrict output_ptr, const uint32_t stride_mag, const int32_t lower_thr,
                                   const int32_t upper_thr)
{
    const auto magnitude = static_cast<const uint16_t *__restrict>(magnitude_ptr);
    const auto phase     = static_cast<const uint8_t *__restrict>(phase_ptr);
    const auto output    = static_cast<uint8_t *__restrict>(output_ptr);

    // Get magnitude and phase of the centre pixels
    uint16x8_t mc = vld1q_u16(magnitude);

    // Angle_quantized: 0 = 0°, 1 = 45°, 2 = 90°, 3 = 135°
    const uint16x8_t pc16 = vmovl_u8(vld1_u8(phase));

    // 0 degree
    const uint16x8_t mk0_0 = vld1q_u16(magnitude - 1);
    const uint16x8_t mk0_1 = vld1q_u16(magnitude + 1);
    uint16x8_t       mask0 = vceqq_u16(pc16, vdupq_n_u16(0));
    mask0                  = vandq_u16(mask0, vcgtq_u16(mc, mk0_0));
    mask0                  = vandq_u16(mask0, vcgtq_u16(mc, mk0_1));

    // 45 degree
    const uint16x8_t mk45_0 = vld1q_u16(magnitude - stride_mag - 1);
    const uint16x8_t mk45_1 = vld1q_u16(magnitude + stride_mag + 1);
    uint16x8_t       mask1  = vceqq_u16(pc16, vdupq_n_u16(1));
    mask1                   = vandq_u16(mask1, vcgtq_u16(mc, mk45_0));
    mask1                   = vandq_u16(mask1, vcgtq_u16(mc, mk45_1));

    // 90 degree
    const uint16x8_t mk90_0 = vld1q_u16(magnitude - stride_mag);
    const uint16x8_t mk90_1 = vld1q_u16(magnitude + stride_mag);
    uint16x8_t       mask2  = vceqq_u16(pc16, vdupq_n_u16(2));
    mask2                   = vandq_u16(mask2, vcgtq_u16(mc, mk90_0));
    mask2                   = vandq_u16(mask2, vcgtq_u16(mc, mk90_1));

    // 135 degree
    const uint16x8_t mk135_0 = vld1q_u16(magnitude - stride_mag + 1);
    const uint16x8_t mk135_1 = vld1q_u16(magnitude + stride_mag - 1);
    uint16x8_t       mask3   = vceqq_u16(pc16, vdupq_n_u16(3));
    mask3                    = vandq_u16(mask3, vcgtq_u16(mc, mk135_0));
    mask3                    = vandq_u16(mask3, vcgtq_u16(mc, mk135_1));

    // Merge masks
    mask0 = vorrq_u16(mask0, mask1);
    mask2 = vorrq_u16(mask2, mask3);
    mask0 = vorrq_u16(mask0, mask2);

    mc = vbslq_u16(mask0, mc, vdupq_n_u16(0));

    // mc > upper_thr
    mask0 = vcgtq_u16(mc, vdupq_n_u16(upper_thr));

    // mc <= lower_thr
    mask1 = vcleq_u16(mc, vdupq_n_u16(lower_thr));

    // mc <= upper_thr && mc > lower_thr
    mask2 = vcleq_u16(mc, vdupq_n_u16(upper_thr));
    mask2 = vandq_u16(mask2, vcgtq_u16(mc, vdupq_n_u16(lower_thr)));

    mc = vbslq_u16(mask0, vdupq_n_u16(EDGE), mc);
    mc = vbslq_u16(mask1, vdupq_n_u16(NO_EDGE), mc);
    mc = vbslq_u16(mask2, vdupq_n_u16(MAYBE), mc);

    vst1_u8(output, vmovn_u16(mc));
}

inline uint16x4_t non_max_U32_helper(const uint32_t *input, const uint16x4_t pc, const uint32_t stride_mag, const int32_t lower_thr, const int32_t upper_thr)
{
    // Phase for 4 pixel
    const uint32x4_t pc32 = vmovl_u16(pc);

    // Get magnitude for 4 pixel
    uint32x4_t mc = vld1q_u32(input);

    // Angle_quantized: 0 = 0°, 1 = 45°, 2 = 90°, 3 = 135°
    // 0 degree
    const uint32x4_t mk0_0 = vld1q_u32(input - 1);
    const uint32x4_t mk0_1 = vld1q_u32(input + 1);
    uint32x4_t       mask0 = vceqq_u32(pc32, vdupq_n_u32(0));
    mask0                  = vandq_u32(mask0, vcgtq_u32(mc, mk0_0));
    mask0                  = vandq_u32(mask0, vcgtq_u32(mc, mk0_1));

    // 45 degree
    const uint32x4_t mk45_0 = vld1q_u32(input - stride_mag - 1);
    const uint32x4_t mk45_1 = vld1q_u32(input + stride_mag + 1);
    uint32x4_t       mask1  = vceqq_u32(pc32, vdupq_n_u32(1));
    mask1                   = vandq_u32(mask1, vcgtq_u32(mc, mk45_0));
    mask1                   = vandq_u32(mask1, vcgtq_u32(mc, mk45_1));

    // 90 degree
    const uint32x4_t mk90_0 = vld1q_u32(input - stride_mag);
    const uint32x4_t mk90_1 = vld1q_u32(input + stride_mag);
    uint32x4_t       mask2  = vceqq_u32(pc32, vdupq_n_u32(2));
    mask2                   = vandq_u32(mask2, vcgtq_u32(mc, mk90_0));
    mask2                   = vandq_u32(mask2, vcgtq_u32(mc, mk90_1));

    // 135 degree
    const uint32x4_t mk135_0 = vld1q_u32(input - stride_mag + 1);
    const uint32x4_t mk135_1 = vld1q_u32(input + stride_mag - 1);
    uint32x4_t       mask3   = vceqq_u32(pc32, vdupq_n_u32(3));
    mask3                    = vandq_u32(mask3, vcgtq_u32(mc, mk135_0));
    mask3                    = vandq_u32(mask3, vcgtq_u32(mc, mk135_1));

    // Merge masks
    mask0 = vorrq_u32(mask0, mask1);
    mask2 = vorrq_u32(mask2, mask3);
    mask0 = vorrq_u32(mask0, mask2);

    mc = vbslq_u32(mask0, mc, vdupq_n_u32(0));

    // mc > upper_thr
    mask0 = vcgtq_u32(mc, vdupq_n_u32(upper_thr));

    // mc <= lower_thr
    mask1 = vcleq_u32(mc, vdupq_n_u32(lower_thr));

    // mc <= upper_thr && mc > lower_thr
    mask2 = vcleq_u32(mc, vdupq_n_u32(upper_thr));
    mask2 = vandq_u32(mask2, vcgtq_u32(mc, vdupq_n_u32(lower_thr)));

    mc = vbslq_u32(mask0, vdupq_n_u32(EDGE), mc);
    mc = vbslq_u32(mask1, vdupq_n_u32(NO_EDGE), mc);
    mc = vbslq_u32(mask2, vdupq_n_u32(MAYBE), mc);

    return vmovn_u32(mc);
}

/* Computes non-maxima suppression and hysteresis when the gradient_size = 7
 *
 * @param[in]  magnitude_ptr Pointer to source image. Magnitude. Data type supported U32
 * @param[in]  phase_ptr     Pointer to source image. Quantized phase. Data type supported U8
 * @param[out] output_ptr    Pointer to destination image. Data type supported U8
 * @param[in]  stride_mag    Stride of magnitude image
 * @param[in]  lower_thr     Lower threshold used for the hysteresis
 * @param[in]  upper_thr     Upper threshold used for the hysteresis
 */
void non_max_suppression_U32_U8_U8(const void *__restrict magnitude_ptr, const void *__restrict phase_ptr, void *__restrict output_ptr, const uint32_t stride_mag, const int32_t lower_thr,
                                   const int32_t upper_thr)
{
    const auto magnitude = static_cast<const uint32_t *__restrict>(magnitude_ptr);
    const auto phase     = static_cast<const uint8_t *__restrict>(phase_ptr);
    const auto output    = static_cast<uint8_t *__restrict>(output_ptr);

    // Get phase for 8 pixel
    const uint16x8_t pc16 = vmovl_u8(vld1_u8(phase));

    // Compute non maxima suppression
    const uint16x4x2_t res =
    {
        {
            non_max_U32_helper(magnitude, vget_low_u16(pc16), stride_mag, lower_thr, upper_thr),
            non_max_U32_helper(magnitude + 4, vget_high_u16(pc16), stride_mag, lower_thr, upper_thr)
        }
    };

    // Store result
    vst1_u8(output, vmovn_u16(vcombine_u16(res.val[0], res.val[1])));
}

/* Computes edge tracing when is called by edge_trace_U8_U8 recursively
 *
 * @param[in]  input         Pointer to source image. Data type supported U8
 * @param[out] output        Pointer to destination image. Data type supported U8
 * @param[in]  input_stride  Stride of the input image
 * @param[in]  output_stride Stride of the output image
 */
void edge_trace_recursive_U8_U8(uint8_t *__restrict input, uint8_t *__restrict output, const int32_t input_stride, const int32_t output_stride)
{
    // Look for MAYBE pixels in 8 directions
    *output = EDGE;

    // (-1, 0)
    uint8_t pixel = *(input - 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input - 1) = EDGE;

        edge_trace_recursive_U8_U8(input - 1, output - 1, input_stride, output_stride);
    }

    // (+1, 0)
    pixel = *(input + 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input + 1) = EDGE;

        edge_trace_recursive_U8_U8(input + 1, output + 1, input_stride, output_stride);
    }

    input -= input_stride;
    output -= output_stride;

    // (-1, -1)
    pixel = *(input - 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input - 1) = EDGE;

        edge_trace_recursive_U8_U8(input - 1, output - 1, input_stride, output_stride);
    }

    // (0, -1)
    pixel = *input;

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *input = EDGE;

        edge_trace_recursive_U8_U8(input, output, input_stride, output_stride);
    }

    // (+1, -1)
    pixel = *(input + 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input + 1) = EDGE;

        edge_trace_recursive_U8_U8(input + 1, output + 1, input_stride, output_stride);
    }

    input += input_stride * 2;
    output += output_stride * 2;

    // (-1, +1)
    pixel = *(input - 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input - 1) = EDGE;

        edge_trace_recursive_U8_U8(input - 1, output - 1, input_stride, output_stride);
    }

    // (0, +1)
    pixel = *input;

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *input = EDGE;

        edge_trace_recursive_U8_U8(input, output, input_stride, output_stride);
    }

    // (+1, +1)
    pixel = *(input + 1);

    if(pixel == MAYBE)
    {
        // Touched a MAYBE point. MAYBE becomes EDGE
        *(input + 1) = EDGE;

        edge_trace_recursive_U8_U8(input + 1, output + 1, input_stride, output_stride);
    }
}

/* Computes edge tracing
 *
 * @param[in]  input         Pointer to source image. Data type supported U8
 * @param[out] output        Pointer to destination image. Data type supported U8
 * @param[in]  input_stride  Stride of the input image
 * @param[in]  output_stride Stride of the output image
 */
void edge_trace_U8_U8(uint8_t *__restrict input, uint8_t *__restrict output, const int32_t input_stride, const int32_t output_stride)
{
    if(*input == NO_EDGE)
    {
        *output = NO_EDGE;
    }
    // Check if EDGE and not yet touched
    else if((*input == EDGE) && (*output == NO_EDGE))
    {
        edge_trace_recursive_U8_U8(input, output, input_stride, output_stride);
    }
}
} // namespace

NEGradientKernel::NEGradientKernel()
    : _func(nullptr), _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr)
{
}

void NEGradientKernel::configure(const ITensor *gx, const ITensor *gy, ITensor *magnitude, ITensor *phase, int32_t norm_type)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(gx, gy, magnitude, phase);

    set_shape_if_empty(*magnitude->info(), gx->info()->tensor_shape());
    set_shape_if_empty(*phase->info(), gx->info()->tensor_shape());

    Format magnitude_format = gx->info()->data_type() == DataType::S16 ? Format::U16 : Format::U32;
    set_format_if_unknown(*magnitude->info(), magnitude_format);
    set_format_if_unknown(*phase->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(gx, gy, magnitude, phase);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gx, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gy, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::U16, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(gx, gy);
    ARM_COMPUTE_ERROR_ON_MSG(element_size_from_data_type(gx->info()->data_type()) != element_size_from_data_type(magnitude->info()->data_type()), "Magnitude must have the same element size as Gx and Gy");

    _gx        = gx;
    _gy        = gy;
    _magnitude = magnitude;
    _phase     = phase;

    if(_gx->info()->data_type() == DataType::S16)
    {
        if(norm_type == 1)
        {
            _func = &mag_phase_l1norm_S16_S16_U16_U8;
        }
        else
        {
            _func = &mag_phase_l2norm_S16_S16_U16_U8;
        }
    }
    else
    {
        if(norm_type == 1)
        {
            _func = &mag_phase_l1norm_S32_S32_U32_U8;
        }
        else
        {
            _func = &mag_phase_l2norm_S32_S32_U32_U8;
        }
    }

    constexpr unsigned int num_elems_processed_per_iteration = 32;

    // Configure kernel window
    Window win = calculate_max_window(*_gx->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal gx_access(_gx->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal gy_access(_gy->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal mag_access(_magnitude->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal phase_access(_phase->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, gx_access, gy_access, mag_access, phase_access);

    mag_access.set_valid_region(win, _gx->info()->valid_region());
    phase_access.set_valid_region(win, _gx->info()->valid_region());

    INEKernel::configure(win);
}

void NEGradientKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator magnitude(_magnitude, window);
    Iterator phase(_phase, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        (*_func)(gx.ptr(), gy.ptr(), magnitude.ptr(), phase.ptr());
    },
    gx, gy, magnitude, phase);
}

NEEdgeNonMaxSuppressionKernel::NEEdgeNonMaxSuppressionKernel()
    : _func(nullptr), _magnitude(nullptr), _phase(nullptr), _output(nullptr), _lower_thr(0), _upper_thr(0)
{
}

BorderSize NEEdgeNonMaxSuppressionKernel::border_size() const
{
    return BorderSize(1);
}

void NEEdgeNonMaxSuppressionKernel::configure(const ITensor *magnitude, const ITensor *phase, ITensor *output,
                                              int32_t upper_thr, int32_t lower_thr, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(magnitude, phase, output);

    set_shape_if_empty(*output->info(), magnitude->info()->tensor_shape());

    set_format_if_unknown(*phase->info(), Format::U8);
    set_format_if_unknown(*output->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(magnitude, phase, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::U16, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(phase, output);

    _magnitude = magnitude;
    _phase     = phase;
    _output    = output;

    switch(_magnitude->info()->data_type())
    {
        case DataType::U16:
            _func = &non_max_suppression_U16_U8_U8;
            break;
        case DataType::U32:
            _func = &non_max_suppression_U32_U8_U8;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type!");
    }

    // Set thresholds
    _lower_thr = lower_thr;
    _upper_thr = upper_thr;

    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 10;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    // Configure kernel window
    Window win = calculate_max_window(*_magnitude->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  mag_access(_magnitude->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal phase_access(_phase->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(_output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, mag_access, phase_access, output_access);

    output_access.set_valid_region(win, _magnitude->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEEdgeNonMaxSuppressionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    Iterator magnitude(_magnitude, window);
    Iterator phase(_phase, window);
    Iterator output(_output, window);

    const size_t input1_stride        = _magnitude->info()->strides_in_bytes()[1];
    const size_t input1_stride_ushort = input1_stride / data_size_from_type(_magnitude->info()->data_type());

    execute_window_loop(window, [&](const Coordinates &)
    {
        (*_func)(magnitude.ptr(), phase.ptr(), output.ptr(), input1_stride_ushort, _lower_thr, _upper_thr);
    },
    magnitude, phase, output);
}

NEEdgeTraceKernel::NEEdgeTraceKernel()
    : _input(nullptr), _output(nullptr)
{
}

BorderSize NEEdgeTraceKernel::border_size() const
{
    return BorderSize(1);
}

bool NEEdgeTraceKernel::is_parallelisable() const
{
    return false;
}

void NEEdgeTraceKernel::configure(ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    set_format_if_unknown(*input->info(), Format::U8);
    set_format_if_unknown(*output->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*_input->info(), Steps(num_elems_processed_per_iteration));

    const ValidRegion &input_valid_region  = input->info()->valid_region();
    const ValidRegion &output_valid_region = output->info()->valid_region();

    // Reads can occur within the valid region of the input + border
    AccessWindowStatic input_access(input->info(),
                                    input_valid_region.anchor[0] - border_size().left,
                                    input_valid_region.anchor[1] - border_size().top,
                                    input_valid_region.anchor[0] + input_valid_region.shape[0] + border_size().right,
                                    input_valid_region.anchor[1] + input_valid_region.shape[1] + border_size().bottom);

    // Writes can occur within the valid region of the output + border
    AccessWindowStatic output_access(output->info(),
                                     output_valid_region.anchor[0] - border_size().left,
                                     output_valid_region.anchor[1] - border_size().top,
                                     output_valid_region.anchor[0] + output_valid_region.shape[0] + border_size().right,
                                     output_valid_region.anchor[1] + output_valid_region.shape[1] + border_size().bottom);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, _input->info()->valid_region());

    INEKernel::configure(win);
}

void NEEdgeTraceKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    Iterator input(_input, window);
    Iterator output(_output, window);

    const size_t input_stride  = _input->info()->strides_in_bytes()[1];
    const size_t output_stride = _output->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates &)
    {
        edge_trace_U8_U8(input.ptr(), output.ptr(), input_stride, output_stride);
    },
    input, output);
}
