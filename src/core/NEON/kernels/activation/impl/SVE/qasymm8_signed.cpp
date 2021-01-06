/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Validate.h"

#include <cmath>
#include <cstddef>

#if defined(__ARM_FEATURE_SVE2)
#include "src/core/NEON/SVEAsymm.h"
#include "src/core/NEON/SVEMath.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void qasymm8_signed_sve_activation(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const UniformQuantizationInfo qi_in           = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out          = dst->info()->quantization_info().uniform();
    const auto                    va              = svdup_n_s8(quantize_qasymm8_signed(act_info.a(), qi_in));
    const auto                    vb              = svdup_n_s8(quantize_qasymm8_signed(act_info.b(), qi_in));
    const auto                    const_0         = quantize_qasymm8_signed(0.f, qi_in);
    const auto                    vconst_0        = svdup_n_s8(const_0);
    const auto                    vconst_1        = svdup_n_f32(1.f);
    const auto                    va_f32          = svdup_n_f32(act_info.a());
    const auto                    vb_f32          = svdup_n_f32(act_info.b());
    const auto                    const_6_f32     = svdup_n_f32(6.f);
    const auto                    const_0_f32     = svdup_n_f32(0.f);
    const auto                    const_3_f32     = svdup_n_f32(3.f);
    const auto                    const_inv_6_f32 = svdup_n_f32(0.166666667f);

    // Initialise scale/offset for re-quantization
    bool requant = true;
    if(qi_in.scale == qi_out.scale && qi_in.offset == qi_out.offset)
    {
        requant = false;
    }
    float s  = qi_in.scale / qi_out.scale;
    float o  = -qi_in.offset * s + qi_out.offset;
    auto  vs = svdup_n_f32(s);
    auto  vo = svdup_n_f32(o);

    // Initialise scale/offset for re-quantization with int32_t
    const auto  voffset_in      = svdup_n_s32(qi_in.offset);
    int32_t     s_s32           = round(s * (1 << 8), arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    int32_t     o_s32           = round(o * (1 << 8), arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    const auto  vs_s32          = svdup_n_s32(s_s32);
    const auto  vo_s32          = svdup_n_s32(o_s32);

    // Initialise scale/offset for re-quantization for leaky relu
    int32_t     s_leaky_s32     = round(s * act_info.a() * (1 << 8), arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    int32_t     o_leaky_s32     = round((-qi_in.offset * s * act_info.a() + qi_out.offset) * (1 << 8),
                                             arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    const auto  vs_leaky_s32    = svdup_n_s32(s_leaky_s32);
    const auto  vo_leaky_s32    = svdup_n_s32(o_leaky_s32);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        svint8_t tmp;

        int      x  = window_start_x;
        svbool_t pg = svwhilelt_b8(x, window_end_x);
        do
        {
            const auto vin = svld1_s8(pg, input_ptr + x);
            if(act == ActivationLayerInfo::ActivationFunction::RELU)
            {
                // Perform activation
                tmp = svmax_s8_z(pg, vconst_0, vin);
                // Re-quantize to new output space
                tmp = requant ? svmla_qasymm8_signed_z(pg, tmp, vs, vo) : tmp;
            }
            else if(act == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
            {
                // Perform activation
                tmp = svmin_s8_z(pg, va, svmax_s8_z(pg, vconst_0, vin));
                // Re-quantize to new output space
                tmp = requant ? svmla_qasymm8_signed_z(pg, tmp, vs, vo) : tmp;
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
            {
                // Perform activation
                tmp = svmin_s8_z(pg, va, svmax_s8_z(pg, vb, vin));
                // Re-quantize to new output space
                tmp = requant ? svmla_qasymm8_signed_z(pg, tmp, vs, vo) : tmp;
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                // De-quantize
                const auto vin_deq = svdequantize_z(pg, vin, qi_in);
                // Perform activation
                const svfloat32x4_t tmp_dep =
                {
                    { {
                            svdiv_f32_z(pg, vconst_1, svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget4_f32(vin_deq, 0))))),
                            svdiv_f32_z(pg, vconst_1, svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget4_f32(vin_deq, 1))))),
                            svdiv_f32_z(pg, vconst_1, svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget4_f32(vin_deq, 2))))),
                            svdiv_f32_z(pg, vconst_1, svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget4_f32(vin_deq, 3))))),
                        }
                    }
                };
                // Re-quantize to new output space
                tmp = svquantize_signed_z(pg, tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::TANH)
            {
                // De-quantize
                const auto vin_deq = svdequantize_z(pg, vin, qi_in);
                // Perform activation
                const svfloat32x4_t tmp_dep =
                {
                    { {
                            svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget4_f32(vin_deq, 0), vb_f32))),
                            svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget4_f32(vin_deq, 1), vb_f32))),
                            svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget4_f32(vin_deq, 2), vb_f32))),
                            svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget4_f32(vin_deq, 3), vb_f32))),
                        }
                    }
                };
                // Re-quantize to new output space
                tmp = svquantize_signed_z(pg, tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::HARD_SWISH)
            {
                // De-quantize
                const auto vin_deq = svdequantize_z(pg, vin, qi_in);
                // Perform activation
                const svfloat32x4_t tmp_dep =
                {
                    { {
                            svmul_f32_z(pg, svget4_f32(vin_deq, 0), svmul_f32_z(pg, const_inv_6_f32, svmin_f32_z(pg, const_6_f32, svmax_f32_z(pg, const_0_f32, svadd_f32_z(pg, svget4_f32(vin_deq, 0), const_3_f32))))),
                            svmul_f32_z(pg, svget4_f32(vin_deq, 1), svmul_f32_z(pg, const_inv_6_f32, svmin_f32_z(pg, const_6_f32, svmax_f32_z(pg, const_0_f32, svadd_f32_z(pg, svget4_f32(vin_deq, 1), const_3_f32))))),
                            svmul_f32_z(pg, svget4_f32(vin_deq, 2), svmul_f32_z(pg, const_inv_6_f32, svmin_f32_z(pg, const_6_f32, svmax_f32_z(pg, const_0_f32, svadd_f32_z(pg, svget4_f32(vin_deq, 2), const_3_f32))))),
                            svmul_f32_z(pg, svget4_f32(vin_deq, 3), svmul_f32_z(pg, const_inv_6_f32, svmin_f32_z(pg, const_6_f32, svmax_f32_z(pg, const_0_f32, svadd_f32_z(pg, svget4_f32(vin_deq, 3), const_3_f32))))),
                        }
                    }
                };
                // Re-quantize to new output space
                tmp = svquantize_signed_z(pg, tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LEAKY_RELU)
            {
                svbool_t p0, p1, p2, p3;
                svint32x4_t tmp_dep;

                // Expand to int32
                const svint32x4_t vin_s32 =
                {
                    { {
                            svmovlb_s32(svmovlb_s16(vin)),
                            svmovlt_s32(svmovlb_s16(vin)),
                            svmovlb_s32(svmovlt_s16(vin)),
                            svmovlt_s32(svmovlt_s16(vin)),
                    } }
                };

                // Compare elements to input offset
                if (qi_in.scale >= 0)
                {
                    p0 = svcmplt_s32(pg, svget4_s32(vin_s32, 0), voffset_in);
                    p1 = svcmplt_s32(pg, svget4_s32(vin_s32, 1), voffset_in);
                    p2 = svcmplt_s32(pg, svget4_s32(vin_s32, 2), voffset_in);
                    p3 = svcmplt_s32(pg, svget4_s32(vin_s32, 3), voffset_in);
                }
                else
                {
                    p0 = svcmpgt_s32(pg, svget4_s32(vin_s32, 0), voffset_in);
                    p1 = svcmpgt_s32(pg, svget4_s32(vin_s32, 1), voffset_in);
                    p2 = svcmpgt_s32(pg, svget4_s32(vin_s32, 2), voffset_in);
                    p3 = svcmpgt_s32(pg, svget4_s32(vin_s32, 3), voffset_in);
                }

                // Multiply negative elements and requantize if necessary
                if (requant)
                {
                    tmp_dep = svcreate4_s32(
                        svasr_n_s32_m(pg, svmla_s32_m(pg, svsel(p0, vo_leaky_s32, vo_s32), svget4_s32(vin_s32, 0), svsel(p0, vs_leaky_s32, vs_s32)), 8),
                        svasr_n_s32_m(pg, svmla_s32_m(pg, svsel(p1, vo_leaky_s32, vo_s32), svget4_s32(vin_s32, 1), svsel(p1, vs_leaky_s32, vs_s32)), 8),
                        svasr_n_s32_m(pg, svmla_s32_m(pg, svsel(p2, vo_leaky_s32, vo_s32), svget4_s32(vin_s32, 2), svsel(p2, vs_leaky_s32, vs_s32)), 8),
                        svasr_n_s32_m(pg, svmla_s32_m(pg, svsel(p3, vo_leaky_s32, vo_s32), svget4_s32(vin_s32, 3), svsel(p3, vs_leaky_s32, vs_s32)), 8)
                    );
                }
                else
                {
                    tmp_dep = svcreate4_s32(
                        svasr_n_s32_m(p0, svmad_s32_m(p0, svget4_s32(vin_s32, 0), vs_leaky_s32, vo_leaky_s32), 8),
                        svasr_n_s32_m(p1, svmad_s32_m(p1, svget4_s32(vin_s32, 1), vs_leaky_s32, vo_leaky_s32), 8),
                        svasr_n_s32_m(p2, svmad_s32_m(p2, svget4_s32(vin_s32, 2), vs_leaky_s32, vo_leaky_s32), 8),
                        svasr_n_s32_m(p3, svmad_s32_m(p3, svget4_s32(vin_s32, 3), vs_leaky_s32, vo_leaky_s32), 8)
                    );
                }

                // Convert uint32 vectors to uint16 vectors (with saturation)
                const auto v_low_s16 = svqxtnt_s32(svqxtnb_s32(svget4_s32(tmp_dep, 0)), svget4_s32(tmp_dep, 1));
                const auto v_high_s16 = svqxtnt_s32(svqxtnb_s32(svget4_s32(tmp_dep, 2)), svget4_s32(tmp_dep, 3));

                // convert uint16 vectors to uint8 vectors (with saturation)
                tmp = svqxtnt_s16(svqxtnb_s16(v_low_s16), v_high_s16);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported activation function");
            }

            svst1_s8(pg, output_ptr + x, tmp);

            x += svcntb();
            pg = svwhilelt_b8(x, window_end_x);

        }
        while(svptest_any(svptrue_b8(), pg));
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_SVE2) */
