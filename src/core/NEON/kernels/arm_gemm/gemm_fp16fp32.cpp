/*
 * Copyright (c) 2025-2026 Arm Limited.
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
#if defined(__aarch64__) && (defined(ENABLE_FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include "arm_gemm/arm_gemm.hpp"
#include "arm_gemm/gemm_common.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_hybrid_fp16fp32_mla_6x16.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#ifdef ARM_COMPUTE_ENABLE_SME2
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SME2
#ifdef ARM_COMPUTE_ENABLE_SME
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SME
#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_hybrid_fp16fp32_mla_6x4VL.hpp"
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/sve_ffhybrid_fp16fp32_mla_6x4VL.hpp"
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_SVE
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/a64_ffhybrid_fp16fp32_mla_6x16.hpp"
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

namespace arm_gemm {

static const GemmImplementation<__fp16, __fp16, float> gemm_fp16fp32_methods[] =
{
#ifdef ARM_COMPUTE_ENABLE_SME2
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL, __fp16, __fp16, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL, __fp16, __fp16, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL, __fp16, __fp16, float>(args); }
},
#endif // ARM_COMPUTE_ENABLE_SME2
#ifdef ARM_COMPUTE_ENABLE_SME
{
    "sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL, __fp16, __fp16, float>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL, __fp16, __fp16, float>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL, __fp16, __fp16, float>(args); }
},
#endif // ARM_COMPUTE_ENABLE_SME
#ifdef ARM_COMPUTE_ENABLE_SVE
{
    "sve_hybrid_fp16fp32_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16fp32_mla_6x4VL, __fp16, __fp16, float>(args); }
},
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
{
    "sve_ffhybrid_fp16fp32_mla_6x4VL",
    KernelWeightFormat::VL1VL_BL16,
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16fp32_mla_6x4VL, __fp16, float>(args); }
},
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_SVE
{
    "a64_hybrid_fp16fp32_mla_6x16",
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp16fp32_mla_6x16, __fp16, __fp16, float>(args); }
},
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
{
    "a64_ffhybrid_fp16fp32_mla_6x16",
    KernelWeightFormat::VL128_BL16,
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16fp32_mla_6x16, __fp16, float>(args); }
},
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
{
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return !args._ci->has_fp16(); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, __fp16, __fp16, float>(args); }
},
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<__fp16, __fp16, float> *gemm_implementation_list<__fp16, __fp16, float>() {
    return gemm_fp16fp32_methods;
}

template UniqueGemmCommon<__fp16, __fp16, float> gemm<__fp16, __fp16, float, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<__fp16, __fp16, float, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16, __fp16, float, Nothing>(const GemmArgs &, const Nothing &);

} // namespace arm_gemm

#endif // defined(__aarch64__) && (defined(ENABLE_FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

