/*
 * Copyright (c) 2017-2020, 2022 Arm Limited.
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
#include "arm_gemm.hpp"
#include "bfloat.hpp"
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_batched.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a32_sgemm_8x6.hpp"

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/a64_ffhybrid_bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_ffinterleaved_bf16fp32_dot_8x12.hpp"
#include "kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp"
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/a64_hybrid_bf16fp32_dot_6x16.hpp"
#include "kernels/a64_hybrid_bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_bf16fp32_dot_8x12.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_sgemm_8x12.hpp"

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/sve_ffhybrid_bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL.hpp"
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/sve_hybrid_bf16fp32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_8x3VL.hpp"

namespace arm_gemm {

static const GemmImplementation<bfloat16, float> gemm_bf16_methods[] =
{
#ifdef __aarch64__
#ifdef ARM_COMPUTE_ENABLE_BF16
#ifdef ARM_COMPUTE_ENABLE_SVE
// gemm_bf16_interleaved
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_bf16fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_bf16fp32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_bf16fp32_dot_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_bf16fp32_dot_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>2); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, float>(args); }
),
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
    KernelWeightFormat::VL2VL_BL64,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_ffhybrid_bf16fp32_mmla_6x4VL",
    KernelWeightFormat::VL2VL_BL64,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_bf16fp32_mmla_6x16",
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_bf16fp32_mmla_8x12",
    [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_bf16fp32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_bf16fp32_dot_8x12",
    [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize>2); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, float>(args); }
),
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_ffinterleaved_bf16fp32_mmla_8x12",
    KernelWeightFormat::VL256_BL64,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_ffhybrid_bf16fp32_mmla_6x16",
    KernelWeightFormat::VL256_BL64,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_ffinterleaved_bf16fp32_dot_8x12",
    KernelWeightFormat::VL128_BL32,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
GemmImplementation<bfloat16, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_BF16
#elif defined(__arm__)
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, bfloat16, float>(args); }
},
#else
# error "Unknown Architecture"
#endif
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<bfloat16, float> *gemm_implementation_list<bfloat16, float>() {
    return gemm_bf16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<bfloat16, float> gemm<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);
template bool has_opt_gemm<bfloat16, float, Nothing>(WeightFormat &weight_format, const GemmArgs &args, const Nothing &);
template KernelDescription get_gemm_method<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm
