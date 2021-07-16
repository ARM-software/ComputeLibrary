/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_batched.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_hybrid_fp32bf16fp32_mmla_4x24.hpp"
#include "kernels/a64_hybrid_fp32bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_hybrid_fp32_mla_4x24.hpp"
#include "kernels/a64_hybrid_fp32_mla_6x16.hpp"
#include "kernels/a64_hybrid_fp32_mla_8x4.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/a64_sgemm_8x6.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_6x4.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_8x4.hpp"

#include "kernels/sve_hybrid_fp32bf16fp32_mmla_4x6VL.hpp"
#include "kernels/sve_hybrid_fp32bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_8x1VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_8x3VL.hpp"
#include "kernels/sve_interleaved_fp32_mla_8x3VL.hpp"
#include "kernels/sve_interleaved_fp32_mmla_8x3VL.hpp"
#include "kernels/sve_smallK_hybrid_fp32_mla_8x1VL.hpp"

namespace arm_gemm {

static const GemmImplementation<float, float> gemm_fp32_methods[] =
{
// GEMV cases - starting with 'gemv_batched' wrapper to turn batched GEMV into GEMM.
{
    GemmMethod::GEMV_BATCHED,
    "gemv_batched",
    [](const GemmArgs &args) { return args._Msize==1 && args._nbatches>1 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemvBatched<float, float>(args); }
},
#ifdef __aarch64__
#ifdef ARM_COMPUTE_ENABLE_BF16
// "fast mode" (BF16) kernels
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_bf16fp32_mmla_8x12",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp32bf16fp32_mmla_6x16",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp32bf16fp32_mmla_4x24",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_BF16
#ifdef ARM_COMPUTE_ENABLE_SVE
#ifdef ARM_COMPUTE_ENABLE_BF16
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_bf16fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32bf16fp32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32bf16fp32_mmla_4x6VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_BF16
#ifdef ARM_COMPUTE_ENABLE_SVEF32MM
// MMLA next due to higher throughput (which is SVE only)
// Prefer this in all cases, except if fast mode is requested and BF16 is available.
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svef32mm() && (args._Ksize>4); },
    [](const GemmArgs &args) { return !(args._fast_mode && args._ci->has_bf16()); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mmla_8x3VL, float, float>(args); }
},
#endif // ARM_COMPUTE_ENABLE_SVEF32MM
// SVE kernels
{
    GemmMethod::GEMM_HYBRID,
    "sve_smallK_hybrid_fp32_mla_8x1VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && args._Ksize <= 24 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_sve_smallK_hybrid_fp32_mla_8x1VL, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32_mla_8x1VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return (args._Nsize < 12); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_8x1VL, float, float>(args); }
},
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_fp32_mla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float>(args); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
// Cortex-A35 specific kernel - use for any problem on A35, and never in any other cases.
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x6",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A35; },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x6, float, float>(args); }
},
// Arm® Neon™ hybrid methods
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_fp32_mla_8x4",
    [](const GemmArgs &args) { return args._Ksize <= 8 && (args._Nsize % 4)==0 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_8x4, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_fp32_mla_6x4",
    [](const GemmArgs &args) { return (args._Ksize > 8 && args._Ksize <= 16) && (args._Nsize % 4)==0 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_6x4, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp32_mla_8x4",
    nullptr,
    [](const GemmArgs &args) { return (args._Nsize < 12); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_8x4, float, float>(args); }
},
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp32_mla_4x24",
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp32_mla_6x16",
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float>(args); }
),
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_sgemm_8x12, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, float, float>(args); }
),
#endif // __aarch64__

#ifdef __arm__
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, float, float>(args); }
},
#endif // __arm__
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

/* Templated function to return this list. */
template<>
const GemmImplementation<float, float> *gemm_implementation_list<float, float>() {
    return gemm_fp32_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<float, float> gemm<float, float, Nothing>(const GemmArgs &args, const Nothing &);
template KernelDescription get_gemm_method<float, float, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<float, float, Nothing> (const GemmArgs &args, const Nothing &);

} // namespace arm_gemm
