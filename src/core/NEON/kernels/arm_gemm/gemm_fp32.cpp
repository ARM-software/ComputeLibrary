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
#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_batched.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_gemv_fp32_mla_32.hpp"
#include "kernels/a64_hybrid_fp32_mla_6x16.hpp"
#include "kernels/a64_hybrid_fp32_mla_8x4.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_6x4.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_8x4.hpp"

#include "kernels/sve_gemv_fp32_mla_8VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_8x1VL.hpp"
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
#ifdef __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_HYBRID,
    "sve_gemv_fp32_mla_8VL",
    [](const GemmArgs &args) { return args._Msize==1 && args._nbatches==1 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sve_gemv_fp32_mla_8VL, float, float>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "a64_gemv_fp32_mla_32",
    [](const GemmArgs &args) { return args._Msize==1 && args._nbatches==1 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_a64_gemv_fp32_mla_32, float, float>(args); }
},

// MMLA next due to higher throughput (SVE only)
#if defined(__ARM_FEATURE_SVE) && defined(MMLA_FP32)
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mmla_8x3VL, float, float>(args); }
},
#endif // __ARM_FEATURE_SVE && MMLA_FP32

#ifdef __ARM_FEATURE_SVE
// SVE smallk / hybrid methods
{
    GemmMethod::GEMM_HYBRID,
    "sve_smallK_hybrid_fp32_mla_8x1VL",
    [](const GemmArgs &args) { return args._Ksize <= 24 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_sve_smallK_hybrid_fp32_mla_8x1VL, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32_mla_8x1VL",
    nullptr,
    [](const GemmArgs &args) { return (args._Nsize < 12); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_8x1VL, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp32_mla_6x4VL",
    nullptr,
    [](const GemmArgs &args) { return ((args._Ksize <= 256) && (args._Nsize <= 256)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float>(args); }
},
#endif // __ARM_FEATURE_SVE

// Neon hybrid methods
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
    "a64_hybrid_fp32_mla_6x16",
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float>::estimate_cycles(args, cls_a64_hybrid_fp32_mla_6x16::get_performance_parameters(args._ci)); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float>(args); }
),
#ifdef __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_fp32_mla_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float>(args); }
},
#endif // __ARM_FEATURE_SVE
GemmImplementation<float, float>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_sgemm_8x12, float, float>::estimate_cycles(args, cls_a64_sgemm_8x12::get_performance_parameters(args._ci)); },
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
