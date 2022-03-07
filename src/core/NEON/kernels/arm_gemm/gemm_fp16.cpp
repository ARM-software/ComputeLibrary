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

// This can only be built if the target/compiler supports FP16 arguments.
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include "arm_gemm.hpp"

#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_hgemm_8x24.hpp"
#include "kernels/a64_hybrid_fp16_mla_6x32.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/sve_hybrid_fp16_mla_6x4VL.hpp"
#include "kernels/sve_interleaved_fp16_mla_8x3VL.hpp"

namespace arm_gemm {

static const GemmImplementation<__fp16, __fp16> gemm_fp16_methods[] = {
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<__fp16, __fp16>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_fp16_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_fp16_mla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize > 4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16>(args); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
#if defined(__aarch64__)
GemmImplementation<__fp16, __fp16>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_fp16_mla_6x32",
    [](const GemmArgs &args) { return args._ci->has_fp16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_hgemm_8x24",
    [](const GemmArgs &args) { return args._ci->has_fp16(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16>(args); }
),
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return !args._ci->has_fp16(); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, __fp16, __fp16>(args); }
},
#elif defined(__arm__)
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, __fp16, __fp16>(args); }
},
#else // not AArch64 or AArch32
# error Unknown Architecture
#endif
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr,
}
};

template<>
const GemmImplementation<__fp16, __fp16> *gemm_implementation_list<__fp16, __fp16>() {
    return gemm_fp16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<__fp16, __fp16> gemm<__fp16, __fp16, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16, __fp16, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))
