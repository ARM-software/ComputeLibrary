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

// This can only be built if the target/compiler supports FP16 arguments.
#ifdef __ARM_FP16_ARGS

#include "arm_gemm.hpp"

#include "gemm_common.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_hgemm_24x8.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a32_sgemm_8x6.hpp"

namespace arm_gemm {

template<>
UniqueGemmCommon<__fp16, __fp16> gemm(const CPUInfo &ci, const unsigned int M, const unsigned int N, const unsigned int K,
                                      const unsigned int nbatches, const unsigned int nmulti,
                                      const bool trA, const bool trB, const __fp16 alpha, const __fp16 beta,
                                      const int maxthreads, const bool pretransposed_hint) {
#ifdef __aarch64__

// Only consider the native FP16 kernel if it will get built.
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(FP16_KERNELS)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // If the compiler is configured to enable this feature always, then assume it is available at runtime too.
    const bool use_fp16=true;
#else
    // Otherwise, detect at runtime via CPUInfo.
    const bool use_fp16=ci.has_fp16();
#endif

    // If FP16 is supported, use it.
    if (use_fp16) {
        return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<hgemm_24x8, __fp16, __fp16>(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
    }
#endif

    // Fallback to using the blocked SGEMM kernel.
    return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<sgemm_12x8, __fp16, __fp16>(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
#else
    // For AArch32, only support the SGEMM route for now.
    return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<sgemm_8x6, __fp16, __fp16>(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
#endif
}

// Instantiate static class members if necessary.
#if defined(__aarch64__) && (defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(FP16_KERNELS))
const int hgemm_24x8::out_width;
const int hgemm_24x8::out_height;
#endif

} // namespace arm_gemm

#endif // __ARM_FP16_ARGS
