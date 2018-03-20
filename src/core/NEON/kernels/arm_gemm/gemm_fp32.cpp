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
#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_native.hpp"
#include "gemv_native_transposed.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a64_sgemm_native_16x4.hpp"
#include "kernels/a64_sgemv_pretransposed.hpp"
#include "kernels/a64_sgemv_trans.hpp"

namespace arm_gemm
{
template <>
UniqueGemmCommon<float, float> gemm<float, float>(const CPUInfo &ci, const unsigned int M, const unsigned int N, const unsigned int K,
                                                  const bool trA, const bool trB, const float alpha, const float beta,
                                                  const int maxthreads, const bool pretransposed_hint)
{
#ifdef __aarch64__
    /* Cases in priority order */
    /* GemvPretransposed: requires M=1, alpha=1, and transposed hint set */
    if(M == 1 && alpha == 1.0f && pretransposed_hint)
    {
        return UniqueGemmCommon<float, float>(new GemvPretransposed<sgemv_pretransposed, float, float>(&ci, N, K, trB, beta));
    }

    /* GemvNativeTransposed: requires M=1, no trA or trB, doesn't handle alpha */
    if(M == 1 && alpha == 1.0f && !trA && !trB)
    {
        return UniqueGemmCommon<float, float>(new GemvNativeTransposed<sgemv_trans, float, float>(&ci, N, K, beta));
    }

    /* Native GEMM: requires M to be a multiple of 4, K at least 4, N a
     * multiple of 16, doesn't handle alpha and only makes sense for small
     * sizes.  */
    if(N <= 128 && K <= 128 && ((M % 4) == 0) && (K >= 4) && ((N % 16) == 0) && alpha == 1.0f)
    {
        return UniqueGemmCommon<float, float>(new GemmNative<sgemm_native_16x4, float, float>(&ci, M, N, K, beta));
    }

    /* Blocked GEMM, handles all cases. */
    return UniqueGemmCommon<float, float>(new GemmInterleaved<sgemm_12x8, float, float>(&ci, M, N, K, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
#else
    return UniqueGemmCommon<float, float>(new GemmInterleaved<sgemm_8x6, float, float>(&ci, M, N, K, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
#endif
}

// Instantiate static class variables.
#ifdef __aarch64__
const int sgemm_12x8::out_width;
const int sgemm_12x8::out_height;

const int sgemm_native_16x4::out_width;
const int sgemm_native_16x4::out_height;
#else
const int sgemm_8x6::out_width;
const int sgemm_8x6::out_height;
#endif

} // namespace arm_gemm
