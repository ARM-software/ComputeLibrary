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
#ifdef __aarch64__

#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_gemm_s16_12x8.hpp"
#include "kernels/a64_gemm_s8_12x8.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"

namespace arm_gemm
{
template <>
UniqueGemmCommon<int8_t, int32_t> gemm<int8_t, int32_t>(const CPUInfo &ci, const unsigned int M, const unsigned int N, const unsigned int K,
                                                        const unsigned int nbatches, const unsigned int nmulti,
                                                        const bool trA, const bool trB, const int32_t alpha, const int32_t beta,
                                                        const int maxthreads, const bool pretransposed_hint)
{
    if(ci.has_dotprod())
    {
        // Dot product supporting CPUs.  This family has a special version for A55r1.
        return UniqueGemmCommon<int8_t, int32_t>(new GemmInterleaved<gemm_s8_12x8, int8_t, int32_t>(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint));
    }

    return UniqueGemmCommon<int8_t, int32_t>(new GemmInterleaved<gemm_s8_4x4, int8_t, int32_t>(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint));

    // TODO: There's a better approach for A53, but it doesn't work
    // well on heterogeneous systems as the required data formats
    // are different.  Figure out how to enable this:
    // gemm = new GemmInterleaved<gemm_s16_12x8, int8_t, int32_t>(ci, M, N, K, trA, trB);
}

// Instantiate static class members
const int gemm_s8_12x8::out_width;
const int gemm_s8_12x8::out_height;
const int gemm_s8_4x4::out_width;
const int gemm_s8_4x4::out_height;

} // namespace arm_gemm

#endif // aarch64
