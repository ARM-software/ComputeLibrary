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
#pragma once

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)

#include "arm_gemm.hpp"

namespace arm_gemm
{
// Actual kernel implementations
void a64_hgemm_asimd_24x8(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);
void a64_hgemm_asimd_24x8_a55r1(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);

// 24x8 HGEMM "strategy" class.  Describes the kernel properties.
//
// The generic "gemm_opt" function will instantiate one of these (allowing
// the constructor to pick a kernel implementation).
class hgemm_24x8
{
public:
    typedef __fp16 operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);

    static const int  A_block      = 1;
    static const int  A_interleave = 8;
    static const bool A_transpose  = false;

    static const int  B_block      = 1;
    static const int  B_interleave = 24;
    static const bool B_transpose  = true;

    static const int out_width  = 24;
    static const int out_height = 8;
    static const int k_unroll   = 1;

    // Default to the generic kernel
    kern_type kernel = a64_hgemm_asimd_24x8;

    hgemm_24x8(const CPUInfo *ci)
    {
        if(ci->get_cpu_model() == CPUModel::A55r1)
        {
            kernel = a64_hgemm_asimd_24x8_a55r1;
        }
    }
};

} // namespace arm_gemm

#endif // __aarch64__
