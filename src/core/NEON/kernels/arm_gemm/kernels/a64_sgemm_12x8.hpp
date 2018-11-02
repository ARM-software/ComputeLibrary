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

#ifdef __aarch64__

namespace arm_gemm
{
// Actual kernel implementations
void a64_sgemm_asimd_12x8(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_12x8_a53(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_12x8_a55(const float *, const float *, float *, int, int, int);
void a64_sgemm_asimd_12x8_a55r1(const float *, const float *, float *, int, int, int);

// 12x8 SGEMM "strategy" class.
//
// This describes the characteristics of a family of kernels, in terms of
// the required interleave properties and the output block size.
//
// All kernels in the family must share these characteristics.  The actual
// kernel to be used can be chosen at runtime, based on the CPU_type
// structure.
class sgemm_12x8
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

    /* Describes the data layout for A input */
    static const int A_interleave = 8;
    static const int A_block      = 1;
    static const int A_transpose  = 0;

    /* Same for B input */
    static const int B_interleave = 12;
    static const int B_block      = 1;
    static const int B_transpose  = 1;

    /* Kernel blocking parameters */
    static const int out_width  = 12;
    static const int out_height = 8;
    static const int k_unroll   = 1;

    kern_type kernel = a64_sgemm_asimd_12x8;

    sgemm_12x8(const CPUInfo *ci)
    {
        // Select specific kernel if available
        switch(ci->get_cpu_model())
        {
            case CPUModel::A53:
                kernel = a64_sgemm_asimd_12x8_a53;
                break;

            case CPUModel::A55r0:
                kernel = a64_sgemm_asimd_12x8_a55;
                break;

            case CPUModel::A55r1:
                kernel = a64_sgemm_asimd_12x8_a55r1;
                break;

            default:
                /* Generic kernel is initialized by default. */
                break;
        }
    }
};

} // namespace arm_gemm

#endif // __aarch64__
