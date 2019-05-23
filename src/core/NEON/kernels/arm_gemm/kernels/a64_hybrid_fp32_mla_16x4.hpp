/*
 * Copyright (c) 2019 Arm Limited.
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


#include "../std_transforms_fixed.hpp"

namespace arm_gemm
{

// Actual kernel implementations
void a64_hybrid_fp32_mla_16x4(const float *, int, const float *, float *, int, float, int, int, int);
void a64_hybrid_fp32_mla_16x4_a55(const float *, int, const float *, float *, int, float, int, int, int);

class hybrid_fp32_mla_16x4
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, int, const float *, float *, int, float, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_height()
    {
        return 4;
    }

    static unsigned int out_width()
    {
        return 16;
    }

    static unsigned int k_unroll()
    {
        return 1;
    }

    StdTransformsFixed<operand_type, result_type, 4, 16, 1> transforms = {};

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_fp32_mla_16x4;

    hybrid_fp32_mla_16x4(const CPUInfo *ci)
    {
        if (ci->get_cpu_model() == CPUModel::A55r1) {
            kernel = a64_hybrid_fp32_mla_16x4_a55;
        }
    }
};

} // namespace arm_gemm

#endif // __aarch64__
