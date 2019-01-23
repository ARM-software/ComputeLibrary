/*
 * Copyright (c) 2017-2019 Arm Limited.
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

#ifdef __arm__

#include "../std_transforms_fixed.hpp"

namespace arm_gemm {

// Actual kernel implementations
void a32_sgemm_8x6(const float *, const float *, float *, int, int, int);
void a32_sgemm_8x6_a53(const float *, const float *, float *, int, int, int);
void a32_sgemm_8x6_a55r1(const float *, const float *, float *, int, int, int);

// 8x6 SGEMM "strategy" class.
//
// This describes the characteristics of a family of kernels, in terms of
// the required interleave properties and the output block size.
//
// All kernels in the family must share these characteristics.  The actual
// kernel to be used can be chosen at runtime, based on the CPU_type
// structure.
class sgemm_8x6 {
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 8;
    }

    static unsigned int out_height() {
        return 6;
    }

    static unsigned int k_unroll() {
        return 1;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<operand_type, result_type, 6, 8> transforms = {};

    kern_type kernel = a32_sgemm_8x6;

    sgemm_8x6(const CPUInfo *ci) {
        switch(ci->get_cpu_model()) {
            case CPUModel::A53:
                kernel = a32_sgemm_8x6_a53;
                break;

            case CPUModel::A55r1:
                kernel = a32_sgemm_8x6_a55r1;
                break;

            default:
                /* Generic kernel is selected by default. */
                break;
        }
    }
};

} // namespace arm_gemm
#endif // __arm__
