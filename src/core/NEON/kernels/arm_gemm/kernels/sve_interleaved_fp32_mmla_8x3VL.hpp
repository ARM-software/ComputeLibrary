/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#ifdef ARM_COMPUTE_ENABLE_SVE


#include "../std_transforms_sve.hpp"

namespace arm_gemm {

// Actual kernel implementations
void sve_interleaved_fp32_mmla_8x3VL(const float *, const float *, float *, int, int, int);

class cls_sve_interleaved_fp32_mmla_8x3VL {
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, const float *, float *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width()
    {
        return get_vector_length<float>() * 3;
    }

    static unsigned int out_height()
    {
        return 8;
    }

    static unsigned int k_unroll()
    {
        return 2;
    }

    // Use the standard fixed size transforms.
    StdTransformsSVE<operand_type, result_type, 8, 6, 2, 2> transforms = {};

    kern_type kernel=sve_interleaved_fp32_mmla_8x3VL;

    cls_sve_interleaved_fp32_mmla_8x3VL(const CPUInfo *)
    {

    }
};

} // namespace arm_gemm

#endif // ARM_COMPUTE_ENABLE_SVE
