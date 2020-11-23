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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include "../std_transforms_fixed.hpp"

#define ARGLIST  \
   unsigned int, const unsigned int *, \
   IndirectInputArg<__fp16>, \
   size_t, size_t, \
   const __fp16 *, \
   IndirectOutputArg<__fp16>, \
   const __fp16 *, Activation, bool

namespace arm_gemm
{

// Actual kernel implementations
void a64_hybrid_fp16_mla_6x32( ARGLIST );

class cls_a64_hybrid_fp16_mla_6x32
{
public:
    typedef __fp16 operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return 32;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsFixed<operand_type, result_type, 6, 32, 1> transforms = {};

    static PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        switch (ci->get_cpu_model()) {
            case CPUModel::A55r1:
                return { 5.22 };

            default:
                return { 14.53 };
        }
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_fp16_mla_6x32;

    cls_a64_hybrid_fp16_mla_6x32(const CPUInfo *)
    {
    }
};

} // namespace arm_gemm

#undef ARGLIST
#endif // __aarch64__
