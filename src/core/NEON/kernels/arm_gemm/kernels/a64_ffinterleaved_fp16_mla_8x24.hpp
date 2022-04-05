/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef __aarch64__

#include "../std_transforms_fixed.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const __fp16 *, const __fp16 *, size_t, \
    __fp16 *, int, size_t, int

namespace arm_gemm
{
// Actual kernel implementations
void a64_ffinterleaved_fp16_mla_8x24( ARGLIST );

class cls_a64_ffinterleaved_fp16_mla_8x24
{
public:
    typedef __fp16 operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 24;
    }
    static unsigned int stripe_width()
    {
        return 8;
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL128_BL16;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }


    StdTransformsFixed<operand_type, result_type, 8, 24, 1> transforms = {};
    StdTransformsFixed<operand_type, result_type, 8, 24, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, __fp16>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 22.87, 7.77, 2.03 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_ffinterleaved_fp16_mla_8x24;
    cls_a64_ffinterleaved_fp16_mla_8x24(const CPUInfo *)
    {
    }
};

} // namespace arm_gemm

#undef ARGLIST
#endif // __aarch64__
