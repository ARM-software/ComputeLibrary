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
#ifdef ARM_COMPUTE_ENABLE_SVE

#include "../std_transforms_sve.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const float *, const float *, size_t, \
    float *, int, size_t, int

namespace arm_gemm
{
// Actual kernel implementations
void sve_ffinterleaved_fp32_mla_8x3VL( ARGLIST );
void sve_ffinterleaved_fp32_mla_8x3VL_a64fx( ARGLIST );

class cls_sve_ffinterleaved_fp32_mla_8x3VL
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return get_vector_length<float>() * 3;
    }
    static unsigned int stripe_width()
    {
        return get_vector_length<float>();
    }

    static KernelWeightFormat kernel_weight_format()
    {
        return KernelWeightFormat::VL1VL_BL32;
    }

    static constexpr unsigned int k_unroll()
    {
        return 1;
    }


    StdTransformsSVE<operand_type, result_type, 8, 3, 1, 1> transforms = {};
    StdTransformsSVE<operand_type, result_type, 8, 3, 1, 1, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, float>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 13.51, 9.27, 3.98 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_ffinterleaved_fp32_mla_8x3VL;
    cls_sve_ffinterleaved_fp32_mla_8x3VL(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A64FX:
                kernel=sve_ffinterleaved_fp32_mla_8x3VL_a64fx;
                break;
        }
    }
};

} // namespace arm_gemm

#undef ARGLIST
#endif // ARM_COMPUTE_ENABLE_SVE
