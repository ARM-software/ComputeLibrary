/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "../performance_parameters.hpp"

#define ARGLIST  \
    const int8_t *, const int8_t *, \
    int32_t *, int, int, int

namespace arm_gemm
{
// Actual kernel implementations
void a64_interleaved_s8s32_mmla_8x12( ARGLIST );
void a64_interleaved_s8s32_mmla_8x12_a510( ARGLIST );

class cls_a64_interleaved_s8s32_mmla_8x12
{
public:
    typedef int8_t operand_type;
    typedef int32_t result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 12;
    }

    static unsigned int stripe_width()
    {
        return 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 8;
    }


    StdTransformsFixed<operand_type, result_type, 8, 12, 8> transforms = {};
    StdTransformsFixed<operand_type, result_type, 8, 12, 8, true> transforms_quantized = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.57, 4.08, 8.01 };
                case CPUModel::A510:
                    return { 48.25, 3.53, 3.71 };
                case CPUModel::V1:
                    return { 117.02, 4.98, 10.87 };
            }
        }


        if (std::is_same<T, int8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 62.53, 3.70, 0.50 };
                case CPUModel::A510:
                    return { 48.22, 2.49, 0.29 };
                case CPUModel::V1:
                    return { 75.54, 8.06, 0.63 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_interleaved_s8s32_mmla_8x12;
    cls_a64_interleaved_s8s32_mmla_8x12(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A510:
                kernel=a64_interleaved_s8s32_mmla_8x12_a510;
                break;
        }
    }

};

} // namespace arm_gemm

#undef ARGLIST

#endif // __aarch64__
