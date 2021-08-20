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
    unsigned int, const unsigned int *, \
    IndirectInputArg<uint8_t>, \
    size_t, size_t, \
    const uint8_t *, \
    IndirectOutputArg<uint8_t>, \
    const Requantize32 *, const int32_t *, unsigned int

namespace arm_gemm
{
// Actual kernel implementations
void a64_hybrid_u8qa_dot_4x16( ARGLIST );
void a64_hybrid_u8qa_dot_4x16_a55( ARGLIST );

class cls_a64_hybrid_u8qa_dot_4x16
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint8_t result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 4;
    }

    static unsigned int out_width()
    {
        return 16;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    StdTransformsFixed<rhs_operand_type, result_type, 4, 16, 4> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {
        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r1:
                    return { 7.5301 };
                default:
                    return { 27.5482 };
                case CPUModel::A510:
                    return { 14.81 };
                case CPUModel::V1:
                    return { 48.36 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=a64_hybrid_u8qa_dot_4x16;
    cls_a64_hybrid_u8qa_dot_4x16(const CPUInfo *ci)
    {
        switch(ci->get_cpu_model()) {
            default:
                break;
            case CPUModel::A55r1:
                kernel=a64_hybrid_u8qa_dot_4x16_a55;
                break;
        }
    }
};

} // namespace arm_gemm

#undef ARGLIST

#endif // __aarch64__
