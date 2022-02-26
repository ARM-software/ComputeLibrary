/*
 * Copyright (c) 2021 Arm Limited.
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
#include "../performance_parameters.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<uint8_t>, \
    size_t, size_t, \
    const uint8_t *, \
    IndirectOutputArg<uint32_t>, \
    const uint32_t *, Activation, bool

namespace arm_gemm
{
// Actual kernel implementations
void sve_hybrid_u8u32_mmla_6x4VL( ARGLIST );

class cls_sve_hybrid_u8u32_mmla_6x4VL
{
public:
    typedef uint8_t lhs_operand_type;
    typedef uint8_t rhs_operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)( ARGLIST );

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 6;
    }

    static unsigned int out_width()
    {
        return get_vector_length<uint32_t>() * 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 8;
    }

    static constexpr bool supports_accumulate()
    {
        return true;
    }

    StdTransformsSVE<rhs_operand_type, result_type, 6, 8, 8> transforms = {};
    template<typename T>
    static inline PerformanceParameters get_performance_parameters(const CPUInfo *ci)
    {

        if (std::is_same<T, uint32_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 54.45 };
                case CPUModel::A510:
                    return { 24.22 };
                case CPUModel::V1:
                    return { 105.16 };
            }
        }


        if (std::is_same<T, uint8_t>::value) {
            switch (ci->get_cpu_model()) {
                default:
                    return { 54.90, 15.69, 0.62 };
                case CPUModel::A510:
                    return { 26.80, 3.89, 0.47 };
                case CPUModel::V1:
                    return { 75.14, 15.87, 0.83 };
            }
        }

        return { 1.0 };
    }

    // Default to the generic kernel
    kern_type kernel=sve_hybrid_u8u32_mmla_6x4VL;
    cls_sve_hybrid_u8u32_mmla_6x4VL(const CPUInfo *)
    {
    }
};

} // namespace arm_gemm

#undef ARGLIST

#endif // ARM_COMPUTE_ENABLE_SVE
