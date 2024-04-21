/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "../performance_parameters.hpp"

namespace arm_gemm {

// Load the actual kernel
void a64_gemm_s8_4x4(const int8_t *, const int8_t *, int32_t *, int, int, int);

#include "arm_gemm.hpp"

class cls_a64_gemm_s8_4x4 {
public:
    typedef int8_t operand_type;
    typedef int32_t result_type;

    typedef void (*kern_type)(const int8_t *, const int8_t *, int32_t *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 4;
    }

    static unsigned int out_height() {
        return 4;
    }

    static unsigned int k_unroll() {
        return 16;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<operand_type, result_type, 4, 4, 16> transforms = {};
    StdTransformsFixed<operand_type, result_type, 4, 4, 16, true> transforms_quantized = {};

    template<typename T>
    static PerformanceParameters get_performance_parameters(const CPUInfo *ci) {
        if (std::is_same<T, int32_t>::value) {
            switch (ci->get_cpu_model()) {
                case CPUModel::A55r0:
                case CPUModel::A55r1:
                    return { 3.12, 2.93, 1.84 };
                case CPUModel::A510:
                    return { 3.32, 2.56, 2.63 };
                default:
                    return { 7.97, 3.72, 7.31 };
            }
        }

        if (std::is_same<T, int8_t>::value) {
            switch(ci->get_cpu_model()) {
                case CPUModel::A55r0:
                case CPUModel::A55r1:
                    return { 3.12, 2.18, 0.09 };
                case CPUModel::A510:
                    return { 3.33, 2.89, 0.09 };
                default:
                    return { 7.97, 3.74, 0.34 };
            }
        }
    }

    kern_type kernel=a64_gemm_s8_4x4;

    cls_a64_gemm_s8_4x4(const CPUInfo *) { }
};

} // namespace arm_gemm

#endif // __aarch64__

