/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

namespace arm_gemm {

// Actual kernel implementations
void a64_hgemm_asimd_8x24(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);
void a64_hgemm_asimd_8x24_a55r1(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);
void a64_hgemm_asimd_8x24_x1(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);

// 8x24 HGEMM "strategy" class.  Describes the kernel properties.
//
// The generic "gemm_opt" function will instantiate one of these (allowing
// the constructor to pick a kernel implementation).
class cls_a64_hgemm_8x24 {
public:
    typedef __fp16 operand_type;
    typedef __fp16 result_type;

    typedef void (*kern_type)(const __fp16 *, const __fp16 *, __fp16 *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width() {
        return 24;
    }

    static unsigned int out_height() {
        return 8;
    }

    static unsigned int k_unroll() {
        return 1;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<operand_type, result_type, 8, 24> transforms = {};

    template<typename T>
    static PerformanceParameters get_performance_parameters(const CPUInfo *ci) {
        switch (ci->get_cpu_model()) {
            case CPUModel::A55r1:
                return {  7.16, 1.14, 0.67  };

            default:
                return { 12.67, 3.98, 1.16  };
        }
    }

    // Default to the generic kernel
    kern_type kernel = a64_hgemm_asimd_8x24;

    cls_a64_hgemm_8x24(const CPUInfo *ci) {
        auto model = ci->get_cpu_model();

        if (model == CPUModel::A55r1) {
            kernel = a64_hgemm_asimd_8x24_a55r1;
        } else if (model == CPUModel::X1) {
            kernel = a64_hgemm_asimd_8x24_x1;
        }
    }
};

} // namespace arm_gemm

#endif // __aarch64__ && (FP16_KERNELS || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
