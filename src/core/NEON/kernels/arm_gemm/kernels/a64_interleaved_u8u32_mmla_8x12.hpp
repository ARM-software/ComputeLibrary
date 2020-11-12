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

#ifdef __aarch64__

#include <cstdint>
#include "../std_transforms_fixed.hpp"

namespace arm_gemm {

// Actual kernel implementations
void a64_interleaved_u8u32_mmla_8x12(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);

class cls_a64_interleaved_u8u32_mmla_8x12 {
public:
    typedef uint8_t operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);

    /* Kernel blocking parameters */
    static unsigned int out_width()
    {
        return 12;
    }

    static unsigned int out_height()
    {
        return 8;
    }

    static unsigned int k_unroll()
    {
        return 8;
    }

    // Use the standard fixed size transforms.
    StdTransformsFixed<operand_type, result_type, 8, 12, 8> transforms = {};
    StdTransformsFixed<operand_type, result_type, 8, 12, 8, true> transforms_quantized = {};

    kern_type kernel=a64_interleaved_u8u32_mmla_8x12;

    cls_a64_interleaved_u8u32_mmla_8x12(const CPUInfo *)
    {

    }
};

} // namespace arm_gemm

#endif // __aarch64__
