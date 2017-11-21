/*
 * Copyright (c) 2017 ARM Limited.
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

// Load the actual kernel
#include "a64_gemm_u8_4x4/generic.hpp"

class gemm_u8_4x4 {
public:
    typedef uint8_t operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint8_t *, const uint8_t *, uint32_t *, int, int, int);

    /* Describes the data layout for A input */
    static const int A_interleave = 4;
    static const int A_block = 16;
    static const bool A_transpose = false;

    /* Same for B input */
    static const int B_interleave = 4;
    static const int B_block = 16;
    static const bool B_transpose = true;

    /* Kernel blocking parameters */
    static const int out_width = 4;
    static const int out_height = 4;
    static const int k_unroll = 16;

    kern_type kernel = nullptr;

    gemm_u8_4x4(const CPUInfo *ci) {
        kernel = a64_gemm_u8_4x4;
    }
};

#endif // __aarch64__

