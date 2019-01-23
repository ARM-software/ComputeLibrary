/*
 * Copyright (c) 2017-2019 Arm Limited.
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

#include <cstddef>

// Macro for unreachable code (e.g. impossible default cases on switch)
#define UNREACHABLE(why)  __builtin_unreachable()

// Paranoid option for the above with assert
// #define UNREACHABLE(why)   assert(0 && why)

template<typename T>
inline T iceildiv(const T a, const T b) {
    return (a + b - 1) / b;
}

template <typename T>
inline T roundup(const T a, const T b) {
    T rem = a % b;

    if (rem) {
        return a + b - rem;
    } else {
        return a;
    }
}

namespace arm_gemm {
namespace utils {
namespace {

#ifdef __ARM_FEATURE_SVE
template<size_t sz>
inline unsigned long get_vector_length_sz() {
    unsigned long v;

    __asm (
        "cntb	%0"
        : "=r" (v)
    );

    return v / sz;
}

#define VEC_LEN_SPEC(sz, opcode) template <> inline unsigned long get_vector_length_sz<sz>() { unsigned long v; __asm ( opcode " %0" : "=r" (v)); return v; }

VEC_LEN_SPEC(8, "cntd")
VEC_LEN_SPEC(4, "cntw")
VEC_LEN_SPEC(2, "cnth")
VEC_LEN_SPEC(1, "cntb")
#endif

} // anonymous namespace

template <typename T>
inline unsigned long get_vector_length() {
#ifdef __ARM_FEATURE_SVE
    return get_vector_length_sz<sizeof(T)>();
#else
    return 16 / sizeof(T);
#endif
}

} // utils namespace
} // arm_gemm namespace

using namespace arm_gemm::utils;