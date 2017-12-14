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

#include "transpose_interleave_common.hpp"

// Generic unblocked transposed 12x32-bit sized specialisation
template <>
template <typename T>
inline void TransformImpl<12, 1, true, 4, 4>::Transform(
    T* out, const T* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  // Redirect to a 24 x uint16_t specialisation
  TransformImpl<24, 1, true, 2, 2>::Transform(
    reinterpret_cast<uint16_t *>(out),
    reinterpret_cast<const uint16_t * const>(in),
    stride*2, x0*2, xmax*2, k0, kmax
  );
}

// Generic 24x16-bit sized specialisation
template <>
template <typename T>
inline void TransformImpl<24, 1, true, 2, 2>::Transform(
    T* out, const T* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  // Redirect to a uint16_t specialisation
  Transform(
    reinterpret_cast<uint16_t *>(out),
    reinterpret_cast<const uint16_t * const>(in),
    stride, x0, xmax, k0, kmax
  );
}

// Specialised 24 x uint16_t version
template <>
inline void TransposeInterleaveCommon<24, uint16_t, uint16_t>::moveblock_1x1(const uint16_t *&in0, uint16_t *out) {
    __asm __volatile (
        "LDP    q0, q1, [%[in0]], #32\n"
        "STP    q0, q1, [%[out]]\n"
        ASM_PREFETCH("[%[in0], #192]")
        "LDR    q2, [%[in0]], #16\n"
        "STR    q2, [%[out], #32]\n"
    : [in0] "+r" (in0), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "memory"
    );
}

template <>
inline void TransposeInterleaveCommon<24, uint16_t, uint16_t>::moveblock_1x2(const uint16_t *&in0, const uint16_t *&in1,uint16_t *out) {
    __asm __volatile (
        "LDP    q0, q1, [%[in0]], #32\n"
        "STP    q0, q1, [%[out]]\n"
        ASM_PREFETCH("[%[in0], #192]")
        "LDR    q2, [%[in0]], #16\n"
        "LDP	q3, q4, [%[in1]], #32\n"
        "STP    q2, q3, [%[out], #32]\n"
        ASM_PREFETCH("[%[in1], #192]")
        "LDR	q5, [%[in1]], #16\n"
        "STP    q4, q5, [%[out], #64]\n"
    : [in0] "+r" (in0), [in1] "+r" (in1), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "memory"
    );
}

template <>
inline void TransposeInterleaveCommon<24, uint16_t, uint16_t>::moveblock_1x4(const uint16_t *&in0, const uint16_t *&in1, const uint16_t *&in2, const uint16_t *&in3, uint16_t *out) {
    __asm __volatile (
        "LDP    q0, q1, [%[in0]], #32\n"
        "STP    q0, q1, [%[out]]\n"
        "LDR    q2, [%[in0]], #16\n"
        ASM_PREFETCH("[%[in0], #192]")
        "LDP	q3, q4, [%[in1]], #32\n"
        "STP    q2, q3, [%[out], #32]\n"
        "LDR	q5, [%[in1]], #16\n"
        ASM_PREFETCH("[%[in1], #192]")
        "STP    q4, q5, [%[out], #64]\n"
        "LDP	q6, q7, [%[in2]], #32\n"
        "STP    q6, q7, [%[out], #96]\n"
        "LDR	q8, [%[in2]], #16\n"
        ASM_PREFETCH("[%[in2], #192]")
        "LDP	q9, q10, [%[in3]], #32\n"
        "STP    q8, q9, [%[out], #128]\n"
        "LDR	q11, [%[in3]], #16\n"
        "STP    q10, q11, [%[out], #160]\n"
        ASM_PREFETCH("[%[in3], #192]")

    : [in0] "+r" (in0), [in1] "+r" (in1), [in2] "+r" (in2), [in3] "+r" (in3), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory"
    );
}

template <>
template <>
inline void TransformImpl<24, 1, true, 2, 2>::Transform(
    uint16_t* out, const uint16_t* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  TransposeInterleaveCommon<24, uint16_t, uint16_t>::Transform(out, in, stride, x0, xmax, k0, kmax);
}

#endif  // __arch64__
