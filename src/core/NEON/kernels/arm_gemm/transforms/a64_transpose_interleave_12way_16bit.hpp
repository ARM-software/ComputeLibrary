/*
 * Copyright (c) 2017-2018 Arm Limited.
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

// Generic unblocked transposed 6x32-bit sized specialisation
template <>
template <typename T>
inline void TransformImpl<6, 1, true, 4, 4, false>::Transform(
    T* out, const T* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  // Redirect to a 12 x uint16_t specialisation
  TransformImpl<12, 1, true, 2, 2, false>::Transform(
    reinterpret_cast<uint16_t *>(out),
    reinterpret_cast<const uint16_t *>(in),
    stride*2, x0*2, xmax*2, k0, kmax
  );
}

// Generic 12x16-bit sized specialisation
template <>
template <typename T>
inline void TransformImpl<12, 1, true, 2, 2, false>::Transform(
    T* out, const T* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  // Redirect to a uint16_t specialisation
  Transform(
    reinterpret_cast<uint16_t *>(out),
    reinterpret_cast<const uint16_t *>(in),
    stride, x0, xmax, k0, kmax
  );
}

// Specialised 12 x uint16_t version
template <>
inline void TransposeInterleaveCommon<12, uint16_t, uint16_t>::moveblock_1x1(const uint16_t *&in0, uint16_t *out) {
  __asm volatile (
    "LDR q0, [%[in0]]\n"
    "STR q0, [%[out]]\n"
    "LDR d1, [%[in0], #0x10]\n"
    "STR d1, [%[out], #0x10]\n"
    "ADD %x[in0], %x[in0], #0x18\n"
    ASM_PREFETCH("[%[in0], #192]")
    : [in0] "+r" (in0),
      [out] "+r" (out)
    :
    : "v0", "v1", "memory"
  );
}

template <>
inline void TransposeInterleaveCommon<12, uint16_t, uint16_t>::moveblock_1x2(const uint16_t *&in0, const uint16_t *&in1, uint16_t *out) {
  __asm volatile (
    "LDR q0, [%[in0]]\n"
    "LDR d1, [%[in0], #0x10]\n"
    "ADD %x[in0], %x[in0], #0x18\n"
    ASM_PREFETCH("[%[in0], #192]")

    "LDR x21, [%[in1]]\n"
    "LDR q2, [%[in1], #0x08]\n"
    "INS v1.d[1], x21\n"
    "ADD %x[in1], %x[in1], #0x18\n"
    "STP q0, q1, [%[out]]\n"
    "STR q2, [%x[out], #0x20]\n"
    ASM_PREFETCH("[%[in1], #192]")
    : [in0] "+r" (in0),
      [in1] "+r" (in1),
      [out] "+r" (out)
    :
    : "x21", "v0", "v1", "v2", "memory"
  );
}

template <>
inline void TransposeInterleaveCommon<12, uint16_t, uint16_t>::moveblock_1x4(const uint16_t *&in0, const uint16_t *&in1, const uint16_t *&in2, const uint16_t *&in3, uint16_t *out) {
  __asm __volatile (
    "LDR q0, [%x[in0]], #0x10\n"
    "STR q0, [%x[out]]\n"
    "LDR d1, [%x[in0]], #0x08\n"
    ASM_PREFETCH("[%[in0], #192]")
    "STR d1, [%x[out], #0x10]\n"

    "LDR q0, [%x[in1]], #0x10\n"
    "STR q0, [%x[out], #0x18]\n"
    "LDR d1, [%x[in1]], #0x08\n"
    ASM_PREFETCH("[%[in1], #192]")
    "STR d1, [%x[out], #0x28]\n"

    "LDR q0, [%x[in2]], #0x10\n"
    "STR q0, [%x[out], #0x30]\n"
    "LDR d1, [%x[in2]], #0x08\n"
    ASM_PREFETCH("[%[in2], #192]")
    "STR d1, [%x[out], #0x40]\n"

    "LDR q0, [%x[in3]], #0x10\n"
    "STR q0, [%x[out], #0x48]\n"
    "LDR d1, [%x[in3]], #0x08\n"
    ASM_PREFETCH("[%[in3], #192]")
    "STR d1, [%x[out], #0x58]\n"
    : [in0] "+r" (in0),
      [in1] "+r" (in1),
      [in2] "+r" (in2),
      [in3] "+r" (in3),
      [out] "+r" (out)
    :
    : "v0", "v1", "memory"
  );
}

template <>
template <>
inline void TransformImpl<12, 1, true, 2, 2, false>::Transform(
    uint16_t* out, const uint16_t* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  TransposeInterleaveCommon<12, uint16_t, uint16_t>::Transform(out, in, stride, x0, xmax, k0, kmax);
}

#endif // __aarch64__
