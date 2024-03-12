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

#ifdef __arm__

#include "transpose_interleave_common.hpp"

// Generic unblocked transposed 8x32-bit sized specialisation
template <>
template <typename T>
void TransformImpl<8, 1, true, 4, 4, VLType::None>::Transform(
    T* out, const T* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  // Redirect to a 16x uint16_t specialisation
  TransformImpl<16, 1, true, 2, 2, VLType::None>::Transform(
    reinterpret_cast<uint16_t *>(out),
    reinterpret_cast<const uint16_t *>(in),
    stride*2, x0*2, xmax*2, k0, kmax
  );
}

// Generic 16x16-bit sized specialisation
template <>
template <typename T>
void TransformImpl<16, 1, true, 2, 2, VLType::None>::Transform(
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

// Specialised 16 x uint16_t version
template <>
void TransposeInterleaveCommon<16, uint16_t, uint16_t>::moveblock_1x1(const uint16_t *&in0, uint16_t *out) {
  __asm volatile (
    "VLD1.32	{d0-d3}, [%[in0]]!\n"
    "VST1.32	{d0-d3}, [%[out]]\n"
    ASM_PREFETCH("[%[in0], #192]")
    : [in0] "+r" (in0),
      [out] "+r" (out)
    :
    : "q0", "q1", "memory"
  );
}

template <>
void TransposeInterleaveCommon<16, uint16_t, uint16_t>::moveblock_1x2(const uint16_t *&in0, const uint16_t *&in1, uint16_t *out) {
  __asm volatile (
    "VLD1.32	{d0-d3}, [%[in0]]!\n"
    "VST1.32	{d0-d3}, [%[out]]!\n"
    ASM_PREFETCH("[%[in0], #192]")
    "VLD1.32	{d0-d3}, [%[in1]]!\n"
    "VST1.32	{d0-d3}, [%[out]]\n"
    ASM_PREFETCH("[%[in1], #192]")
    "SUB	%[out], %[out], #32\n"
    : [in0] "+r" (in0),
      [in1] "+r" (in1),
      [out] "+r" (out)
    :
    : "q0", "q1", "memory"
  );
}

template <>
void TransposeInterleaveCommon<16, uint16_t, uint16_t>::moveblock_1x4(const uint16_t *&in0, const uint16_t *&in1, const uint16_t *&in2, const uint16_t *&in3, uint16_t *out) {
  __asm __volatile (
    "VLD1.32	{d0-d3}, [%[in0]]!\n"
    "VST1.32	{d0-d3}, [%[out]]!\n"
    ASM_PREFETCH("[%[in0], #192]")
    "VLD1.32	{d0-d3}, [%[in1]]!\n"
    "VST1.32	{d0-d3}, [%[out]]!\n"
    ASM_PREFETCH("[%[in1], #192]")
    "VLD1.32	{d0-d3}, [%[in2]]!\n"
    "VST1.32	{d0-d3}, [%[out]]!\n"
    ASM_PREFETCH("[%[in2], #192]")
    "VLD1.32	{d0-d3}, [%[in3]]!\n"
    "VST1.32	{d0-d3}, [%[out]]\n"
    ASM_PREFETCH("[%[in3], #192]")
    "SUB	%[out], %[out], #96\n"
    : [in0] "+r" (in0),
      [in1] "+r" (in1),
      [in2] "+r" (in2),
      [in3] "+r" (in3),
      [out] "+r" (out)
    :
    : "q0", "q1", "memory"
  );
}

template <>
template <>
void TransformImpl<16, 1, true, 2, 2, VLType::None>::Transform(
    uint16_t* out, const uint16_t* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  TransposeInterleaveCommon<16, uint16_t, uint16_t>::Transform(out, in, stride, x0, xmax, k0, kmax);
}

#endif // __arm__
