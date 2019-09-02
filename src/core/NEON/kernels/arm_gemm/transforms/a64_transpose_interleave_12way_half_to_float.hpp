/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS)

#include "transpose_interleave_common.hpp"

template <>
inline void TransposeInterleaveCommon<12, __fp16, float>::moveblock_1x1(const __fp16 *&in0, float *out) {
    __asm __volatile (
        "LDR    q0, [%[in0]], #16\n"
        "FCVTL2	v1.4s, v0.8h\n"
        "FCVTL	v0.4s, v0.4h\n"
        "STP    q0, q1, [%[out]]\n"
        ASM_PREFETCH("[%[in0], #192]")
        "LDR    d2, [%[in0]], #8\n"
        "FCVTL	v2.4s, v2.4h\n"
        "STR    q2, [%[out], #32]\n"
    : [in0] "+r" (in0), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "memory"
    );
}

template <>
inline void TransposeInterleaveCommon<12, __fp16, float>::moveblock_1x2(const __fp16 *&in0, const __fp16 *&in1, float *out) {
    __asm __volatile (
        "LDR    q0, [%[in0]], #16\n"
        "FCVTL2	v1.4s, v0.8h\n"
        "FCVTL	v0.4s, v0.4h\n"
        "STP    q0, q1, [%[out]]\n"
        ASM_PREFETCH("[%[in0], #192]")
        "LDR    d2, [%[in0]], #8\n"
        "FCVTL	v2.4s, v2.4h\n"
        "LDR	q3, [%[in1]], #16\n"
        "FCVTL2	v4.4s, v3.8h\n"
        "FCVTL	v3.4s, v3.4h\n"
        "STP    q2, q3, [%[out], #32]\n"
        ASM_PREFETCH("[%[in1], #192]")
        "LDR	d5, [%[in1]], #8\n"
        "FCVTL	v5.4s, v5.4h\n"
        "STP    q4, q5, [%[out], #64]\n"
    : [in0] "+r" (in0), [in1] "+r" (in1), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "memory"
    );
}

template <>
inline void TransposeInterleaveCommon<12, __fp16, float>::moveblock_1x4(const __fp16 *&in0, const __fp16 *&in1, const __fp16 *&in2, const __fp16 *&in3, float *out) {
    __asm __volatile (
        "LDR    q0, [%[in0]], #16\n"
        "FCVTL2	v1.4s, v0.8h\n"
        "FCVTL	v0.4s, v0.4h\n"
        "STP    q0, q1, [%[out]]\n"
        "LDR    d2, [%[in0]], #8\n"
        ASM_PREFETCH("[%[in0], #192]")
        "FCVTL	v2.4s, v2.4h\n"
        "LDR	q3, [%[in1]], #16\n"
        "FCVTL2	v4.4s, v3.8h\n"
        "FCVTL	v3.4s, v3.4h\n"
        "STP    q2, q3, [%[out], #32]\n"
        "LDR	d5, [%[in1]], #8\n"
        "FCVTL	v5.4s, v5.4h\n"
        ASM_PREFETCH("[%[in1], #192]")
        "STP    q4, q5, [%[out], #64]\n"
        "LDR	q6, [%[in2]], #16\n"
        "FCVTL2	v7.4s, v6.8h\n"
        "FCVTL	v6.4s, v6.4h\n"
        "STP    q6, q7, [%[out], #96]\n"
        "LDR	d8, [%[in2]], #8\n"
        "FCVTL	v8.4s, v8.4h\n"
        ASM_PREFETCH("[%[in2], #192]")
        "LDR	q9, [%[in3]], #16\n"
        "FCVTL2	v10.4s, v9.8h\n"
        "FCVTL	v9.4s, v9.4h\n"
        "STP    q8, q9, [%[out], #128]\n"
        "LDR	d11, [%[in3]], #8\n"
        "FCVTL	v11.4s, v11.4h\n"
        "STP    q10, q11, [%[out], #160]\n"
        ASM_PREFETCH("[%[in3], #192]")

    : [in0] "+r" (in0), [in1] "+r" (in1), [in2] "+r" (in2), [in3] "+r" (in3), [out] "+r" (out)
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory"
    );
}

template <>
template <>
inline void TransformImpl<12, 1, true, 4, 2, false>::Transform(
    float* out, const __fp16* const in, const int stride,
    const int x0, const int xmax, const int k0, const int kmax
) {
  TransposeInterleaveCommon<12, __fp16, float>::Transform(out, in, stride, x0, xmax, k0, kmax);
}

#endif // __aarch64__ && __ARM_FP16_ARGS
