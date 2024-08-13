/*
 * Copyright (c) 2021-2024 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

#include "arm_gemm.hpp"
#include <cstdint>

namespace arm_conv {
namespace depthwise {

void sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst_impl(const unsigned int n_channels, const int8_t *const *const inptrs, const int8_t *params, const int32_t *, const arm_gemm::Requantize32& qp, const int32_t *, const int32_t *, int8_t *const *const outptrs)
{
  __asm__ __volatile__(
    "mov x14, #0x0\n"
    "ldp x27, x26, [%x[inptrs], #0x0]\n"
    "ldp x25, x24, [%x[inptrs], #0x10]\n"
    "ptrue p2.b\n"
    "ldp x23, x22, [%x[inptrs], #0x20]\n"
    "ldp x21, x20, [%x[inptrs], #0x30]\n"
    "mov x13, #0x0\n"
    "ldp x12, x11, [%x[outptrs], #0x0]\n"
    "ldp x10, x9, [%x[outptrs], #0x10]\n"
    "whilelt p0.b, x14, %x[n_channels]\n"
    "ld1rw { z11.s }, p2/Z, [%x[qp], %[offsetof_Requantize32_minval]]\n"
    "ld1rw { z14.s }, p2/Z, [%x[qp], %[offsetof_Requantize32_maxval]]\n"
    "ld1rw { z30.s }, p2/Z, [%x[qp], %[offsetof_Requantize32_c_offset]]\n"
    "ld1b { z3.b }, p2/Z, [%x[params], #1, MUL VL]\n"
    "ld1b { z12.b }, p0/Z, [x27, x14]\n"
    "ld1b { z17.b }, p0/Z, [x26, x14]\n"
    "ldp x27, x26, [%x[inptrs], #0x40]\n"
    "ld1b { z16.b }, p0/Z, [x25, x14]\n"
    "ld1b { z15.b }, p0/Z, [x24, x14]\n"
    "ldp x25, x24, [%x[inptrs], #0x50]\n"
    "ld1b { z10.b }, p0/Z, [x23, x14]\n"
    "ld1b { z24.b }, p0/Z, [x22, x14]\n"
    "ldp x23, x22, [%x[inptrs], #0x60]\n"
    "ld1b { z19.b }, p0/Z, [x21, x14]\n"
    "zip2 z18.b, z12.b, z16.b\n"
    "zip1 z12.b, z12.b, z16.b\n"
    "ld1b { z8.b }, p0/Z, [x20, x14]\n"
    "ldp x21, x20, [%x[inptrs], #0x70]\n"
    "zip1 z16.b, z17.b, z15.b\n"
    "zip2 z15.b, z17.b, z15.b\n"
    "ld1b { z2.b }, p0/Z, [x27, x14]\n"
    "ld1b { z23.b }, p0/Z, [x26, x14]\n"
    "ld1b { z17.b }, p0/Z, [x25, x14]\n"
    "ld1b { z7.b }, p0/Z, [x24, x14]\n"
    "zip2 z22.b, z10.b, z19.b\n"
    "zip1 z10.b, z10.b, z19.b\n"
    "ld1b { z4.b }, p0/Z, [x23, x14]\n"
    "ld1b { z21.b }, p0/Z, [x22, x14]\n"
    "zip2 z5.b, z12.b, z16.b\n"
    "zip1 z12.b, z12.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x21, x14]\n"
    "ld1b { z6.b }, p0/Z, [x20, x14]\n"
    "zip1 z9.b, z18.b, z15.b\n"
    "zip2 z15.b, z18.b, z15.b\n"
    "zip1 z20.b, z24.b, z8.b\n"
    "zip2 z8.b, z24.b, z8.b\n"
    "ld1w { z13.s }, p2/Z, [%x[params]]\n"
    "ldp x28, x27, [%x[inptrs], #0x0]\n"
    "zip2 z19.b, z2.b, z17.b\n"
    "zip1 z2.b, z2.b, z17.b\n"
    "ldp x26, x25, [%x[inptrs], #0x10]\n"
    "ldp x24, x22, [%x[inptrs], #0x20]\n"
    "zip1 z18.b, z23.b, z7.b\n"
    "zip2 z7.b, z23.b, z7.b\n"
    "ldp x21, x20, [%x[inptrs], #0x30]\n"
    "ld1b { z0.b }, p2/Z, [%x[params], #2, MUL VL]\n"
    "zip2 z17.b, z4.b, z16.b\n"
    "zip1 z4.b, z4.b, z16.b\n"
    "ld1b { z1.b }, p2/Z, [%x[params], #3, MUL VL]\n"
    "addvl %x[params], %x[params], #4\n"
    "zip1 z16.b, z21.b, z6.b\n"
    "zip2 z6.b, z21.b, z6.b\n"
    "zip2 z31.b, z10.b, z20.b\n"
    "zip1 z10.b, z10.b, z20.b\n"
    "zip1 z26.b, z22.b, z8.b\n"
    "zip2 z8.b, z22.b, z8.b\n"
    "zip2 z25.b, z2.b, z18.b\n"
    "zip1 z2.b, z2.b, z18.b\n"
    "zip1 z28.b, z19.b, z7.b\n"
    "zip2 z7.b, z19.b, z7.b\n"
    "zip2 z27.b, z4.b, z16.b\n"
    "zip1 z4.b, z4.b, z16.b\n"
    "zip1 z29.b, z17.b, z6.b\n"
    "zip2 z6.b, z17.b, z6.b\n"
    "mov z21.d, z13.d\n"
    "mov z20.d, z13.d\n"
    "mov z23.d, z13.d\n"
    "1:"  // Loop
    "sdot z13.s, z3.b, z12.b\n"
    "sdot z20.s, z3.b, z10.b\n"
    "ext z12.b, z12.b, z12.b, #0x1\n"
    "whilelt p0.s, x13, %x[n_channels]\n"
    "incw x14, ALL, MUL #4\n"
    "sdot z21.s, z3.b, z12.b\n"
    "ld1w { z17.s }, p2/Z, [%x[params]]\n"
    "sdot z13.s, z0.b, z10.b\n"
    "ext z10.b, z10.b, z10.b, #0x1\n"
    "sdot z20.s, z0.b, z2.b\n"
    "sdot z23.s, z3.b, z10.b\n"
    "sdot z13.s, z1.b, z2.b\n"
    "ext z2.b, z2.b, z2.b, #0x1\n"
    "sdot z21.s, z0.b, z10.b\n"
    "ld1w { z22.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "sdot z23.s, z0.b, z2.b\n"
    "sdot z20.s, z1.b, z4.b\n"
    "ext z4.b, z4.b, z4.b, #0x1\n"
    ".inst 0x04b175ad  // sqrdmulh z13.s, z13.s, z17.s\n"
    "sdot z21.s, z1.b, z2.b\n"
    "sdot z23.s, z1.b, z4.b\n"
    "and z16.d, z13.d, z22.d\n"
    ".inst 0x04b17694  // sqrdmulh z20.s, z20.s, z17.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x04b176b5  // sqrdmulh z21.s, z21.s, z17.s\n"
    ".inst 0x04b176f7  // sqrdmulh z23.s, z23.s, z17.s\n"
    "ld1w { z19.s }, p2/Z, [%x[params], #6, MUL VL]\n"
    "and z18.d, z20.d, z22.d\n"
    "sqadd z13.s, z13.s, z16.s\n"
    "and z17.d, z21.d, z22.d\n"
    "and z16.d, z23.d, z22.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44828acd  // srshl z13.s, p2/M, z13.s, z22.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z20.s, z20.s, z18.s\n"
    "ld1b { z18.b }, p2/Z, [%x[params], #5, MUL VL]\n"
    "sqadd z21.s, z21.s, z17.s\n"
    "ld1b { z17.b }, p2/Z, [%x[params], #4, MUL VL]\n"
    "sqadd z23.s, z23.s, z16.s\n"
    "ld1b { z16.b }, p2/Z, [%x[params], #3, MUL VL]\n"
    "add z13.s, z13.s, z30.s\n"
    ".inst 0x44828ad5  // srshl z21.s, p2/M, z21.s, z22.s\n"
    ".inst 0x44828ad4  // srshl z20.s, p2/M, z20.s, z22.s\n"
    ".inst 0x44828ad7  // srshl z23.s, p2/M, z23.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [%x[params], #7, MUL VL]\n"
    "smax z13.s, p2/M, z13.s, z11.s\n"
    "add z21.s, z21.s, z30.s\n"
    "add z20.s, z20.s, z30.s\n"
    "add z23.s, z23.s, z30.s\n"
    "smin z13.s, p2/M, z13.s, z14.s\n"
    "smax z21.s, p2/M, z21.s, z11.s\n"
    "smax z20.s, p2/M, z20.s, z11.s\n"
    "smax z23.s, p2/M, z23.s, z11.s\n"
    "st1b { z13.s }, p0, [x12, x13]\n"
    "ld1w { z24.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "addvl %x[params], %x[params], #16\n"
    "smin z21.s, p2/M, z21.s, z14.s\n"
    "smin z20.s, p2/M, z20.s, z14.s\n"
    "smin z23.s, p2/M, z23.s, z14.s\n"
    "st1b { z21.s }, p0, [x11, x13]\n"
    "mov z13.d, z24.d\n"
    "st1b { z20.s }, p0, [x10, x13]\n"
    "mov z21.d, z24.d\n"
    "st1b { z23.s }, p0, [x9, x13]\n"
    "mov z20.d, z24.d\n"
    "sdot z24.s, z16.b, z5.b\n"
    "incw x13\n"
    "sdot z21.s, z16.b, z31.b\n"
    "ext z5.b, z5.b, z5.b, #0x1\n"
    "whilelt p0.s, x13, %x[n_channels]\n"
    "sdot z24.s, z17.b, z31.b\n"
    "ext z31.b, z31.b, z31.b, #0x1\n"
    "sdot z13.s, z16.b, z5.b\n"
    "sdot z20.s, z16.b, z31.b\n"
    "sdot z21.s, z17.b, z25.b\n"
    "sdot z24.s, z18.b, z25.b\n"
    "ext z25.b, z25.b, z25.b, #0x1\n"
    "sdot z13.s, z17.b, z31.b\n"
    "sdot z20.s, z17.b, z25.b\n"
    "sdot z21.s, z18.b, z27.b\n"
    "ext z27.b, z27.b, z27.b, #0x1\n"
    ".inst 0x04b37718  // sqrdmulh z24.s, z24.s, z19.s\n"
    "sdot z13.s, z18.b, z25.b\n"
    "sdot z20.s, z18.b, z27.b\n"
    "and z16.d, z24.d, z22.d\n"
    ".inst 0x04b376b5  // sqrdmulh z21.s, z21.s, z19.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x04b375ad  // sqrdmulh z13.s, z13.s, z19.s\n"
    ".inst 0x04b37694  // sqrdmulh z20.s, z20.s, z19.s\n"
    "ld1w { z19.s }, p2/Z, [%x[params], #-4, MUL VL]\n"
    "and z18.d, z21.d, z22.d\n"
    "sqadd z24.s, z24.s, z16.s\n"
    "and z17.d, z13.d, z22.d\n"
    "and z16.d, z20.d, z22.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44828ad8  // srshl z24.s, p2/M, z24.s, z22.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z21.s, z21.s, z18.s\n"
    "ld1b { z18.b }, p2/Z, [%x[params], #-5, MUL VL]\n"
    "sqadd z13.s, z13.s, z17.s\n"
    "ld1b { z17.b }, p2/Z, [%x[params], #-6, MUL VL]\n"
    "sqadd z20.s, z20.s, z16.s\n"
    "ld1b { z16.b }, p2/Z, [%x[params], #-7, MUL VL]\n"
    "add z24.s, z24.s, z30.s\n"
    ".inst 0x44828acd  // srshl z13.s, p2/M, z13.s, z22.s\n"
    ".inst 0x44828ad5  // srshl z21.s, p2/M, z21.s, z22.s\n"
    ".inst 0x44828ad4  // srshl z20.s, p2/M, z20.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [%x[params], #-3, MUL VL]\n"
    "smax z24.s, p2/M, z24.s, z11.s\n"
    "add z13.s, z13.s, z30.s\n"
    "add z21.s, z21.s, z30.s\n"
    "add z20.s, z20.s, z30.s\n"
    "smin z24.s, p2/M, z24.s, z14.s\n"
    "smax z13.s, p2/M, z13.s, z11.s\n"
    "smax z21.s, p2/M, z21.s, z11.s\n"
    "smax z20.s, p2/M, z20.s, z11.s\n"
    "st1b { z24.s }, p0, [x12, x13]\n"
    "ld1w { z24.s }, p2/Z, [%x[params], #-8, MUL VL]\n"
    "smin z13.s, p2/M, z13.s, z14.s\n"
    "smin z21.s, p2/M, z21.s, z14.s\n"
    "smin z20.s, p2/M, z20.s, z14.s\n"
    "st1b { z13.s }, p0, [x11, x13]\n"
    "mov z23.d, z24.d\n"
    "st1b { z21.s }, p0, [x10, x13]\n"
    "mov z21.d, z24.d\n"
    "st1b { z20.s }, p0, [x9, x13]\n"
    "mov z20.d, z24.d\n"
    "sdot z24.s, z16.b, z9.b\n"
    "incw x13\n"
    "sdot z21.s, z16.b, z26.b\n"
    "ext z9.b, z9.b, z9.b, #0x1\n"
    "whilelt p0.s, x13, %x[n_channels]\n"
    "sdot z24.s, z17.b, z26.b\n"
    "ext z26.b, z26.b, z26.b, #0x1\n"
    "sdot z23.s, z16.b, z9.b\n"
    "sdot z20.s, z16.b, z26.b\n"
    "sdot z21.s, z17.b, z28.b\n"
    "sdot z24.s, z18.b, z28.b\n"
    "ext z28.b, z28.b, z28.b, #0x1\n"
    "sdot z23.s, z17.b, z26.b\n"
    "sdot z20.s, z17.b, z28.b\n"
    "sdot z21.s, z18.b, z29.b\n"
    "ext z29.b, z29.b, z29.b, #0x1\n"
    ".inst 0x04b37718  // sqrdmulh z24.s, z24.s, z19.s\n"
    "sdot z23.s, z18.b, z28.b\n"
    "sdot z20.s, z18.b, z29.b\n"
    "and z16.d, z24.d, z22.d\n"
    ".inst 0x04b376b5  // sqrdmulh z21.s, z21.s, z19.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x04b376f7  // sqrdmulh z23.s, z23.s, z19.s\n"
    ".inst 0x04b37694  // sqrdmulh z20.s, z20.s, z19.s\n"
    "ld1w { z19.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "and z18.d, z21.d, z22.d\n"
    "sqadd z24.s, z24.s, z16.s\n"
    "and z17.d, z23.d, z22.d\n"
    "and z16.d, z20.d, z22.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44828ad8  // srshl z24.s, p2/M, z24.s, z22.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z21.s, z21.s, z18.s\n"
    "ld1b { z18.b }, p2/Z, [%x[params], #1, MUL VL]\n"
    "sqadd z23.s, z23.s, z17.s\n"
    "ld1b { z17.b }, p2/Z, [%x[params]]\n"
    "sqadd z20.s, z20.s, z16.s\n"
    "ld1b { z16.b }, p2/Z, [%x[params], #-1, MUL VL]\n"
    "add z24.s, z24.s, z30.s\n"
    ".inst 0x44828ad7  // srshl z23.s, p2/M, z23.s, z22.s\n"
    ".inst 0x44828ad5  // srshl z21.s, p2/M, z21.s, z22.s\n"
    ".inst 0x44828ad4  // srshl z20.s, p2/M, z20.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "smax z24.s, p2/M, z24.s, z11.s\n"
    "add z23.s, z23.s, z30.s\n"
    "add z21.s, z21.s, z30.s\n"
    "add z20.s, z20.s, z30.s\n"
    "smin z24.s, p2/M, z24.s, z14.s\n"
    "smax z23.s, p2/M, z23.s, z11.s\n"
    "smax z21.s, p2/M, z21.s, z11.s\n"
    "smax z20.s, p2/M, z20.s, z11.s\n"
    "st1b { z24.s }, p0, [x12, x13]\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #-2, MUL VL]\n"
    "smin z23.s, p2/M, z23.s, z14.s\n"
    "smin z21.s, p2/M, z21.s, z14.s\n"
    "smin z20.s, p2/M, z20.s, z14.s\n"
    "st1b { z23.s }, p0, [x11, x13]\n"
    "mov z29.d, z13.d\n"
    "st1b { z21.s }, p0, [x10, x13]\n"
    "mov z28.d, z13.d\n"
    "st1b { z20.s }, p0, [x9, x13]\n"
    "mov z27.d, z13.d\n"
    "sdot z13.s, z16.b, z15.b\n"
    "incw x13\n"
    "sdot z28.s, z16.b, z8.b\n"
    "ext z15.b, z15.b, z15.b, #0x1\n"
    "whilelt p1.s, x13, %x[n_channels]\n"
    "whilelt p0.b, x14, %x[n_channels]\n"
    "sdot z13.s, z17.b, z8.b\n"
    "ext z8.b, z8.b, z8.b, #0x1\n"
    "sdot z29.s, z16.b, z15.b\n"
    "ld1b { z26.b }, p0/Z, [x27, x14]\n"
    "ld1b { z21.b }, p0/Z, [x26, x14]\n"
    "ld1b { z15.b }, p0/Z, [x25, x14]\n"
    "ld1b { z25.b }, p0/Z, [x22, x14]\n"
    "ld1b { z20.b }, p0/Z, [x21, x14]\n"
    "sdot z27.s, z16.b, z8.b\n"
    "sdot z28.s, z17.b, z7.b\n"
    "sdot z13.s, z18.b, z7.b\n"
    "ext z7.b, z7.b, z7.b, #0x1\n"
    "sdot z29.s, z17.b, z8.b\n"
    "ld1b { z8.b }, p0/Z, [x20, x14]\n"
    "sdot z27.s, z17.b, z7.b\n"
    "sdot z28.s, z18.b, z6.b\n"
    "ext z6.b, z6.b, z6.b, #0x1\n"
    ".inst 0x04b375ad  // sqrdmulh z13.s, z13.s, z19.s\n"
    "sdot z29.s, z18.b, z7.b\n"
    "sdot z27.s, z18.b, z6.b\n"
    "and z16.d, z13.d, z22.d\n"
    ".inst 0x04b3779c  // sqrdmulh z28.s, z28.s, z19.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x04b377bd  // sqrdmulh z29.s, z29.s, z19.s\n"
    ".inst 0x04b3777b  // sqrdmulh z27.s, z27.s, z19.s\n"
    "ld1b { z12.b }, p0/Z, [x28, x14]\n"
    "ldp x23, x22, [%x[inptrs], #0x40]\n"
    "and z19.d, z28.d, z22.d\n"
    "ldp x21, x20, [%x[inptrs], #0x50]\n"
    "sqadd z13.s, z13.s, z16.s\n"
    "and z17.d, z29.d, z22.d\n"
    "and z16.d, z27.d, z22.d\n"
    "asr z19.s, z19.s, #0x1f\n"
    "ld1b { z2.b }, p0/Z, [x23, x14]\n"
    "ld1b { z24.b }, p0/Z, [x22, x14]\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44828acd  // srshl z13.s, p2/M, z13.s, z22.s\n"
    "ld1b { z18.b }, p0/Z, [x21, x14]\n"
    "ld1b { z7.b }, p0/Z, [x20, x14]\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z28.s, z28.s, z19.s\n"
    "ld1b { z1.b }, p2/Z, [%x[params], #7, MUL VL]\n"
    "sqadd z29.s, z29.s, z17.s\n"
    "ld1b { z0.b }, p2/Z, [%x[params], #6, MUL VL]\n"
    "add z13.s, z13.s, z30.s\n"
    "sqadd z27.s, z27.s, z16.s\n"
    "ld1b { z3.b }, p2/Z, [%x[params], #5, MUL VL]\n"
    ".inst 0x44828adc  // srshl z28.s, p2/M, z28.s, z22.s\n"
    ".inst 0x44828add  // srshl z29.s, p2/M, z29.s, z22.s\n"
    "smax z13.s, p2/M, z13.s, z11.s\n"
    ".inst 0x44828adb  // srshl z27.s, p2/M, z27.s, z22.s\n"
    "ld1b { z10.b }, p0/Z, [x24, x14]\n"
    "ldp x23, x22, [%x[inptrs], #0x60]\n"
    "ldp x21, x20, [%x[inptrs], #0x70]\n"
    "ldp x28, x27, [%x[inptrs], #0x0]\n"
    "add z29.s, z29.s, z30.s\n"
    "add z28.s, z28.s, z30.s\n"
    "ldp x26, x25, [%x[inptrs], #0x10]\n"
    "add z27.s, z27.s, z30.s\n"
    "smin z13.s, p2/M, z13.s, z14.s\n"
    "ld1b { z4.b }, p0/Z, [x23, x14]\n"
    "ld1b { z23.b }, p0/Z, [x22, x14]\n"
    "ldp x24, x22, [%x[inptrs], #0x20]\n"
    "smax z29.s, p2/M, z29.s, z11.s\n"
    "smax z28.s, p2/M, z28.s, z11.s\n"
    "ld1b { z22.b }, p0/Z, [x21, x14]\n"
    "ld1b { z6.b }, p0/Z, [x20, x14]\n"
    "smax z27.s, p2/M, z27.s, z11.s\n"
    "st1b { z13.s }, p1, [x12, x13]\n"
    "zip2 z17.b, z12.b, z21.b\n"
    "zip1 z12.b, z12.b, z21.b\n"
    "ldp x21, x20, [%x[inptrs], #0x30]\n"
    "zip1 z16.b, z26.b, z15.b\n"
    "zip2 z15.b, z26.b, z15.b\n"
    "smin z29.s, p2/M, z29.s, z14.s\n"
    "smin z28.s, p2/M, z28.s, z14.s\n"
    "smin z27.s, p2/M, z27.s, z14.s\n"
    "st1b { z29.s }, p1, [x11, x13]\n"
    "zip2 z21.b, z10.b, z20.b\n"
    "zip1 z10.b, z10.b, z20.b\n"
    "zip1 z20.b, z25.b, z8.b\n"
    "zip2 z8.b, z25.b, z8.b\n"
    "st1b { z28.s }, p1, [x10, x13]\n"
    "zip2 z5.b, z12.b, z16.b\n"
    "zip1 z12.b, z12.b, z16.b\n"
    "st1b { z27.s }, p1, [x9, x13]\n"
    "incw x13\n"
    "zip1 z9.b, z17.b, z15.b\n"
    "zip2 z15.b, z17.b, z15.b\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "addvl %x[params], %x[params], #8\n"
    "zip2 z19.b, z2.b, z18.b\n"
    "zip1 z2.b, z2.b, z18.b\n"
    "zip1 z18.b, z24.b, z7.b\n"
    "zip2 z7.b, z24.b, z7.b\n"
    "zip2 z17.b, z4.b, z22.b\n"
    "zip1 z4.b, z4.b, z22.b\n"
    "zip1 z16.b, z23.b, z6.b\n"
    "zip2 z6.b, z23.b, z6.b\n"
    "zip2 z31.b, z10.b, z20.b\n"
    "zip1 z10.b, z10.b, z20.b\n"
    "zip1 z26.b, z21.b, z8.b\n"
    "zip2 z8.b, z21.b, z8.b\n"
    "zip2 z25.b, z2.b, z18.b\n"
    "zip1 z2.b, z2.b, z18.b\n"
    "zip1 z28.b, z19.b, z7.b\n"
    "zip2 z7.b, z19.b, z7.b\n"
    "zip2 z27.b, z4.b, z16.b\n"
    "zip1 z4.b, z4.b, z16.b\n"
    "zip1 z29.b, z17.b, z6.b\n"
    "zip2 z6.b, z17.b, z6.b\n"
    "mov z21.d, z13.d\n"
    "mov z20.d, z13.d\n"
    "mov z23.d, z13.d\n"
    "b.any 1b\n"
    : [params] "+&r" (params)
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [outptrs] "r" (outptrs), [qp] "r" (&qp)
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
