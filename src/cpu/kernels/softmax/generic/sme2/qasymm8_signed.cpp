/*
 * Copyright (c) 2023-2024 Arm Limited.
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

#ifdef ARM_COMPUTE_ENABLE_SME2

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{

// SoftMax
//
// Steps:
//   * Find max:   max_value = max(src)
//   * Regularize: dst[i] = exp(src[i] - max_value)
//                 sum_value = sum(dst)
//   * Normalize:  dst[i] = dst[i] / sum_value
void sme2_qasymm8_signed_softmax_kernel_512VL( //
    const int8_t   *src,
    int8_t         *dst,
    float           beta,
    const uintptr_t shape[4],
    const uintptr_t src_strides[4],
    const uintptr_t dst_strides[4],
    const float    *lut,
    float          *tmp)
{
    // Precondition:
    //   * src_strides[0] == sizeof(int8_t)
    //   * dst_strides[0] == sizeof(int8_t)
    //   * tmp_strides[0] == sizeof(float)

    __asm__ volatile(
        R"(
            .inst 0xd503477f  // smstart

            // For register list explanation refer to qasymm8.cpp.

            // Prepares all constant values

            ptrue p0.b
            .inst 0x25a07811  // ptrue pn9.s
            .inst 0x25207810  // ptrue pn8.b

            // ---------------------------------------------------------------- x13: body_length = (length / vl) * vl
            cntb x13, ALL, MUL #4
            udiv x9, %x[length], x13
            mul x13, x13, x9

            // ==================================================
            // 3D loop opening
            // ==================================================

            mov x20, %x[shape_3]
            mov x21, %x[src]
            mov x22, %x[dst]
            mov x29, %x[tmp]

            // Load the LUT to the register file.
            mov x2, %x[lut]
            .inst 0xa040c440  //ld1w    { z0.s - z3.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c444  //ld1w    { z4.s - z7.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c448  //ld1w    { z8.s - z11.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c44c  //ld1w    { z12.s - z15.s }, pn9/z, [x2]


1: // loop_3_start
            // for index_3 in shape_3 downto 1
            cmp x20, #0
            b.eq 16f // loop_3_end
            sub x20, x20, #1

            mov x23, %x[shape_2]
            mov x24, x21
            mov x25, x22

2: // loop_2_start
            // for index_2 in shape_2 downto 1
            cmp x23, #0
            b.eq 15f // loop_2_end
            sub x23, x23, #1

            mov x26, %x[shape_1]
            mov x27, x24
            mov x28, x25

3: // loop_1_start
            // for index_1 in shape_2 downto 1
            cmp x26, #0
            b.eq 14f // loop_1_end
            sub x26, x26, #1

            // ==================================================
            // Step 1: Find max
            // ==================================================
            // z16-z19 = minimum QASYMM8_SIGNED value (-128) to allow for it to be used for comparison to find the max.
            dup z16.b, #0x80
            dup z17.b, #0x80
            dup z18.b, #0x80
            dup z19.b, #0x80

            mov x1, #0                                                  // x1: index
4: // find_max_body_start
            cmp x1, x13
            b.eq 5f // find_max_body_end
            .inst 0xa0018374 // ld1b    { z20.b - z23.b }, pn8/z, [x27, x1]  z16-z19: x
            .inst 0xc134b810 // smax    { z16.b - z19.b }, { z16.b - z19.b }, { z20.b - z23.b } z16-z19: max_value = max(max_value, x)
            add x1, x1, #256 // Advance index by 256 bytes/integers: Z registers = 2048-bit data = 256 8-bit integers.
            b 4b // find_max_body_start
5: // find_max_body_end

            // Loop for processing the leftover part.
6: // find_max_leftover_start
            whilelo p1.b, x1, %x[length]
            b.none 7f // find_max_leftover_end

            ld1b z30.b, p1/z, [x27, x1]                                // z30: x
            smax z16.b, p1/m, z16.b, z30.b                             // z16: max_value = max(max_value, x)

            add x1, x1, #64

            b 6b // find_max_leftover_start

7: // find_max_leftover_end
            .inst 0xc132b010 // smax    { z16.b, z17.b }, { z16.b, z17.b }, { z18.b, z19.b }
            smax z16.b, p0/m, z16.b, z17.b
            smaxv b16, p0, z16.b // Reduction signed max operation to get maximum_value
            mov z16.b, b16       // z16: duplicated max_value for current row

            sunpklo z16.h, z16.b // Using unpack instructions to align the max value with the FP32 entries in the LUT for use in the TBX instruction
            sunpklo z16.s, z16.h

            mov x1, #0 // reset index
            dup z25.s, #0

8: // regularize_start:
            whilelo p1.b, x1, %x[length]
            b.none 9f // regularize_end

            mov w9, 0xFF80
            movk w9, 0xFFFF, LSL #16 // Moving -127.f into w9 to set the registers below to the minimum QASYMM8_SIGNED value
            dup z17.s, w9
            dup z18.s, w9
            dup z19.s, w9
            dup z20.s, w9

            dup z21.s, #0x0
            dup z22.s, #0x0
            dup z23.s, #0x0
            dup z24.s, #0x0

            // p2-p5 are - together - the 32-bit version of p1, the instructions below unpack p1 into those four predicate registers to allow for the 32-bit loads below to be correctly predicated
            punpklo  p2.h, p1.b
            punpkhi  p4.h, p1.b

            punpkhi  p3.h, p2.b
            punpklo  p2.h, p2.b

            punpkhi  p5.h, p4.b
            punpklo  p4.h, p4.b

            ld1b z17.b, p1/z, [x27, x1] //z17: input data

            sunpklo z18.h, z17.b // Using unpack instructions to align the input QASYMM8_SIGNED values with the FP32 entries in the LUT for use in the TBX instruction
            sunpkhi z19.h, z17.b //

            sunpklo z17.s, z18.h // z17 = low  low  input QASYMM8_SIGNED values
            sunpkhi z18.s, z18.h // z18 = low  high input QASYMM8_SIGNED values

            sunpkhi z20.s, z19.h // z20 = high high input QASYMM8_SIGNED values
            sunpklo z19.s, z19.h // z19 = high low  input QASYMM8_SIGNED values

            sub z17.s, z16.s, z17.s                                          // z12: x =  max_value - input_data
            sub z18.s, z16.s, z18.s                                          // z13: x =  max_value - input_data
            sub z19.s, z16.s, z19.s                                          // z14: x =  max_value - input_data
            sub z20.s, z16.s, z20.s                                          // z15: x =  max_value - input_data

            add z17.s, z17.s, #128
            add z18.s, z18.s, #128
            add z19.s, z19.s, #128
            add z20.s, z20.s, #128

            tbx z21.s, z0.s, z17.s  // Look-up entries 0-15 in the LUT.
            tbx z22.s, z0.s, z18.s
            tbx z23.s, z0.s, z19.s
            tbx z24.s, z0.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z1.s, z17.s  // Look-up entries 16-31 in the LUT.
            tbx z22.s, z1.s, z18.s
            tbx z23.s, z1.s, z19.s
            tbx z24.s, z1.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z2.s, z17.s  // Look-up entries 32-47 in the LUT.
            tbx z22.s, z2.s, z18.s
            tbx z23.s, z2.s, z19.s
            tbx z24.s, z2.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z3.s, z17.s  // Look-up entries 48-63 in the LUT.
            tbx z22.s, z3.s, z18.s
            tbx z23.s, z3.s, z19.s
            tbx z24.s, z3.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z4.s, z17.s  // Look-up entries 64-79 in the LUT.
            tbx z22.s, z4.s, z18.s
            tbx z23.s, z4.s, z19.s
            tbx z24.s, z4.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z5.s, z17.s  // Look-up entries 80-95 in the LUT.
            tbx z22.s, z5.s, z18.s
            tbx z23.s, z5.s, z19.s
            tbx z24.s, z5.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z6.s, z17.s  // Look-up entries 96-111 in the LUT.
            tbx z22.s, z6.s, z18.s
            tbx z23.s, z6.s, z19.s
            tbx z24.s, z6.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z7.s, z17.s  // Look-up entries 112-127 in the LUT.
            tbx z22.s, z7.s, z18.s
            tbx z23.s, z7.s, z19.s
            tbx z24.s, z7.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z8.s, z17.s  // Look-up entries 128-143 in the LUT.
            tbx z22.s, z8.s, z18.s
            tbx z23.s, z8.s, z19.s
            tbx z24.s, z8.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z9.s, z17.s  // Look-up entries 144-159 in the LUT.
            tbx z22.s, z9.s, z18.s
            tbx z23.s, z9.s, z19.s
            tbx z24.s, z9.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z10.s, z17.s  // Look-up entries 160-175 in the LUT.
            tbx z22.s, z10.s, z18.s
            tbx z23.s, z10.s, z19.s
            tbx z24.s, z10.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z11.s, z17.s  // Look-up entries 176-191 in the LUT.
            tbx z22.s, z11.s, z18.s
            tbx z23.s, z11.s, z19.s
            tbx z24.s, z11.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z12.s, z17.s  // Look-up entries 192-207 in the LUT.
            tbx z22.s, z12.s, z18.s
            tbx z23.s, z12.s, z19.s
            tbx z24.s, z12.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z13.s, z17.s  // Look-up entries 208-223 in the LUT.
            tbx z22.s, z13.s, z18.s
            tbx z23.s, z13.s, z19.s
            tbx z24.s, z13.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z14.s, z17.s  // Look-up entries 224-239 in the LUT.
            tbx z22.s, z14.s, z18.s
            tbx z23.s, z14.s, z19.s
            tbx z24.s, z14.s, z20.s

            sub z17.s, z17.s, #16
            sub z18.s, z18.s, #16
            sub z19.s, z19.s, #16
            sub z20.s, z20.s, #16

            tbx z21.s, z15.s, z17.s  // Look-up entries 240-255 in the LUT.
            tbx z22.s, z15.s, z18.s
            tbx z23.s, z15.s, z19.s
            tbx z24.s, z15.s, z20.s


            st1w z21.s, p2, [x29, x1, LSL #2]// z21 store exp(-scale*beta*x) into the tmp tensor
            fadd z25.s, p2/m, z25.s, z21.s
            add x1, x1, #16

            st1w z22.s, p3, [x29, x1, LSL #2]// z22 store exp(-scale*beta*x) into the tmp tensor
            fadd z25.s, p3/m, z25.s, z22.s
            add x1, x1, #16

            st1w z23.s, p4, [x29, x1, LSL #2]// z23 store exp(-scale*beta*x) into the tmp tensor
            fadd z25.s, p4/m, z25.s, z23.s
            add x1, x1, #16

            st1w z24.s, p5, [x29, x1, LSL #2]// z24 store exp(-scale*beta*x) into the tmp tensor
            fadd z25.s, p5/m, z25.s, z24.s
            add x1, x1, #16

            b 8b // regularize_start
9: // regularize_end

            mov w9, 0x0000
            movk w9, 0x4380, LSL #16 // Moving 256.f into w9 to scale - via multiplication (division by reciprocal) - the floating point [0,1] range of the results to the [-128, 127] integer range of QASYMM8_SIGNED
            mov w10, 0x0000
            movk w10, 0x4300, LSL #16 // Moving 128.f into w10 for the subtraction to move the results - via subtraction - from the [0,255] range to the [-128,127] range
            dup z29.s, w9
            dup z30.s, w10
            faddv s25, p0, z25.s
            fdiv s25, s29, s25
            dup z25.s, z25.s[0] // z25: 256.f/sum. 256 is needed to get the full range and 1/sum is part of softmax.

            // ==================================================
            // Step 3: Normalize
            // ==================================================
            mov x1, #0
10: // normalize_body_start
            cmp x1, x13
            b.eq 11f // normalize_body_end

            mov x2, x1       // Preserve the index into x2 for the final store to dst.
            .inst 0xa001c7b0 // ld1w    { z16.s - z19.s }, pn9/z, [x29, x1, lsl #2]
            add x1, x1, #64
            .inst 0xa001c7b4 // ld1w    { z20.s - z23.s }, pn9/z, [x29, x1, lsl #2]
            add x1, x1, #64

            // z16-z23: effectively divides exp(-scale*beta*x) by the sum of the exponentials for the current row and multiplies by 256.
            fmul z16.s, z25.s, z16.s
            fmul z17.s, z25.s, z17.s
            fmul z18.s, z25.s, z18.s
            fmul z19.s, z25.s, z19.s
            fmul z20.s, z25.s, z20.s
            fmul z21.s, z25.s, z21.s
            fmul z22.s, z25.s, z22.s
            fmul z23.s, z25.s, z23.s

            // z16-z23: subtract 128.f.
            fsub z16.s, z16.s, z30.s // Subtract 128.f
            fsub z17.s, z17.s, z30.s // Subtract 128.f
            fsub z18.s, z18.s, z30.s // Subtract 128.f
            fsub z19.s, z19.s, z30.s // Subtract 128.f
            fsub z20.s, z20.s, z30.s // Subtract 128.f
            fsub z21.s, z21.s, z30.s // Subtract 128.f
            fsub z22.s, z22.s, z30.s // Subtract 128.f
            fsub z23.s, z23.s, z30.s // Subtract 128.f

            // z16-z23: convert the FP32 values from the tmp tensor to int32.
            fcvtzs z16.s, p0/m, z16.s
            fcvtzs z17.s, p0/m, z17.s
            fcvtzs z18.s, p0/m, z18.s
            fcvtzs z19.s, p0/m, z19.s
            fcvtzs z20.s, p0/m, z20.s
            fcvtzs z21.s, p0/m, z21.s
            fcvtzs z22.s, p0/m, z22.s
            fcvtzs z23.s, p0/m, z23.s

            // z16-z17: narrow the int32 values into int8 and saturate them.
            .inst 0xc133e210 // sqcvt    z16.b, { z16.s - z19.s }
            .inst 0xc133e291 // sqcvt    z17.b, { z20.s - z23.s }

            // Juggling the value to z20 (resp. 21) as z25 (resp. z30) will be overwritten by the load below.
            dup z20.s, z25.s[0]
            dup z21.s, z30.s[0]

            .inst 0xa001c7b8 // ld1w    { z24.s - z27.s }, pn9/z, [x29, x1, lsl #2]
            add x1, x1, #64
            .inst 0xa001c7bc // ld1w    { z28.s - z31.s }, pn9/z, [x29, x1, lsl #2]
            add x1, x1, #64

            // z24-z31: effectively divides exp(-scale*beta*x) by the sum of the exponentials for the current row and multiplies by 256.
            fmul z24.s, z20.s, z24.s
            fmul z25.s, z20.s, z25.s
            fmul z26.s, z20.s, z26.s
            fmul z27.s, z20.s, z27.s
            fmul z28.s, z20.s, z28.s
            fmul z29.s, z20.s, z29.s
            fmul z30.s, z20.s, z30.s
            fmul z31.s, z20.s, z31.s

            // z24-z31: subtract 128.f.
            fsub z24.s, z24.s, z21.s
            fsub z25.s, z25.s, z21.s
            fsub z26.s, z26.s, z21.s
            fsub z27.s, z27.s, z21.s
            fsub z28.s, z28.s, z21.s
            fsub z29.s, z29.s, z21.s
            fsub z30.s, z30.s, z21.s
            fsub z31.s, z31.s, z21.s

            // z24-z31: convert the FP32 values from the tmp tensor to int32.
            fcvtzs z24.s, p0/m, z24.s
            fcvtzs z25.s, p0/m, z25.s
            fcvtzs z26.s, p0/m, z26.s
            fcvtzs z27.s, p0/m, z27.s
            fcvtzs z28.s, p0/m, z28.s
            fcvtzs z29.s, p0/m, z29.s
            fcvtzs z30.s, p0/m, z30.s
            fcvtzs z31.s, p0/m, z31.s

            // z18-z19: narrow the int32 values into int8 and saturate them.
            .inst 0xc133e312 // sqcvt    z18.b, { z24.s - z27.s }
            .inst 0xc133e393 // sqcvt    z19.b, { z28.s - z31.s }

            .inst 0xa0228390 // st1b    { z16.b - z19.b }, pn8, [x28, x2]

            // Juggling the values back to z25 (resp. z30) as z20 (resp. z21) will be overwritten by the next iteration or z25 (resp. z30) will be used below.
            dup z25.s, z20.s[0]
            dup z30.s, z21.s[0]
            b 10b // normalize_body_start
11: // normalize_body_end
12: // normalize_leftover_start
            whilelo p1.b, x1, %x[length]
            b.none 13f // normalize_leftover_end

            // p2-p5 are - together - the 32-bit version of p1, the instructions below unpack p1 into those four predicate registers to allow for the 32-bit loads below to be correctly predicated
            punpklo  p2.h, p1.b
            punpkhi  p4.h, p1.b

            punpkhi  p3.h, p2.b
            punpklo  p2.h, p2.b

            punpkhi  p5.h, p4.b
            punpklo  p4.h, p4.b

            mov x2, x1 // Preserve the index into x2 for the final store to dst.

            // z20-z23: load exp(-scale*beta*x) from the tmp tensor
            ld1w z20.s, p2/z, [x29, x1, LSL #2]
            add x1, x1, #16

            ld1w z21.s, p3/z, [x29, x1, LSL #2]
            add x1, x1, #16

            ld1w z22.s, p4/z, [x29, x1, LSL #2]
            add x1, x1, #16

            ld1w z23.s, p5/z, [x29, x1, LSL #2]
            add x1, x1, #16

            // z20-z23: effectively divides exp(-scale*beta*x) by the sum of the exponentials for the current row and multiplies by 256.
            fmul z20.s, z25.s, z20.s
            fmul z21.s, z25.s, z21.s
            fmul z22.s, z25.s, z22.s
            fmul z23.s, z25.s, z23.s

            //z20-z23: Subtract 128.f.
            fsub z20.s, z20.s, z30.s
            fsub z21.s, z21.s, z30.s
            fsub z22.s, z22.s, z30.s
            fsub z23.s, z23.s, z30.s

            // z20-23: convert the FP32 values from the tmp tensor to int32.
            fcvtzs z20.s, p0/m, z20.s
            fcvtzs z21.s, p0/m, z21.s
            fcvtzs z22.s, p0/m, z22.s
            fcvtzs z23.s, p0/m, z23.s

            .inst 0xc133e293 // sqcvt	z19.b, { z20.s - z23.s }, narrow the int32 values into int8 and saturate them into z19.

            st1b z19.b, p1, [x28, x2]

            b 12b // normalize_leftover_start
13: // normalize_leftover_end
            // ==================================================
            // 3D loop closing
            // ==================================================
            add x27, x27, %x[src_stride_1]
            add x28, x28, %x[dst_stride_1]
            b 3b // loop_1_start
14: // loop_1_end

            add x24, x24, %x[src_stride_2]
            add x25, x25, %x[dst_stride_2]
            b 2b // loop_2_start
15: // loop_2_end

            add x21, x21, %x[src_stride_3]
            add x22, x22, %x[dst_stride_3]
            b 1b // loop_3_start
16: // loop_3_end
            .inst 0xd503467f  // smstop
        )"
        :
        : [src] "r"(src), [tmp] "r"(tmp), [dst] "r"(dst), [beta] "r"(beta), [lut] "r"(lut), //
          [shape_1] "r"(shape[1]), [shape_2] "r"(shape[2]), [shape_3] "r"(shape[3]),        //
          [src_stride_1] "r"(src_strides[1]), [src_stride_2] "r"(src_strides[2]),
          [src_stride_3] "r"(src_strides[3]), //
          [dst_stride_1] "r"(dst_strides[1]), [dst_stride_2] "r"(dst_strides[2]),
          [dst_stride_3] "r"(dst_strides[3]),                            //
          [length] "r"(shape[0])                                         //
        : "cc", "memory",                                                //
          "p0", "p1", "p2", "p3", "p4",                                  //
          "x2", "x9", "x13",                                             //
          "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", //
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                //
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",          //
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",        //
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"         //
    );
}

void sme2_qasymm8_signed_softmax_lut_512VL(const ITensor *in,
                                           void *const    tmp,
                                           ITensor       *out,
                                           const float    beta,
                                           int            axis,
                                           const Window  &window,
                                           const void    *lut_ptr)
{
    ARM_COMPUTE_UNUSED(axis);

    auto lut_fp32_ptr = reinterpret_cast<const float *>(lut_ptr);

    const auto *src_info = in->info();
    const auto *dst_info = out->info();

    const auto &full_shape  = dst_info->tensor_shape();
    const auto &src_strides = src_info->strides_in_bytes();
    const auto &dst_strides = dst_info->strides_in_bytes();
    Strides     tmp_strides;

    tmp_strides[0] = src_strides[0] * 4;
    tmp_strides[1] = src_strides[1] * 4;
    tmp_strides[2] = src_strides[2] * 4;
    tmp_strides[3] = src_strides[3] * 4;

    const uintptr_t k_shape[] = {
        full_shape[0],
        window.num_iterations(1),
        window.num_iterations(2),
        window.num_iterations(3),
    };

    const uintptr_t k_src_strides[] = {
        src_strides[0],
        src_strides[1],
        src_strides[2],
        src_strides[3],
    };

    const uintptr_t k_dst_strides[] = {
        dst_strides[0],
        dst_strides[1],
        dst_strides[2],
        dst_strides[3],
    };

    const uintptr_t k_src_offset = window[0].start() * src_strides[0] + //
                                   window[1].start() * src_strides[1] + //
                                   window[2].start() * src_strides[2] + //
                                   window[3].start() * src_strides[3];

    const uintptr_t k_dst_offset = window[0].start() * dst_strides[0] + //
                                   window[1].start() * dst_strides[1] + //
                                   window[2].start() * dst_strides[2] + //
                                   window[3].start() * dst_strides[3];

    const uintptr_t k_tmp_offset = window[0].start() * tmp_strides[0] + //
                                   window[1].start() * tmp_strides[1] + //
                                   window[2].start() * tmp_strides[2] + //
                                   window[3].start() * tmp_strides[3];

    const auto *k_src         = reinterpret_cast<const int8_t *>(in->buffer() + k_src_offset);
    float      *tmp_float_ptr = reinterpret_cast<float *>(tmp);
    auto       *k_tmp         = reinterpret_cast<float *>(tmp_float_ptr + k_tmp_offset);
    auto       *k_dst         = reinterpret_cast<int8_t *>(out->buffer() + k_dst_offset);

    sme2_qasymm8_signed_softmax_kernel_512VL(k_src, k_dst, beta, k_shape, k_src_strides, k_dst_strides, lut_fp32_ptr,
                                             k_tmp);
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SME2
