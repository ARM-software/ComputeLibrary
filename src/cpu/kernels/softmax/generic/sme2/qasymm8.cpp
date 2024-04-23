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
void sme2_qasymm8_softmax_kernel_512VL( //
    const uint8_t  *src,
    uint8_t        *dst,
    float           beta,
    const uintptr_t shape[4],
    const uintptr_t src_strides[4],
    const uintptr_t dst_strides[4],
    const float    *lut,
    float          *tmp)
{
    // Precondition:
    //   * src_strides[0] == sizeof(uint8_t)
    //   * dst_strides[0] == sizeof(uint8_t)
    //   * tmp_strides[0] == sizeof(float)

    __asm__ volatile(
        R"(
            .inst 0xd503477f // smstart

            // Registers
            //
            //   *  x1: Loop index
            //   *  x2: LUT index
            //   * x13: temporary, body_length
            //
            //   * x20: index_3
            //   * x21: src_3
            //   * x22: dst_3
            //   * x23: index_2
            //   * x24: src_2
            //   * x25: dst_2
            //   * x26: index_1
            //   * x27: src_1
            //   * x28: dst_1
            //   * x29  tmp
            //
            //
            //   * p0: all-true
            //   * p1: predicate for QASYMM8 values
            //   * p2: predicate 0 for FP32 values (first quarter of expanded/unpacked p1)
            //   * p3: predicate 1 for FP32 values (second quarter of expanded/unpacked p1)
            //   * p4: predicate 2 for FP32 values (third quarter of expanded/unpacked p1)
            //   * p5: predicate 3 for FP32 values (fourth quarter of expanded/unpacked p1)
            //   * pn9: all-true for 32 bit values
            //   * pn8: all-true for 8-bit values
            //
            //   * z0-z15 the 256 LUT values of exp(-scale*beta*x) for x in QASYMM8, stored as FP32 values

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
            mov x19, %x[lut]
            mov x29, %x[tmp]

            // Load the LUT to the register file.
            mov x2, %x[lut]
            .inst 0xa040c440 //ld1w    { z0.s - z3.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c444 //ld1w    { z4.s - z7.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c448 //ld1w    { z8.s - z11.s }, pn9/z, [x2]
            add x2, x2, #256
            .inst 0xa040c44c //ld1w    { z12.s - z15.s }, pn9/z, [x2]


loop_3_start%=:
            // for index_3 in shape_3 downto 1
            cmp x20, #0
            b.eq loop_3_end%=
            sub x20, x20, #1

            mov x23, %x[shape_2]
            mov x24, x21
            mov x25, x22

loop_2_start%=:
            // for index_2 in shape_2 downto 1
            cmp x23, #0
            b.eq loop_2_end%=
            sub x23, x23, #1

            mov x26, %x[shape_1]
            mov x27, x24
            mov x28, x25

loop_1_start%=:
            // for index_1 in shape_2 downto 1
            cmp x26, #0
            b.eq loop_1_end%=
            sub x26, x26, #1

            // ==================================================
            // Step 1: Find max
            // ==================================================
            // z16-z19 = minimum QASYMM8 value (0) to allow for it to be used for comparison to find the max.
            dup z16.b, #0
            dup z17.b, #0
            dup z18.b, #0
            dup z19.b, #0
            mov x1, #0                                                  // x1: index
find_max_body_start%=:
            cmp x1, x13
            b.eq find_max_body_end%=
            .inst 0xa0018374 // ld1b    { z20.b - z23.b }, pn8/z, [x27, x1]  z20-z23: x
            .inst 0xc134b811 // umax    { z16.b - z19.b }, { z16.b - z19.b }, { z20.b - z23.b } z16-z19: max_value = max(max_value, x)
            add x1, x1, #256 // Advance index by 256 bytes/integers: Z registers = 2048-bit data = 256 8-bit integers.
            b find_max_body_start%=
find_max_body_end%=:

            // Loop for processing the leftover part.
find_max_leftover_start%=:
            whilelo p1.b, x1, %x[length]
            b.none find_max_leftover_end%=

            ld1b z30.b, p1/z, [x27, x1]                                // z30: x
            umax z16.b, p1/m, z16.b, z30.b                             // z16: max_value = max(max_value, x)

            add x1, x1, #64

            b find_max_leftover_start%=
find_max_leftover_end%=:

            .inst 0xc132b011 // umax    { z16.b, z17.b }, { z16.b, z17.b }, { z18.b, z19.b }
            umax z16.b, p0/m, z16.b, z17.b
            umaxv b16, p0, z16.b // Reduction unsigned max operation to get maximum_value
            dup z16.b, z16.b[0]
            uunpklo z16.h, z16.b // Using unpack instructions to align the max value with the FP32 entries in the LUT for use in the TBX instruction
            uunpklo z16.s, z16.h

            mov x1, #0 // reset index
            dup z25.s, #0

            mov x1, #0

regularize_start%=:
            whilelo p1.b, x1, %x[length]
            b.none regularize_end%=

            // p2-p5 are - together - the 32-bit version of p1, the instructions below unpack p1 into those four predicate registers to allow for the 32-bit loads below to be correctly predicated
            punpklo  p2.h, p1.b
            punpkhi  p4.h, p1.b

            punpkhi  p3.h, p2.b
            punpklo  p2.h, p2.b

            punpkhi  p5.h, p4.b
            punpklo  p4.h, p4.b

            ld1b z17.b, p1/z, [x27, x1] //z17: input data

            uunpklo z18.h, z17.b //Using unpack instructions to align the input QASYMM8 values with the FP32 entries in the LUT for use in the TBX instruction
            uunpkhi z19.h, z17.b

            uunpklo z17.s, z18.h // z17 = low  low  input QASYMM8 values
            uunpkhi z18.s, z18.h // z18 = low  high input QASYMM8 values

            uunpkhi z20.s, z19.h // z20 = high high input QASYMM8 values
            uunpklo z19.s, z19.h // z19 = high low  input QASYMM8 values

            sub z17.s, z16.s, z17.s                                          // z12: x =  max_value - input_data
            sub z18.s, z16.s, z18.s                                          // z13: x =  max_value - input_data
            sub z19.s, z16.s, z19.s                                          // z14: x =  max_value - input_data
            sub z20.s, z16.s, z20.s                                          // z15: x =  max_value - input_data

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

            b regularize_start%=
regularize_end%=:

            mov w9, 0x0000
            movk w9, 0x4380, LSL #16 // Moving 256.f into w9 to scale - via multiplication (division by reciprocal) - the floating point [0,1] range of the results to the [0,255] integer range of QASYMM8
            dup z29.s, w9
            faddv s25, p0, z25.s
            fdiv s25, s29, s25
            dup z25.s, z25.s[0] // z25: 256.f/sum. 256 is needed to get the full range and 1/sum is part of softmax.

            // ==================================================
            // Step 3: Normalize
            // ==================================================
            mov x1, #0
normalize_body_start%=:
            cmp x1, x13
            b.eq normalize_body_end%=

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

            // z16-z23: convert the FP32 values from the tmp tensor to uint32.
            fcvtzu z16.s, p0/m, z16.s
            fcvtzu z17.s, p0/m, z17.s
            fcvtzu z18.s, p0/m, z18.s
            fcvtzu z19.s, p0/m, z19.s
            fcvtzu z20.s, p0/m, z20.s
            fcvtzu z21.s, p0/m, z21.s
            fcvtzu z22.s, p0/m, z22.s
            fcvtzu z23.s, p0/m, z23.s

            // z16-z17: narrow the uint32 values into uint8 and saturate them.
            .inst 0xc133e230 // uqcvt    z16.b, { z16.s - z19.s }
            .inst 0xc133e2b1 // uqcvt    z17.b, { z20.s - z23.s }

            dup z20.s, z25.s[0] // Juggling the value to z20 as z25 will be overwritten by the load below

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

            // z24-z31: convert the FP32 values from the tmp tensor to uint32.
            fcvtzu z24.s, p0/m, z24.s
            fcvtzu z25.s, p0/m, z25.s
            fcvtzu z26.s, p0/m, z26.s
            fcvtzu z27.s, p0/m, z27.s
            fcvtzu z28.s, p0/m, z28.s
            fcvtzu z29.s, p0/m, z29.s
            fcvtzu z30.s, p0/m, z30.s
            fcvtzu z31.s, p0/m, z31.s

            // z18-z19: narrow the uint32 values into uint8 and saturate them.
            .inst 0xc133e332 // uqcvt    z18.b, { z24.s - z27.s }
            .inst 0xc133e3b3 // uqcvt    z19.b, { z28.s - z31.s }

            .inst 0xa0228390 // st1b    { z16.b - z19.b }, pn8, [x28, x2]

            dup z25.s, z20.s[0] // Juggling the value back to z25 as z20 will be overwritten by the next iteration or z25 will be used below.

b normalize_body_start%=
normalize_body_end%=:

normalize_leftover_start%=:
            whilelo p1.b, x1, %x[length]
            b.none normalize_leftover_end%=

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

            // z20-23: convert the FP32 values from the tmp tensor to uint32.
            fcvtzu z20.s, p0/m, z20.s
            fcvtzu z21.s, p0/m, z21.s
            fcvtzu z22.s, p0/m, z22.s
            fcvtzu z23.s, p0/m, z23.s

            .inst 0xc133e2b3 // uqcvt    z19.b, { z20.s - z23.s }, narrow the uint32 values into uint8 and saturate them into z19.

            st1b z19.b, p1, [x28, x2]

            b normalize_leftover_start%=
normalize_leftover_end%=:
            // ==================================================
            // 3D loop closing
            // ==================================================
            add x27, x27, %x[src_stride_1]
            add x28, x28, %x[dst_stride_1]
            b loop_1_start%=
loop_1_end%=:

            add x24, x24, %x[src_stride_2]
            add x25, x25, %x[dst_stride_2]
            b loop_2_start%=
loop_2_end%=:

            add x21, x21, %x[src_stride_3]
            add x22, x22, %x[dst_stride_3]
            b loop_3_start%=
loop_3_end%=:
            .inst 0xd503467f // smstop
        )"
        :
        : [src] "r"(src), [tmp] "r"(tmp), [dst] "r"(dst), [beta] "r"(beta), [lut] "r"(lut), //
          [shape_1] "r"(shape[1]), [shape_2] "r"(shape[2]), [shape_3] "r"(shape[3]),        //
          [src_stride_1] "r"(src_strides[1]), [src_stride_2] "r"(src_strides[2]),
          [src_stride_3] "r"(src_strides[3]), //
          [dst_stride_1] "r"(dst_strides[1]), [dst_stride_2] "r"(dst_strides[2]),
          [dst_stride_3] "r"(dst_strides[3]),                                   //
          [length] "r"(shape[0])                                                //
        : "cc", "memory",                                                       //
          "p0", "p1", "p2", "p3", "p4",                                         //
          "x2", "x9", "x13",                                                    //
          "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x19", //
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                       //
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",                 //
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",               //
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"                //
    );
}

void sme2_qasymm8_softmax_lut_512VL(const ITensor *in,
                                    void *const    tmp,
                                    ITensor       *out,
                                    const float    beta,
                                    int            axis,
                                    const Window  &window,
                                    const float   *lut_ptr)
{
    ARM_COMPUTE_UNUSED(axis);

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

    const auto *k_src         = reinterpret_cast<const uint8_t *>(in->buffer() + k_src_offset);
    float      *tmp_float_ptr = reinterpret_cast<float *>(tmp);
    auto       *k_tmp         = reinterpret_cast<float *>(tmp_float_ptr + k_tmp_offset);
    auto       *k_dst         = reinterpret_cast<uint8_t *>(out->buffer() + k_dst_offset);

    sme2_qasymm8_softmax_kernel_512VL(k_src, k_dst, beta, k_shape, k_src_strides, k_dst_strides, lut_ptr, k_tmp);
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SME2
