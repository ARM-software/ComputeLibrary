/*
 * Copyright (c) 2024 Arm Limited.
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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{

// This function expects a collapsed 2D shape.
void sme2_f32_logistic_kernel(const float    *src,
                              float          *dst,
                              const uintptr_t shape[2],
                              const uintptr_t src_strides[2],
                              const uintptr_t dst_strides[2])
{
    // Precondition:
    assert(src_strides[0] == sizeof(float));
    assert(dst_strides[0] == sizeof(float));
    __asm__ volatile(
        R"(
            .inst 0xd503477f  // smstart

            ptrue p0.b
            .inst 0x25207811  // ptrue pn9.b

            // Registers
            //
            //   *  x9: temporary, index
            //   * x10: temporary, inf
            //   * x11: temporary, 0
            //   * x12: temporary, 1.0f
            //   * x13: temporary, body_length
            //
            //   * x26: index_1
            //   * x27: src_1
            //   * x28: dst_1
            //
            //   *  z0: c1
            //   *  z1: c2
            //   *  z2: c3
            //   *  z3: c4
            //   *  z4: c5
            //   *  z5: shift
            //   *  z6: inv_ln2
            //   *  z7: neg_ln2_hi
            //   *  z8: neg_ln2_lo
            //   *  z9: min_input, max_input
            //   * z10: 23, 0, 1, inf
            //   * z11: max_value
            //   * z12-z15: x, r_hi, r, r2
            //   * z16-z19: max_value, shift, z, scale, poly
            //   * z20-z21: n, p1, p12345
            //   * z22-z23: n, p23, p2345
            //   * z24-z25: p45
            //   * z26: max_input
            //   * z28-z31: sum_value
            //
            //   * za0-za3: sum_value
            //
            //   * p0: all-true
            //   * p1-p4: underflow,
            //   * p4: leftover predicate
            //   * p5-p8: overflow,
            //   * pn9: all-true

            // TAYLORS CONSTANTS
            mov  w9, #0xfff6  // c1: 0x1.ffffecp-1f = 0x3f7ffff6
            mov w10, #0xfedb  // c2: 0x1.fffdb6p-2f = 0x3efffedb
            mov w11, #0xaf33  // c3: 0x1.555e66p-3f = 0x3e2aaf33
            mov w12, #0x9f17  // c4: 0x1.573e2ep-5f = 0x3d2b9f17
            mov w13, #0x2010  // c5: 0x1.0e4020p-7f = 0x3c072010

            movk  w9, #0x3f7f, LSL #16  // c1: 0x1.ffffecp-1f = 0x3f7ffff6
            movk w10, #0x3eff, LSL #16  // c2: 0x1.fffdb6p-2f = 0x3efffedb
            movk x11, #0x3e2a, LSL #16  // c3: 0x1.555e66p-3f = 0x3e2aaf33
            movk w12, #0x3d2b, LSL #16  // c4: 0x1.573e2ep-5f = 0x3d2b9f17
            movk w13, #0x3c07, LSL #16  // c5: 0x1.0e4020p-7f = 0x3c072010

            dup z0.s, w9   // c1.
            dup z1.s, w10  // c2.
            dup z2.s, w11  // c3.
            dup z3.s, w12  // c4.
            dup z4.s, w13  // c5.

            mov  w9, #0x007f  // shift: 2^23 + 127 = 0x1.0000fep23f = 0x4b00007f
            mov w10, #0xaa3b  // inv_ln2: 1 / ln(2) = 0x1.715476p+0f = 0x3fb8aa3b
            mov w11, #0x7200  // neg_ln2_hi: -ln(2) from bits  -1 to -19 = -0x1.62e400p-1f = 0xbf317200
            mov w12, #0xbe8e  // neg_ln2_lo: -ln(2) from bits -20 to -42 = -0x1.7f7d1cp-20f = 0xb5bfbe8e
            mov w13, #0x47ae  // min_input (Approximately ln 2^-125): -86.64 = 0xc2ad47ae
            mov w14, #0xBD71  // max_input 88.37 = 0x42B0BD71

            movk  w9, #0x4b00, LSL #16  // shift: 2^23 + 127 = 0x1.0000fep23f = 0x4b00007f
            movk w10, #0x3fb8, LSL #16  // inv_ln2: 1 / ln(2) = 0x1.715476p+0f = 0x3fb8aa3b
            movk w11, #0xbf31, LSL #16  // neg_ln2_hi: -ln(2) from bits  -1 to -19 = -0x1.62e400p-1f = 0xbf317200
            movk w12, #0xb5bf, LSL #16  // neg_ln2_lo: -ln(2) from bits -20 to -42 = -0x1.7f7d1cp-20f = 0xb5bfbe8e
            movk w13, #0xc2ad, LSL #16  // min_input (Approximately ln 2^-125): -86.64 = 0xc2ad47ae
            movk w14, #0x42B0, LSL #16  // max_input (88.37) = 0x42B0BD71

            dup z5.s, w9   // shift
            dup z6.s, w10  // inv_ln2
            dup z7.s, w11  // neg_ln2_hi
            dup z8.s, w12  // neg_ln2_lo
            dup z9.s, w13  // min_input
            dup z26.s, w14  // max_input

            mov w10, #0x0000  // inf: 0x7F800000
            movk w10, #0x7F80,  LSL #16 // inf: 0x7F800000

            mov w15, #0x0000
            movk w15, #0x3F80, LSL #16 // 1

            mov w11, #0  // 0

            // ---------------------------------------------------------------- x13: body_length = (length / vl) * vl
            cntw x13, ALL, MUL #4 // x13 is vl
            udiv x9, %x[length], x13 // length/vl
            mul x13, x13, x9 // x13 = vl * result

            // ==================================================
            // Outer loop opening
            // ==================================================

            mov x27, %x[src]  // starting point of pointers for src.
            mov x28, %x[dst]  // starting point of pointers for dst.
            mov x26, %x[shape_1]

outer_loop_start%=:
            // for index_1 in shape_1 downto 1
            cmp x26, #0
            b.eq outer_loop_end%=
            sub x26, x26, #1

            mov x9, #0                                                         // x9: index

inner_body_start%=:
            cmp x9, x13
            b.eq inner_body_end%=

            // Loads the input data to 4 consecutive registers ---------------- z12-z15: input_data
            .inst 0xa009c76c  // ld1w {z12.s-z15.s}, pn9/z, [x27, x9, LSL #2]

            // ---------------------------------------------------------------- z12-z15: x = neg(x)
            fneg z12.s, p0/m, z12.s
            fneg z13.s, p0/m, z13.s
            fneg z14.s, p0/m, z14.s
            fneg z15.s, p0/m, z15.s

            // ---------------------------------------------------------------- p4-p7: underflow = x < min_input
            fcmlt p1.s, p0/z, z12.s, z9.s
            fcmlt p2.s, p0/z, z13.s, z9.s
            fcmlt p3.s, p0/z, z14.s, z9.s
            fcmlt p4.s, p0/z, z15.s, z9.s

            // ---------------------------------------------------------------- p4-p7: overflow = x > max_input
            fcmlt p5.s, p0/z, z26.s, z12.s
            fcmlt p6.s, p0/z, z26.s, z13.s
            fcmlt p7.s, p0/z, z26.s, z14.s
            fcmlt p8.s, p0/z, z26.s, z15.s

            // ---------------------------------------------------------------- z16-z19: shift
            mov z16.d, z5.d
            mov z17.d, z5.d
            mov z18.d, z5.d
            mov z19.d, z5.d

            // ---------------------------------------------------------------- z16-z19: z = shift + x * inv_ln2
            fmla z16.s, p0/m, z12.s, z6.s
            fmla z17.s, p0/m, z13.s, z6.s
            fmla z18.s, p0/m, z14.s, z6.s
            fmla z19.s, p0/m, z15.s, z6.s

            // ---------------------------------------------------------------- z20-z23: n = z - shift
            fsub z20.s, z16.s, z5.s
            fsub z21.s, z17.s, z5.s
            fsub z22.s, z18.s, z5.s
            fsub z23.s, z19.s, z5.s

            // ---------------------------------------------------------------- z12-z15: r_hi = x + n * neg_ln2_hi
            fmla z12.s, p0/m, z20.s, z7.s
            fmla z13.s, p0/m, z21.s, z7.s
            fmla z14.s, p0/m, z22.s, z7.s
            fmla z15.s, p0/m, z23.s, z7.s

            // ---------------------------------------------------------------- z12-z15: r = r_hi + n * neg_ln2_lo
            fmla z12.s, p0/m, z20.s, z8.s
            fmla z13.s, p0/m, z21.s, z8.s
            fmla z14.s, p0/m, z22.s, z8.s
            fmla z15.s, p0/m, z23.s, z8.s

            // ---------------------------------------------------------------- z16-z19: scale = z << 23 (2^n)
            dup z10.s, #23
            urshl z16.s, p0/m, z16.s, z10.s
            urshl z17.s, p0/m, z17.s, z10.s
            urshl z18.s, p0/m, z18.s, z10.s
            urshl z19.s, p0/m, z19.s, z10.s

            // Processes the first 2 vectors.

            // ---------------------------------------------------------------- z20-z21: p1 = r * c1
            fmul z20.s, z12.s, z0.s
            fmul z21.s, z13.s, z0.s

            // ---------------------------------------------------------------- z22-z23: p23 = c2
            mov z22.d, z1.d
            mov z23.d, z1.d

            // ---------------------------------------------------------------- z22-z23: p23 = c2 + r * c3
            fmla z22.s, p0/m, z12.s, z2.s
            fmla z23.s, p0/m, z13.s, z2.s

            // ---------------------------------------------------------------- z24-z35: c4
            mov z24.d, z3.d
            mov z25.d, z3.d

            // ---------------------------------------------------------------- z24-z25: p45 = c4 + r * c5
            fmla z24.s, p0/m, z12.s, z4.s
            fmla z25.s, p0/m, z13.s, z4.s

            // ---------------------------------------------------------------- z12-z13: r2 = r * r
            fmul z12.s, z12.s, z12.s
            fmul z13.s, z13.s, z13.s

            // ---------------------------------------------------------------- z22-z23: p2345 = p23 + r2 * p45
            fmla z22.s, p0/m, z12.s, z24.s
            fmla z23.s, p0/m, z13.s, z25.s

            // ---------------------------------------------------------------- z20-z21: p12345 = p1 + r2 * p2345
            fmla z20.s, p0/m, z12.s, z22.s
            fmla z21.s, p0/m, z13.s, z23.s

            // ---------------------------------------------------------------- z16-z17: poly = scale + p12345 * scale
            fmla z16.s, p0/m, z20.s, z16.s
            fmla z17.s, p0/m, z21.s, z17.s

            // Processes the last 2 vectors

            // ---------------------------------------------------------------- z20-z21: p1 = r * c1
            fmul z20.s, z14.s, z0.s
            fmul z21.s, z15.s, z0.s

            // ---------------------------------------------------------------- z22-z23: p23 = c2
            mov z22.d, z1.d
            mov z23.d, z1.d

            // ---------------------------------------------------------------- z22-z23: p23 = c2 + r * c3
            fmla z22.s, p0/m, z14.s, z2.s
            fmla z23.s, p0/m, z15.s, z2.s

            // ---------------------------------------------------------------- z24-z35: c4
            mov z24.d, z3.d
            mov z25.d, z3.d

            // ---------------------------------------------------------------- z24-z25: p45 = c4 + r * c5
            fmla z24.s, p0/m, z14.s, z4.s
            fmla z25.s, p0/m, z15.s, z4.s

            // ---------------------------------------------------------------- z14-z15: r2 = r * r
            fmul z14.s, z14.s, z14.s
            fmul z15.s, z15.s, z15.s

            // ---------------------------------------------------------------- z22-z23: p2345 = p23 + r2 * p45
            fmla z22.s, p0/m, z14.s, z24.s
            fmla z23.s, p0/m, z15.s, z25.s

            // ---------------------------------------------------------------- z20-z21: p12345 = p1 + r2 * p2345
            fmla z20.s, p0/m, z14.s, z22.s
            fmla z21.s, p0/m, z15.s, z23.s

            // ---------------------------------------------------------------- z18-z19: poly = scale + p12345 * scale
            fmla z18.s, p0/m, z20.s, z18.s
            fmla z19.s, p0/m, z21.s, z19.s

            // ---------------------------------------------------------------- z16-z19: poly = underflow ? 0 : poly
            dup z10.s, #0
            sel z16.s, p1, z10.s, z16.s
            sel z17.s, p2, z10.s, z17.s
            sel z18.s, p3, z10.s, z18.s
            sel z19.s, p4, z10.s, z19.s

            // ---------------------------------------------------------------- z16-z19: poly = overflow ? inf : poly
            dup z10.s, w10 // z10: inf
            sel z16.s, p5, z10.s, z16.s
            sel z17.s, p6, z10.s, z17.s
            sel z18.s, p7, z10.s, z18.s
            sel z19.s, p8, z10.s, z19.s

            // 1 / 1 + poly
            dup z10.s, w15 // z10: 1
            fadd z16.s, z10.s, z16.s  // poly + 1
            fadd z17.s, z10.s, z17.s  // poly + 1
            fadd z18.s, z10.s, z18.s  // poly + 1
            fadd z19.s, z10.s, z19.s  // poly + 1

            fdivr z16.s, p0/m, z16.s, z10.s                                    // z16: 1/(poly+1)
            fdivr z17.s, p0/m, z17.s, z10.s                                    // z16: 1/(poly+1)
            fdivr z18.s, p0/m, z18.s, z10.s                                    // z16: 1/(poly+1)
            fdivr z19.s, p0/m, z19.s, z10.s                                    // z16: 1/(poly+1)

            // Stores 4 consecutive registers to the output
            .inst 0xa029c790  // st1w {z16.s-z19.s}, pn9, [x28, x9, LSL #2]

            incw x9, ALL, MUL #4
            b inner_body_start%=
inner_body_end%=:

inner_leftover_start%=:
            // Largely ordinary Sve code to handle taylor series 1/1+e^-x for leftover loop.
            whilelo p1.s, x9, %x[length]                                       // While x9<length
            b.none inner_leftover_end%=

            ld1w z12.s, p1/z, [x27, x9, LSL #2]                                // x12: input_data (LOADS POINTERS)
            fneg  z12.s, p1/m, z12.s

            mov z16.d, z5.d                                                    // z16: shift
            fcmlt p4.s, p1/z, z12.s, z9.s                                      // p4: underflow = x < min_input
            fcmlt p5.s, p1/z, z26.s, z12.s                                     // p5: overflow = x > max_input

            fmla z16.s, p1/m, z12.s, z6.s                                      // z16: z = shift + x * inv_ln2
            fsub z20.s, z16.s, z5.s                                            // z20: n = z - shift
            fmla z12.s, p1/m, z20.s, z7.s                                      // z12: r_hi = x + n * neg_ln2_hi
            fmla z12.s, p1/m, z20.s, z8.s                                      // z12: r = r_hi + n * neg_ln2_lo
            dup z10.s, #23                                                     // z10: 23
            urshl z16.s, p1/m, z16.s, z10.s                                    // z16: scale = z << 23 (2^n)
            fmul z20.s, z12.s, z0.s                                            // z20: p1 = r * c1
            mov z22.d, z1.d                                                    // z22: p23 = c2
            fmla z22.s, p1/m, z12.s, z2.s                                      // z22: p23 = c2 + r * c3
            mov z24.d, z3.d                                                    // z24: c4
            fmla z24.s, p1/m, z12.s, z4.s                                      // z24: p45 = c4 + r * c5
            fmul z12.s, z12.s, z12.s                                           // z12: r2 = r * r
            fmla z22.s, p1/m, z12.s, z24.s                                     // z22: p2345 = p23 + r2 * p45
            fmla z20.s, p1/m, z12.s, z22.s                                     // z20: p12345 = p1 + r2 * p2345
            fmla z16.s, p1/m, z20.s, z16.s                                     // z16: poly = scale + p12345 * scale
            dup z10.s, #0                                                      // z10: 0
            sel z16.s, p4, z10.s, z16.s                                        // z16: poly = underflow ? 0 : poly
            dup z10.s, w10
            sel z16.s, p5, z10.s, z16.s                                        // z16: poly = overflow ? inf : poly

            // 1 / 1+poly
            dup z10.s, w15                                                     // z10: 1
            fadd z16.s, z10.s, z16.s                                           // z16: z16 + 1
            fdivr z16.s, p0/m, z16.s, z10.s                                    // z16: 1/(poly+1)

            st1w z16.s, p1, [x28, x9, LSL #2]

            incw x9 // each word + 1
            b inner_leftover_start%=
inner_leftover_end%=:

            // ==================================================
            // Outer loop closing
            // ==================================================

            add x27, x27, %x[src_stride_1]
            add x28, x28, %x[dst_stride_1]
            b outer_loop_start%=
outer_loop_end%=:

            .inst 0xd503467f  // smstop
        )"
        :
        : [src] "r"(src), [dst] "r"(dst), [shape_1] "r"(shape[1]), [src_stride_1] "r"(src_strides[1]),
          [dst_stride_1] "r"(dst_strides[1]), [length] "r"(shape[0])
        : "cc", "memory",                                                      //
          "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p9",                //
          "x9", "x10", "x11", "x12", "x13", "x14", "x15",                      //
          "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", //
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",                //
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",              //
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"               //
    );
}

void sme2_fp32_logistic(const ITensor *in, ITensor *out, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_UNUSED(act_info);
    const auto *src_info = in->info();
    const auto *dst_info = out->info();

    const auto &src_strides = src_info->strides_in_bytes();
    const auto &dst_strides = dst_info->strides_in_bytes();

    // Iterator calculates pointer offsets and takes into account padding.
    Iterator input(in, window);
    Iterator output(out, window);

    // NOTE: This kernel uses collapsed 2D shapes.
    // The excecution window is expected to be pre-collapsed in kernel configure(...) function.
    const uintptr_t k_shape[] = {window.num_iterations(0), window.num_iterations(1)};

    const uintptr_t k_src_strides[] = {src_strides[0], src_strides[1]};
    const uintptr_t k_dst_strides[] = {dst_strides[0], dst_strides[1]};

    const auto *k_src = reinterpret_cast<const float *>(input.ptr());
    auto       *k_dst = reinterpret_cast<float *>(output.ptr());

    sme2_f32_logistic_kernel(k_src, k_dst, k_shape, k_src_strides, k_dst_strides);
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SME2
