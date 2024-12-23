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

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{

// Mul SME kernel
void sme2_q8_signed_mul_kernel( //
    const int8_t   *src,
    const int8_t   *weights,
    int8_t         *dst,
    const int16_t   offset_a,
    const int16_t   offset_b,
    const int16_t   offset_c,
    const float     multiplier, //  = (scale_a * scale_b * mul) / scale_c
    const uintptr_t win_shape[4],
    const uintptr_t src_strides[4],
    const uintptr_t wei_strides[4],
    const uintptr_t dst_strides[4])
{
    struct Args
    {
        uintptr_t     shape1;
        uintptr_t     shape2;
        uintptr_t     shape3;
        const int8_t *src;
        const int8_t *wei;
        int8_t       *dst;
        int           multiplier14p18;
        int           offsetC14p18;
        int16_t       offsetA;
        int16_t       offsetB;
    } args;

    // Constants used to express values in the 14p18 fixed point format
    constexpr int32_t two_pwr18i = 262144;
    constexpr float   two_pwr18f = 262144.f;

    args.shape1          = win_shape[1];
    args.shape2          = win_shape[2];
    args.shape3          = win_shape[3];
    args.src             = src;
    args.wei             = weights;
    args.dst             = dst;
    args.multiplier14p18 = static_cast<int>(multiplier * two_pwr18f);
    args.offsetC14p18    = static_cast<int>(offset_c * two_pwr18i);
    // Offsets a/b need to be negated as assembly kernel uses addition instructions where subtraction is needed.
    // Offset C is not negated as it needs to be added rather than subtracted.
    args.offsetA = offset_a * -1;
    args.offsetB = offset_b * -1;

    // Precondition:
    assert(src_strides[0] == sizeof(int8_t));
    assert(wei_strides[0] == sizeof(int8_t));
    assert(dst_strides[0] == sizeof(int8_t));
    __asm__ volatile(
        R"(
            .inst 0xd503477f  // smstart
            .inst 0x25207811  // ptrue pn9.b
            ptrue p0.b

            // ==================================================
            // 3D loop opening
            // ==================================================

            // ---------------------------------------------------------------- x13: body_length = (length / vl) * vl
            cntb x8, ALL, MUL #2 // x13 is vl (of 16 bit values)
            udiv x9, %x[length], x8 // length/vl
            mul x8, x8, x9 // x13 = vl * result


            // ---------------------------------------------------------------- x13: body_length = (length / vl) * vl
            ldr x10, [%[args_ptr], %[offset_shape_3]]
            ldr x11, [%[args_ptr], %[offset_src_ptr]]
            ldr x12, [%[args_ptr], %[offset_wei_ptr]]
            ldr x13, [%[args_ptr], %[offset_dst_ptr]]

            // Could potentially be replaced with explicit loads.
            ld1rh {z1.h}, p0/z, [%[args_ptr], %[offset_A_offset]]
            ld1rh {z2.h}, p0/z, [%[args_ptr], %[offset_B_offset]]
            ld1rw {z3.s}, p0/z, [%[args_ptr], %[multiplier_offset]]

1: // loop_3_start%=:
            // for index_3 in shape_3 downto 1
            cmp x10, #0
            b.eq 10f // loop_3_end%=
            sub x10, x10, #1

            ldr x14, [%[args_ptr], %[offset_shape_2]]
            mov x15, x11
            mov x16, x12
            mov x17, x13

2: // loop_2_start%=:
            // for index_2 in shape_2 downto 1
            cmp x14, #0
            b.eq 9f // loop_2_end%=
            sub x14, x14, #1

            ldr x7, [%[args_ptr], %[offset_shape_1]]
            mov x20, x15
            mov x21, x16
            mov x22, x17

3: // loop_1_start%=:
            // for index_1 in shape_2 downto 1
            cmp x7, #0
            b.eq 8f // loop_1_end%=
            sub x7, x7, #1

            mov x9, #0                                                         // x9: index/count

4: // inner_loop_body_start%=:
            cmp x9, x8
            b.eq 5f // inner_loop_body_end%=

            // WIDEN LOAD. LOAD 4 Z-REGS FOR BOTH A/B

            // NOTE: INSTEAD OF LOADING 4 LOAD 2 due to REG LIMITATIONS
            .inst 0xa0090684 	// ld1b	{z4.b-z5.b}, pn9/z, [x20, x9]
            .inst 0xa00906a6 	// ld1b	{z6.b-z7.b}, pn9/z, [x21, x9]

            // Widen to 16 bits
            .inst 0xc175e08c 	// sunpk	{z12.h-z15.h}, {z4.b-z5.b} // (a)
            .inst 0xc175e0d0 	// sunpk	{z16.h-z19.h}, {z6.b-z7.b} // (b)

            // Apply offset to all registers in 16-bit
            .inst 0xc161ab0c 	// add	{z12.h-z15.h}, {z12.h-z15.h}, z1.h //a
            .inst 0xc162ab10 	// add	{z16.h-z19.h}, {z16.h-z19.h}, z2.h //b

            // Widen to 32-bit now.
            // 12-19 are taken
            // 4-11 a, 20-27 b
            .inst 0xc1b5e184 	// sunpk	{z4.s-z7.s}, {z12.h-z13.h}      //a
            .inst 0xc1b5e1c8 	// sunpk	{z8.s-z11.s}, {z14.h-z15.h}
            .inst 0xc1b5e214 	// sunpk	{z20.s-z23.s}, {z16.h-z17.h}    //b
            .inst 0xc1b5e258 	// sunpk	{z24.s-z27.s}, {z18.h-z19.h}

            // Multiply a*b in int32
            // Output in z4-z11
            MUL z4.s, z4.s, z20.s
            MUL z5.s, z5.s, z21.s
            MUL z6.s, z6.s, z22.s
            MUL z7.s, z7.s, z23.s
            MUL z8.s, z8.s, z24.s
            MUL z9.s, z9.s, z25.s
            MUL z10.s, z10.s, z26.s
            MUL z11.s, z11.s, z27.s

            // offsets
            dup z12.s, %w[offset_C]
            dup z13.s, %w[offset_C]
            dup z14.s, %w[offset_C]
            dup z15.s, %w[offset_C]
            dup z16.s, %w[offset_C]
            dup z17.s, %w[offset_C]
            dup z18.s, %w[offset_C]
            dup z19.s, %w[offset_C]

            // MLA Fixed Point multiplication integer
            MLA z12.s, p0/m, z4.s, z3.s
            MLA z13.s, p0/m, z5.s, z3.s
            MLA z14.s, p0/m, z6.s, z3.s
            MLA z15.s, p0/m, z7.s, z3.s
            MLA z16.s, p0/m, z8.s, z3.s
            MLA z17.s, p0/m, z9.s, z3.s
            MLA z18.s, p0/m, z10.s, z3.s
            MLA z19.s, p0/m, z11.s, z3.s

            // Int32 to Int8 saturate
            .inst 0xc16eda05 	// sqrshr	z5.b, {z16.s-z19.s}, #18
            .inst 0xc16ed984 	// sqrshr	z4.b, {z12.s-z15.s}, #18
            // Store
            .inst 0xa02906c4 	// st1b	{z4.b-z5.b}, pn9, [x22, x9]

            incb x9, ALL, MUL #2
            b 4b // inner_loop_body_start%=
5: // inner_loop_body_end%=:

6: // inner_loop_leftover_start%=:
            whilelo p1.b, x9, %x[length]    // While x9<length
            b.none 7f // inner_loop_leftover_end%=

            // HANDLE MULTIPLICATION HERE
            ld1b z4.b, p1/z, [x20, x9]                                // z4: a input_data
            ld1b z5.b, p1/z, [x21, x9]                                // z5: b input_data

            // Widen register z4 (a)
            sunpklo z6.h, z4.b                                       // lower as 16 bits
            sunpkhi z7.h, z4.b                                       // upper as 16 bits

            // Widen register z5 (b)
            sunpklo z8.h, z5.b                                       // lower as 16 bits
            sunpkhi z9.h, z5.b                                       // upper as 16 bits

            // Apply offset in 16bit maths to all resulting vectors.
            add z6.h, z6.h, z1.h //a
            add z7.h, z7.h, z1.h
            add z8.h, z8.h, z2.h //b
            add z9.h, z9.h, z2.h

            // Widen a,b to 32-bit z-registers.
            // Multiply a and b and store result as 32 bit int.
            // a lower - 32-bit
            sunpklo z10.s, z6.h
            sunpkhi z11.s, z6.h
            // a upper - 32-bit
            sunpklo z12.s, z7.h
            sunpkhi z13.s, z7.h

            // b lower - 32-bit
            sunpklo z14.s, z8.h
            sunpkhi z15.s, z8.h
            // b upper - 32-bit
            sunpklo z16.s, z9.h
            sunpkhi z17.s, z9.h

            // offsets
            dup z4.s, %w[offset_C]
            dup z5.s, %w[offset_C]
            dup z6.s, %w[offset_C]
            dup z7.s, %w[offset_C]

            // Multiply a*b (lower) in int32
            MUL z10.s, z10.s, z14.s
            MUL z11.s, z11.s, z15.s

            // Multiply a*b (upper) in int32
            MUL z12.s, z12.s, z16.s
            MUL z13.s, z13.s, z17.s

            // Still int32 here.
            // Now MLA in fixed point
            MLA z4.s, p0/m, z10.s, z3.s
            MLA z5.s, p0/m, z11.s, z3.s
            MLA z6.s, p0/m, z12.s, z3.s
            MLA z7.s, p0/m, z13.s, z3.s

            // Right shift, no narrow
            LSR z20.s, z4.s, #8
            LSR z21.s, z5.s, #8
            LSR z22.s, z6.s, #8
            LSR z23.s, z7.s, #8

            // Right shift rounding (lower)
            // Do not saturate.
            RSHRNB z20.h, z20.s, #8
            RSHRNB z21.h, z21.s, #8
            UZP1 z25.h, z20.h, z21.h
            // Right shift upper.
            RSHRNB z22.h, z22.s, #8
            RSHRNB z23.h, z23.s, #8
            UZP1 z26.h, z22.h, z23.h

            // Shift again to 8 bit both vectors. Recombine.
            SQRSHRNB z25.b, z25.h, #2
            SQRSHRNB z26.b, z26.h, #2
            UZP1 z27.b, z25.b, z26.b

            st1b z27.b, p1, [x22, x9]

            incb x9 // x9 : x9 += sizeof(element) * predicate_count
            b 6b // inner_loop_leftover_start%=
7: // inner_loop_leftover_end%=:

            // ==================================================
            // 3D loop closing
            // ==================================================

            add x20, x20, %[src_stride_1]
            add x21, x21, %[wei_stride_1]
            add x22, x22, %[dst_stride_1]
            b 3b // loop_1_start%=
8: // loop_1_end%=:

            add x15, x15, %[src_stride_2]
            add x16, x16, %[wei_stride_2]
            add x17, x17, %[dst_stride_2]
            b 2b // loop_2_start%=
9: // loop_2_end%=:

            add x11, x11, %[src_stride_3]
            add x12, x12, %[wei_stride_3]
            add x13, x13, %[dst_stride_3]
            b 1b // loop_3_start%=
10: // loop_3_end%=:

            .inst 0xd503467f  // smstop
        )"
        :
        : // The following arguments are loaded via arg ptr values and a constant offset.
        [args_ptr] "r"(&args), [offset_src_ptr] "I"(offsetof(Args, src)), [offset_wei_ptr] "I"(offsetof(Args, wei)),
        [offset_dst_ptr] "I"(offsetof(Args, dst)), [offset_shape_1] "I"(offsetof(Args, shape1)),
        [offset_shape_2] "I"(offsetof(Args, shape2)), [offset_shape_3] "I"(offsetof(Args, shape3)),
        [multiplier_offset] "I"(offsetof(Args, multiplier14p18)), //
        [offset_A_offset] "I"(offsetof(Args, offsetA)),           //
        [offset_B_offset] "I"(offsetof(Args, offsetB)),           //
        // Use registers for efficiency sake.
        [src_stride_1] "r"(src_strides[1]), [src_stride_2] "r"(src_strides[2]), [src_stride_3] "r"(src_strides[3]),
        [wei_stride_1] "r"(wei_strides[1]), [wei_stride_2] "r"(wei_strides[2]), [wei_stride_3] "r"(wei_strides[3]),
        [dst_stride_1] "r"(dst_strides[1]), [dst_stride_2] "r"(dst_strides[2]), [dst_stride_3] "r"(dst_strides[3]),
        [offset_C] "r"(args.offsetC14p18), //
        [length] "r"(win_shape[0])
        : "cc", "memory", //
          "p0", "p1", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",         //
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",   //
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", //
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"  //
    );
}

void sme2_q8_signed_mul(const ITensor *in0, const ITensor *in1, ITensor *out, const Window &window, const float scale)
{
    const auto *src_info  = in0->info();
    const auto *src2_info = in1->info();
    const auto *dst_info  = out->info();

    const UniformQuantizationInfo src_q_info  = src_info->quantization_info().uniform();
    const UniformQuantizationInfo src2_q_info = src2_info->quantization_info().uniform();
    const UniformQuantizationInfo dst_q_info  = dst_info->quantization_info().uniform();

    const auto &src_strides_bytes = src_info->strides_in_bytes();
    const auto &wei_strides_bytes = src2_info->strides_in_bytes();
    const auto &dst_strides_bytes = dst_info->strides_in_bytes();

    // NOTE: This kernel does not support shapes above 4D (Unless excecution window has been collapsed)
    assert(window.num_iterations(4) == 1 && window.num_iterations(5) == 1);

    // Note : The window is expected to handle y-broadcasting by setting relevant strides to 0.
    const uintptr_t shape[] = {
        window.num_iterations(0),
        window.num_iterations(1),
        window.num_iterations(2),
        window.num_iterations(3),
    };

    Window input1_win = window.broadcast_if_dimension_le_one(src_info->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src2_info->tensor_shape());

    // First dim is always datasize. If broadcasting in other dims, set stride to 0.
    uintptr_t src_strides[] = {src_strides_bytes[0], (input1_win.is_broadcasted(1)) ? 0 : src_strides_bytes[1],
                               (input1_win.is_broadcasted(2)) ? 0 : src_strides_bytes[2],
                               (input1_win.is_broadcasted(3)) ? 0 : src_strides_bytes[3]};
    uintptr_t wei_strides[] = {wei_strides_bytes[0], (input2_win.is_broadcasted(1)) ? 0 : wei_strides_bytes[1],
                               (input2_win.is_broadcasted(2)) ? 0 : wei_strides_bytes[2],
                               (input2_win.is_broadcasted(3)) ? 0 : wei_strides_bytes[3]};

    const uintptr_t dst_strides[] = {
        dst_strides_bytes[0],
        dst_strides_bytes[1],
        dst_strides_bytes[2],
        dst_strides_bytes[3],
    };

    const uintptr_t src_offset = window[0].start() * src_strides[0] + window[1].start() * src_strides[1] +
                                 window[2].start() * src_strides[2] + window[3].start() * src_strides[3] +
                                 in0->info()->offset_first_element_in_bytes();
    const uintptr_t src2_offset = window[0].start() * wei_strides[0] + window[1].start() * wei_strides[1] +
                                  window[2].start() * wei_strides[2] + window[3].start() * wei_strides[3] +
                                  in1->info()->offset_first_element_in_bytes();
    const uintptr_t dst_offset = window[0].start() * dst_strides[0] + window[1].start() * dst_strides[1] +
                                 window[2].start() * dst_strides[2] + window[3].start() * dst_strides[3] +
                                 out->info()->offset_first_element_in_bytes();

    const auto *src  = reinterpret_cast<const int8_t *>(in0->buffer() + src_offset);
    const auto *src2 = reinterpret_cast<const int8_t *>(in1->buffer() + src2_offset);
    auto       *dst  = reinterpret_cast<int8_t *>(out->buffer() + dst_offset);

    // Calculate or retrieve necessary offsets/scale values.
    const int16_t offset_a   = src_q_info.offset;
    const int16_t offset_b   = src2_q_info.offset;
    float         multiplier = (src_q_info.scale * src2_q_info.scale * scale) / dst_q_info.scale;

    sme2_q8_signed_mul_kernel(src, src2, dst, offset_a, offset_b, dst_q_info.offset, multiplier, shape, src_strides,
                              wei_strides, dst_strides);
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SME2
