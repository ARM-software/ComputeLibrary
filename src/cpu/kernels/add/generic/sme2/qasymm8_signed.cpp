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

// Add SME kernel
void sme2_q8_signed_add_kernel( //
    const int8_t   *src0,
    const int8_t   *src1,
    int8_t         *dst,
    const float     scale_0,
    const float     scale_1,
    const float     offset,
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
        const int8_t *src_0;
        const int8_t *src_1;
        int8_t       *dst;
        int32_t       scale_0_5p11;
        int32_t       scale_1_5p11;
        int32_t       offset_21p11;
    } args;

    // Constant used to express values in the 21p11 and 5p11 fixed point format
    constexpr float _2pow11 = 2048;

    args.shape1       = win_shape[1];
    args.shape2       = win_shape[2];
    args.shape3       = win_shape[3];
    args.src_0        = src0;
    args.src_1        = src1;
    args.dst          = dst;
    args.scale_0_5p11 = static_cast<int32_t>(static_cast<int16_t>(support::cpp11::lround(scale_0 * _2pow11)));
    args.scale_1_5p11 = static_cast<int32_t>(static_cast<int16_t>(support::cpp11::lround(scale_1 * _2pow11)));
    args.offset_21p11 = static_cast<int32_t>(support::cpp11::lround(offset * _2pow11));

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

            // ---------------------------------------------------------------- x8: body_length = (length / vl) * vl
            cntb x8, ALL, MUL #2 // x8 is vl (of 8 bit values)
            udiv x9, %x[length], x8 // length/vl
            mul x8, x8, x9 // x8 = vl * result

            ldr x10, [%[args_ptr], %[offset_shape_3]]
            ldr x11, [%[args_ptr], %[offset_src_ptr]]
            ldr x12, [%[args_ptr], %[offset_wei_ptr]]
            ldr x13, [%[args_ptr], %[offset_dst_ptr]]

            // Could potentially be replaced with explicit loads.
            ld1rw {z1.s}, p0/z, [%[args_ptr], %[scale_0_offset]]
            ld1rw {z2.s}, p0/z, [%[args_ptr], %[scale_1_offset]]
            ld1rw {z3.s}, p0/z, [%[args_ptr], %[offset_offset]]

6: // loop_3_start
            // for index_3 in shape_3 downto 1
            cmp x10, #0
            b.eq 1f // loop_3_end
            sub x10, x10, #1

            ldr x14, [%[args_ptr], %[offset_shape_2]]
            mov x15, x11
            mov x16, x12
            mov x17, x13

5: // loop_2_start
            // for index_2 in shape_2 downto 1
            cmp x14, #0
            b.eq 7f // loop_2_end
            sub x14, x14, #1

            ldr x7, [%[args_ptr], %[offset_shape_1]]
            mov x20, x15
            mov x21, x16
            mov x22, x17

4: // loop_1_start
            // for index_1 in shape_2 downto 1
            cmp x7, #0
            b.eq 3f // loop_1_end
            sub x7, x7, #1

            mov x9, #0                                                         // x9: index/count

inner_loop_body_start%=:
            cmp x9, x8
            b.eq 8f // inner_loop_body_end

            /*
            Two - instead of the maximal four - registers of each input are processed per loop iteration
            due to the need for at least 32 registers just for the data processing which leaves no space
            for the registers that contain the pre-loop loaded constants.
            Once the would be 4 registers are expanded into 16 as the data goes from 8 to 32-bit, the
            same number of registers (another 16) is needed to accumulate onto the offset constant for
            each of those 16 lanes. One advantage of only processing two registers per loop is that more
            of the elements to be processed will be in this vectorised loop instead of the left-over one.
            */

            // Load src0
            .inst 0xa0090684    // ld1b     {z4.b-z5.b}, pn9/z, [x20, x9]

            // Widen src0 to 16 bits
            .inst 0xc175e08c    // sunpk    {z12.h-z15.h}, {z4.b-z5.b}

            // Widen src0 to 32-bits
            .inst 0xc1b5e184    // sunpk    {z4.s-z7.s}, {z12.h-z13.h}
            .inst 0xc1b5e1c8    // sunpk    {z8.s-z11.s}, {z14.h-z15.h}

            // Duplicate the offset value into registers for all the values to be processed
            mov z16.d, z3.d
            mov z17.d, z3.d
            mov z18.d, z3.d
            mov z19.d, z3.d
            mov z20.d, z3.d
            mov z21.d, z3.d
            mov z22.d, z3.d
            mov z23.d, z3.d

            // MLA Fixed Point multiplication and accumulation integer
            // Multiply src0 by scale_0 (z1) and add offset
            mla z16.s, p0/m, z4.s, z1.s
            mla z17.s, p0/m, z5.s, z1.s
            mla z18.s, p0/m, z6.s, z1.s
            mla z19.s, p0/m, z7.s, z1.s
            mla z20.s, p0/m, z8.s, z1.s
            mla z21.s, p0/m, z9.s, z1.s
            mla z22.s, p0/m, z10.s, z1.s
            mla z23.s, p0/m, z11.s, z1.s

            //Load src1 into the same registers that were used for src0 since they are no longer needed
            .inst 0xa00906a4    // ld1b     {z4.b-z5.b}, pn9/z, [x21, x9]

            // Widen src1 to 16 bits
            .inst 0xc175e08c    // sunpk    {z12.h-z15.h}, {z4.b-z5.b}

            // Widen src1 32-bits
            .inst 0xc1b5e184    // sunpk    {z4.s-z7.s}, {z12.h-z13.h}
            .inst 0xc1b5e1c8    // sunpk    {z8.s-z11.s}, {z14.h-z15.h}

            // MLA Fixed Point multiplication and accumulation integer
            // Multiply src1 by scale_1 (z2) and accumulate into registers containing src0*scale_0 + offset
            mla z16.s, p0/m, z4.s, z2.s
            mla z17.s, p0/m, z5.s, z2.s
            mla z18.s, p0/m, z6.s, z2.s
            mla z19.s, p0/m, z7.s, z2.s
            mla z20.s, p0/m, z8.s, z2.s
            mla z21.s, p0/m, z9.s, z2.s
            mla z22.s, p0/m, z10.s, z2.s
            mla z23.s, p0/m, z11.s, z2.s

            // Int32 to Int8 saturate
            .inst 0xc175da85    // sqrshr   z5.b, {z20.s-z23.s}, #11
            .inst 0xc175da04    // sqrshr   z4.b, {z16.s-z19.s}, #11
            // Store
            .inst 0xa02906c4    // st1b     {z4.b-z5.b}, pn9, [x22, x9]

            incb x9, ALL, MUL #2
            b inner_loop_body_start%=
8: // inner_loop_body_end

inner_loop_leftover_start%=:
            whilelo p1.b, x9, %x[length]    // While x9<length
            b.none 2f // inner_loop_leftover_end

            // Load src0
            ld1b z4.b, p1/z, [x20, x9]

            // Widen src0 to 16 bits
            sunpklo z6.h, z4.b                                       // lower as 16 bits
            sunpkhi z7.h, z4.b                                       // upper as 16 bits

            // Widen src0 to 32 bits
            // Lower - 32-bit
            sunpklo z10.s, z6.h
            sunpkhi z11.s, z6.h
            // Upper - 32-bit
            sunpklo z12.s, z7.h
            sunpkhi z13.s, z7.h

            // Duplicate the offset value into registers for all the values to be processed
            mov z14.d, z3.d
            mov z15.d, z3.d
            mov z16.d, z3.d
            mov z17.d, z3.d

            // MLA Fixed Point multiplication and accumulation integer
            // Multiply src0 by scale_0 (z1) and add offset
            mla z14.s, p0/m, z10.s, z1.s
            mla z15.s, p0/m, z11.s, z1.s
            mla z16.s, p0/m, z12.s, z1.s
            mla z17.s, p0/m, z13.s, z1.s

            // Load src1
            ld1b z5.b, p1/z, [x21, x9]                                // z5: b input_data

            // Widen src1 to 16 bits
            sunpklo z8.h, z5.b                                       // lower as 16 bits
            sunpkhi z9.h, z5.b                                       // upper as 16 bits

            // Widen src1 to 32 bits
            // Lower - 32-bit
            sunpklo z10.s, z8.h
            sunpkhi z11.s, z8.h
            // Upper - 32-bit
            sunpklo z12.s, z9.h
            sunpkhi z13.s, z9.h

            // MLA Fixed Point multiplication and accumulation integer
            // Multiply src1 by scale_1 (z2) and accumulate into registers containing src0*scale_0 + offset
            mla z14.s, p0/m, z10.s, z2.s
            mla z15.s, p0/m, z11.s, z2.s
            mla z16.s, p0/m, z12.s, z2.s
            mla z17.s, p0/m, z13.s, z2.s

            // Right shift rounding (lower)
            rshrnb z20.h, z14.s, #8
            rshrnb z21.h, z15.s, #8
            uzp1 z25.h, z20.h, z21.h
            // Right shift upper.
            rshrnb z22.h, z16.s, #8
            rshrnb z23.h, z17.s, #8
            uzp1 z26.h, z22.h, z23.h

            // Shift again to 8 bit both vectors. Recombine.
            sqrshrnb z25.b, z25.h, #3
            sqrshrnb z26.b, z26.h, #3
            uzp1 z27.b, z25.b, z26.b

            st1b z27.b, p1, [x22, x9]

            incb x9 // x9 : x9 += sizeof(element) * predicate_count
            b inner_loop_leftover_start%=
2: // inner_loop_leftover_end

            // ==================================================
            // 3D loop closing
            // ==================================================

            add x20, x20, %[src_stride_1]
            add x21, x21, %[wei_stride_1]
            add x22, x22, %[dst_stride_1]
            b 4b // loop_1_start
3: // loop_1_end

            add x15, x15, %[src_stride_2]
            add x16, x16, %[wei_stride_2]
            add x17, x17, %[dst_stride_2]
            b 5b // loop_2_start
7: // loop_2_end

            add x11, x11, %[src_stride_3]
            add x12, x12, %[wei_stride_3]
            add x13, x13, %[dst_stride_3]
            b 6b // loop_3_start
1: // loop_3_end

            .inst 0xd503467f  // smstop
        )"
        :
        : // The following arguments are loaded via arg ptr values and a constant offset.
        [args_ptr] "r"(&args), [offset_src_ptr] "I"(offsetof(Args, src_0)), [offset_wei_ptr] "I"(offsetof(Args, src_1)),
        [offset_dst_ptr] "I"(offsetof(Args, dst)), [offset_shape_1] "I"(offsetof(Args, shape1)),
        [offset_shape_2] "I"(offsetof(Args, shape2)), [offset_shape_3] "I"(offsetof(Args, shape3)),
        [scale_0_offset] "I"(offsetof(Args, scale_0_5p11)), //
        [scale_1_offset] "I"(offsetof(Args, scale_1_5p11)), //
        [offset_offset] "I"(offsetof(Args, offset_21p11)),  //
        // Use registers for efficiency sake.
        [src_stride_1] "r"(src_strides[1]), [src_stride_2] "r"(src_strides[2]), [src_stride_3] "r"(src_strides[3]),
        [wei_stride_1] "r"(wei_strides[1]), [wei_stride_2] "r"(wei_strides[2]), [wei_stride_3] "r"(wei_strides[3]),
        [dst_stride_1] "r"(dst_strides[1]), [dst_stride_2] "r"(dst_strides[2]), [dst_stride_3] "r"(dst_strides[3]),
        [length] "r"(win_shape[0])
        : "cc", "memory", //
          "p0", "p1", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22",
          "z1", "z2", "z3", "z4", "z5", "z6", "z7",               //
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",   //
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", //
          "z25", "z26", "z27"                                     //
    );
}

void add_qasymm8_signed_sme2(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);
    const auto *src0_info = src0->info();
    const auto *src1_info = src1->info();
    const auto *dst_info  = dst->info();

    const UniformQuantizationInfo src0_q_info = src0_info->quantization_info().uniform();
    const UniformQuantizationInfo src1_q_info = src1_info->quantization_info().uniform();
    const UniformQuantizationInfo dst_q_info  = dst_info->quantization_info().uniform();

    const auto &src0_strides_bytes = src0_info->strides_in_bytes();
    const auto &src1_strides_bytes = src1_info->strides_in_bytes();
    const auto &dst_strides_bytes  = dst_info->strides_in_bytes();

    // NOTE: This kernel does not support shapes above 4D (Unless excecution window has been collapsed)
    assert(window.num_iterations(4) == 1 && window.num_iterations(5) == 1);

    // Note : The window is expected to handle broadcasting in higher axis than x by setting relevant strides to 0.
    const uintptr_t shape[] = {
        window.num_iterations(0),
        window.num_iterations(1),
        window.num_iterations(2),
        window.num_iterations(3),
    };

    Window input0_win = window.broadcast_if_dimension_le_one(src0_info->tensor_shape());
    Window input1_win = window.broadcast_if_dimension_le_one(src1_info->tensor_shape());

    // First dim is always datasize. If broadcasting in other dims, set stride to 0.
    uintptr_t src0_strides[] = {src0_strides_bytes[0], (input0_win.is_broadcasted(1)) ? 0 : src0_strides_bytes[1],
                                (input0_win.is_broadcasted(2)) ? 0 : src0_strides_bytes[2],
                                (input0_win.is_broadcasted(3)) ? 0 : src0_strides_bytes[3]};
    uintptr_t src1_strides[] = {src1_strides_bytes[0], (input1_win.is_broadcasted(1)) ? 0 : src1_strides_bytes[1],
                                (input1_win.is_broadcasted(2)) ? 0 : src1_strides_bytes[2],
                                (input1_win.is_broadcasted(3)) ? 0 : src1_strides_bytes[3]};

    const uintptr_t dst_strides[] = {
        dst_strides_bytes[0],
        dst_strides_bytes[1],
        dst_strides_bytes[2],
        dst_strides_bytes[3],
    };

    const uintptr_t src0_offset = window[0].start() * src0_strides[0] + window[1].start() * src0_strides[1] +
                                  window[2].start() * src0_strides[2] + window[3].start() * src0_strides[3] +
                                  src0->info()->offset_first_element_in_bytes();
    const uintptr_t src1_offset = window[0].start() * src1_strides[0] + window[1].start() * src1_strides[1] +
                                  window[2].start() * src1_strides[2] + window[3].start() * src1_strides[3] +
                                  src1->info()->offset_first_element_in_bytes();
    const uintptr_t dst_offset = window[0].start() * dst_strides[0] + window[1].start() * dst_strides[1] +
                                 window[2].start() * dst_strides[2] + window[3].start() * dst_strides[3] +
                                 dst->info()->offset_first_element_in_bytes();

    const auto *src0_ptr = reinterpret_cast<const int8_t *>(src0->buffer() + src0_offset);
    const auto *src1_ptr = reinterpret_cast<const int8_t *>(src1->buffer() + src1_offset);
    auto       *dst_ptr  = reinterpret_cast<int8_t *>(dst->buffer() + dst_offset);

    // Calculate or retrieve necessary offsets/scale values.
    const int32_t offset_a = src0_q_info.offset;
    const int32_t offset_b = src1_q_info.offset;
    const float   scale0   = src0_q_info.scale / dst_q_info.scale;
    const float   scale1   = src1_q_info.scale / dst_q_info.scale;
    const float   offset   = static_cast<float>(dst_q_info.offset) - static_cast<float>(offset_a) * scale0 -
                         static_cast<float>(offset_b) * scale1;

    sme2_q8_signed_add_kernel(src0_ptr, src1_ptr, dst_ptr, scale0, scale1, offset, shape, src0_strides, src1_strides,
                              dst_strides);
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SME2
