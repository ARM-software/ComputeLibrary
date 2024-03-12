/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef ARM_COMPUTE_ENABLE_SVE

#include <cstddef>
#include "../../bfloat.hpp"

namespace arm_gemm {

void sve_ffinterleaved_bf16fp32_mmla_8x3VL(
    const bfloat16 *Apanel,
    const bfloat16 *Bpanel,
    size_t B_stride,
    float *Cpanel,
    int ablocks,
    size_t N,
    int K) {

    struct KernelArgs {
        size_t K = {};
        const bfloat16 *Bpanel = {};
        size_t N = {};
        size_t B_stride = {};
        const bfloat16 *cur_B_ptr = {};
    } ka;

    ka.K = (K/4) - 1;
    ka.Bpanel = Bpanel;
    ka.N = N;
    ka.B_stride = B_stride;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "1:"  // Height loop
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "ldr x26, [%x[args_ptr], %[offsetof_N]]\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "mov x25, %x[Apanel]\n"
      "2:"  // Width loop
      "ldr x24, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_B_stride]]\n"
      "cntw x23, ALL, MUL #2\n"
      "add x22, x24, x20, LSL #1\n"
      "add x21, x22, x20, LSL #1\n"
      "add x20, x21, x20, LSL #1\n"
      "cmp x26, x23\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "mov %x[Apanel], x25\n"
      "bgt 3f\n"
      "decw x23\n"
      "cmp x26, x23\n"
      "mov x21, x24\n"
      "bgt 3f\n"
      "mov x22, x24\n"
      "3:"  // B setup done
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "cmp x20, #0x2\n"
      "mov z8.b, #0x0\n"
      "mov z9.b, #0x0\n"
      "mov z10.b, #0x0\n"
      "ld1h { z4.h }, p0/Z, [x24]\n"
      "mov z11.b, #0x0\n"
      "mov z12.b, #0x0\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "mov z13.b, #0x0\n"
      "mov z14.b, #0x0\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #16]\n"
      "mov z15.b, #0x0\n"
      "mov z16.b, #0x0\n"
      "ld1h { z5.h }, p0/Z, [x24, #1, MUL VL]\n"
      "mov z17.b, #0x0\n"
      "mov z18.b, #0x0\n"
      "ld1rqh { z2.h }, p0/Z, [%x[Apanel], #32]\n"
      "mov z19.b, #0x0\n"
      "mov z20.b, #0x0\n"
      "addvl x24, x24, #2\n"
      "mov z21.b, #0x0\n"
      "mov z22.b, #0x0\n"
      "add %x[Apanel], %x[Apanel], #0x30\n"
      "mov z23.b, #0x0\n"
      "mov z24.b, #0x0\n"
      "mov z25.b, #0x0\n"
      "mov z26.b, #0x0\n"
      "mov z27.b, #0x0\n"
      "mov z28.b, #0x0\n"
      "mov z29.b, #0x0\n"
      "mov z30.b, #0x0\n"
      "mov z31.b, #0x0\n"
      "blt 5f\n"
      "4:"  // main loop head
      "ld1rqh { z6.h }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x6464e408  // bfmmla z8.s, z0.h, z4.h\n"
      ".inst 0x6465e40b  // bfmmla z11.s, z0.h, z5.h\n"
      ".inst 0x6464e42e  // bfmmla z14.s, z1.h, z4.h\n"
      ".inst 0x6465e431  // bfmmla z17.s, z1.h, z5.h\n"
      "ld1h { z7.h }, p0/Z, [x22]\n"
      ".inst 0x6464e454  // bfmmla z20.s, z2.h, z4.h\n"
      ".inst 0x6465e457  // bfmmla z23.s, z2.h, z5.h\n"
      "ld1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
      ".inst 0x6464e4da  // bfmmla z26.s, z6.h, z4.h\n"
      ".inst 0x6465e4dd  // bfmmla z29.s, z6.h, z5.h\n"
      "ld1h { z5.h }, p0/Z, [x21]\n"
      "ld1h { z4.h }, p0/Z, [x21, #1, MUL VL]\n"
      ".inst 0x6467e409  // bfmmla z9.s, z0.h, z7.h\n"
      ".inst 0x6463e40c  // bfmmla z12.s, z0.h, z3.h\n"
      ".inst 0x6467e42f  // bfmmla z15.s, z1.h, z7.h\n"
      ".inst 0x6463e432  // bfmmla z18.s, z1.h, z3.h\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x6467e455  // bfmmla z21.s, z2.h, z7.h\n"
      ".inst 0x6463e458  // bfmmla z24.s, z2.h, z3.h\n"
      "cmp x20, #0x2\n"
      ".inst 0x6467e4db  // bfmmla z27.s, z6.h, z7.h\n"
      ".inst 0x6463e4de  // bfmmla z30.s, z6.h, z3.h\n"
      "ld1h { z3.h }, p0/Z, [x24]\n"
      ".inst 0x6465e40a  // bfmmla z10.s, z0.h, z5.h\n"
      ".inst 0x6464e40d  // bfmmla z13.s, z0.h, z4.h\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel], #16]\n"
      ".inst 0x6465e430  // bfmmla z16.s, z1.h, z5.h\n"
      ".inst 0x6464e433  // bfmmla z19.s, z1.h, z4.h\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #32]\n"
      ".inst 0x6465e456  // bfmmla z22.s, z2.h, z5.h\n"
      ".inst 0x6464e459  // bfmmla z25.s, z2.h, z4.h\n"
      "ld1h { z7.h }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0x6465e4dc  // bfmmla z28.s, z6.h, z5.h\n"
      ".inst 0x6464e4df  // bfmmla z31.s, z6.h, z4.h\n"
      "ld1rqh { z5.h }, p0/Z, [%x[Apanel], #48]\n"
      "ld1rqh { z6.h }, p0/Z, [%x[Apanel], #64]\n"
      ".inst 0x6463e408  // bfmmla z8.s, z0.h, z3.h\n"
      ".inst 0x6467e40b  // bfmmla z11.s, z0.h, z7.h\n"
      ".inst 0x6463e42e  // bfmmla z14.s, z1.h, z3.h\n"
      ".inst 0x6467e431  // bfmmla z17.s, z1.h, z7.h\n"
      "ld1h { z2.h }, p0/Z, [x22, #2, MUL VL]\n"
      ".inst 0x6463e4b4  // bfmmla z20.s, z5.h, z3.h\n"
      ".inst 0x6467e4b7  // bfmmla z23.s, z5.h, z7.h\n"
      "ld1h { z4.h }, p0/Z, [x22, #3, MUL VL]\n"
      ".inst 0x6463e4da  // bfmmla z26.s, z6.h, z3.h\n"
      ".inst 0x6467e4dd  // bfmmla z29.s, z6.h, z7.h\n"
      "ld1h { z3.h }, p0/Z, [x21, #2, MUL VL]\n"
      "ld1h { z7.h }, p0/Z, [x21, #3, MUL VL]\n"
      ".inst 0x6462e409  // bfmmla z9.s, z0.h, z2.h\n"
      ".inst 0x6464e40c  // bfmmla z12.s, z0.h, z4.h\n"
      ".inst 0x6462e42f  // bfmmla z15.s, z1.h, z2.h\n"
      ".inst 0x6464e432  // bfmmla z18.s, z1.h, z4.h\n"
      "addvl x22, x22, #4\n"
      ".inst 0x6462e4b5  // bfmmla z21.s, z5.h, z2.h\n"
      ".inst 0x6464e4b8  // bfmmla z24.s, z5.h, z4.h\n"
      "addvl x21, x21, #4\n"
      ".inst 0x6462e4db  // bfmmla z27.s, z6.h, z2.h\n"
      ".inst 0x6464e4de  // bfmmla z30.s, z6.h, z4.h\n"
      "ld1h { z4.h }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0x6463e40a  // bfmmla z10.s, z0.h, z3.h\n"
      ".inst 0x6467e40d  // bfmmla z13.s, z0.h, z7.h\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel], #80]\n"
      ".inst 0x6463e430  // bfmmla z16.s, z1.h, z3.h\n"
      ".inst 0x6467e433  // bfmmla z19.s, z1.h, z7.h\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #96]\n"
      ".inst 0x6463e4b6  // bfmmla z22.s, z5.h, z3.h\n"
      ".inst 0x6467e4b9  // bfmmla z25.s, z5.h, z7.h\n"
      "ld1h { z5.h }, p0/Z, [x24, #3, MUL VL]\n"
      ".inst 0x6463e4dc  // bfmmla z28.s, z6.h, z3.h\n"
      ".inst 0x6467e4df  // bfmmla z31.s, z6.h, z7.h\n"
      "ld1rqh { z2.h }, p0/Z, [%x[Apanel], #112]\n"
      "add %x[Apanel], %x[Apanel], #0x80\n"
      "addvl x24, x24, #4\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "ld1rqh { z7.h }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x6464e408  // bfmmla z8.s, z0.h, z4.h\n"
      ".inst 0x6465e40b  // bfmmla z11.s, z0.h, z5.h\n"
      ".inst 0x6464e42e  // bfmmla z14.s, z1.h, z4.h\n"
      ".inst 0x6465e431  // bfmmla z17.s, z1.h, z5.h\n"
      "ld1h { z6.h }, p0/Z, [x22]\n"
      ".inst 0x6464e454  // bfmmla z20.s, z2.h, z4.h\n"
      ".inst 0x6465e457  // bfmmla z23.s, z2.h, z5.h\n"
      "ld1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
      ".inst 0x6464e4fa  // bfmmla z26.s, z7.h, z4.h\n"
      ".inst 0x6465e4fd  // bfmmla z29.s, z7.h, z5.h\n"
      "ld1h { z5.h }, p0/Z, [x21]\n"
      "ld1h { z4.h }, p0/Z, [x21, #1, MUL VL]\n"
      ".inst 0x6466e409  // bfmmla z9.s, z0.h, z6.h\n"
      ".inst 0x6463e40c  // bfmmla z12.s, z0.h, z3.h\n"
      ".inst 0x6466e42f  // bfmmla z15.s, z1.h, z6.h\n"
      ".inst 0x6463e432  // bfmmla z18.s, z1.h, z3.h\n"
      "add %x[Apanel], %x[Apanel], #0x10\n"
      ".inst 0x6466e455  // bfmmla z21.s, z2.h, z6.h\n"
      ".inst 0x6463e458  // bfmmla z24.s, z2.h, z3.h\n"
      "addvl x22, x22, #2\n"
      ".inst 0x6466e4fb  // bfmmla z27.s, z7.h, z6.h\n"
      ".inst 0x6463e4fe  // bfmmla z30.s, z7.h, z3.h\n"
      "addvl x21, x21, #2\n"
      ".inst 0x6465e40a  // bfmmla z10.s, z0.h, z5.h\n"
      ".inst 0x6464e40d  // bfmmla z13.s, z0.h, z4.h\n"
      ".inst 0x6465e430  // bfmmla z16.s, z1.h, z5.h\n"
      ".inst 0x6464e433  // bfmmla z19.s, z1.h, z4.h\n"
      ".inst 0x6465e456  // bfmmla z22.s, z2.h, z5.h\n"
      ".inst 0x6464e459  // bfmmla z25.s, z2.h, z4.h\n"
      ".inst 0x6465e4fc  // bfmmla z28.s, z7.h, z5.h\n"
      ".inst 0x6464e4ff  // bfmmla z31.s, z7.h, z4.h\n"
      "cbz x20, 6f\n"
      "ld1h { z1.h }, p0/Z, [x24]\n"
      "ld1rqh { z7.h }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x6461e4e8  // bfmmla z8.s, z7.h, z1.h\n"
      "ld1rqh { z6.h }, p0/Z, [%x[Apanel], #16]\n"
      "ld1h { z0.h }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0x6460e4eb  // bfmmla z11.s, z7.h, z0.h\n"
      "ld1rqh { z5.h }, p0/Z, [%x[Apanel], #32]\n"
      "ld1rqh { z4.h }, p0/Z, [%x[Apanel], #48]\n"
      ".inst 0x6461e4ce  // bfmmla z14.s, z6.h, z1.h\n"
      ".inst 0x6460e4d1  // bfmmla z17.s, z6.h, z0.h\n"
      ".inst 0x6461e4b4  // bfmmla z20.s, z5.h, z1.h\n"
      "ld1h { z3.h }, p0/Z, [x22]\n"
      ".inst 0x6460e4b7  // bfmmla z23.s, z5.h, z0.h\n"
      ".inst 0x6461e49a  // bfmmla z26.s, z4.h, z1.h\n"
      "ld1h { z2.h }, p0/Z, [x22, #1, MUL VL]\n"
      ".inst 0x6460e49d  // bfmmla z29.s, z4.h, z0.h\n"
      "ld1h { z1.h }, p0/Z, [x21]\n"
      "ld1h { z0.h }, p0/Z, [x21, #1, MUL VL]\n"
      ".inst 0x6463e4e9  // bfmmla z9.s, z7.h, z3.h\n"
      ".inst 0x6462e4ec  // bfmmla z12.s, z7.h, z2.h\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x6463e4cf  // bfmmla z15.s, z6.h, z3.h\n"
      ".inst 0x6462e4d2  // bfmmla z18.s, z6.h, z2.h\n"
      ".inst 0x6463e4b5  // bfmmla z21.s, z5.h, z3.h\n"
      ".inst 0x6462e4b8  // bfmmla z24.s, z5.h, z2.h\n"
      ".inst 0x6463e49b  // bfmmla z27.s, z4.h, z3.h\n"
      ".inst 0x6462e49e  // bfmmla z30.s, z4.h, z2.h\n"
      ".inst 0x6461e4ea  // bfmmla z10.s, z7.h, z1.h\n"
      ".inst 0x6460e4ed  // bfmmla z13.s, z7.h, z0.h\n"
      ".inst 0x6461e4d0  // bfmmla z16.s, z6.h, z1.h\n"
      ".inst 0x6460e4d3  // bfmmla z19.s, z6.h, z0.h\n"
      ".inst 0x6461e4b6  // bfmmla z22.s, z5.h, z1.h\n"
      ".inst 0x6460e4b9  // bfmmla z25.s, z5.h, z0.h\n"
      ".inst 0x6461e49c  // bfmmla z28.s, z4.h, z1.h\n"
      ".inst 0x6460e49f  // bfmmla z31.s, z4.h, z0.h\n"
      "6:"  // multiply loop done
      "decw x26, ALL, MUL #3\n"
      "uzp1 z0.d, z8.d, z11.d\n"
      "uzp2 z8.d, z8.d, z11.d\n"
      "uzp1 z1.d, z9.d, z12.d\n"
      "uzp2 z9.d, z9.d, z12.d\n"
      "st1w { z0.s }, p0, [%x[Cpanel]]\n"
      "uzp1 z0.d, z10.d, z13.d\n"
      "uzp2 z10.d, z10.d, z13.d\n"
      "st1w { z1.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z0.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "uzp1 z2.d, z14.d, z17.d\n"
      "uzp2 z14.d, z14.d, z17.d\n"
      "st1w { z8.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "uzp1 z1.d, z15.d, z18.d\n"
      "cmp x26, XZR\n"
      "st1w { z9.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "uzp2 z15.d, z15.d, z18.d\n"
      "uzp1 z17.d, z16.d, z19.d\n"
      "st1w { z10.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "uzp2 z16.d, z16.d, z19.d\n"
      "uzp1 z0.d, z20.d, z23.d\n"
      "st1w { z2.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "uzp2 z20.d, z20.d, z23.d\n"
      "uzp1 z23.d, z21.d, z24.d\n"
      "st1w { z1.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "uzp2 z21.d, z21.d, z24.d\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "uzp1 z19.d, z22.d, z25.d\n"
      "uzp2 z22.d, z22.d, z25.d\n"
      "st1w { z14.s }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "uzp1 z18.d, z26.d, z29.d\n"
      "uzp2 z26.d, z26.d, z29.d\n"
      "st1w { z15.s }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "uzp1 z17.d, z27.d, z30.d\n"
      "uzp2 z27.d, z27.d, z30.d\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "uzp1 z16.d, z28.d, z31.d\n"
      "uzp2 z28.d, z28.d, z31.d\n"
      "st1w { z0.s }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1w { z23.s }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1w { z19.s }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1w { z20.s }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1w { z21.s }, p0, [%x[Cpanel]]\n"
      "st1w { z22.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z18.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z26.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z27.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z28.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bne 1b\n"
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_B_stride] "I" (offsetof(KernelArgs, B_stride)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_cur_B_ptr] "I" (offsetof(KernelArgs, cur_B_ptr))
      : "cc", "memory", "p0", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // namespace arm_gemm
#endif // ARM_COMPUTE_ENABLE_SVE
