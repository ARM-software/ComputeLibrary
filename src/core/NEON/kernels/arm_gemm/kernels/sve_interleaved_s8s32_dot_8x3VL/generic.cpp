/*
 * Copyright (c) 2019-2021 Arm Limited.
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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifdef ARM_COMPUTE_ENABLE_SVE

#include <cstddef>
#include <cstdint>

namespace arm_gemm {

void sve_interleaved_s8s32_dot_8x3VL(
    const int8_t *Apanel, const int8_t *Bpanel,
    int32_t *Cpanel, int ablocks, int bblocks, int K) {

    struct KernelArgs {
        size_t bblocks = {};
        size_t K = {};
        const int8_t *Bpanel = {};
    } ka;

    ka.bblocks = bblocks;
    ka.K = (K/4) - 1;
    ka.Bpanel = Bpanel;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "1:"  // Height loop
      "ldr x22, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x21, %x[Apanel]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "mov z8.s, #0x0\n"
      "mov z9.s, #0x0\n"
      "ldr x19, [%x[args_ptr], %[offsetof_K]]\n"
      "mov z10.s, #0x0\n"
      "mov z11.s, #0x0\n"
      "ld1b { z4.b }, p0/Z, [x20]\n"
      "mov z12.s, #0x0\n"
      "mov z13.s, #0x0\n"
      "mov %x[Apanel], x21\n"
      "mov z14.s, #0x0\n"
      "mov z15.s, #0x0\n"
      "cmp x19, #0x2\n"
      "mov z16.s, #0x0\n"
      "mov z17.s, #0x0\n"
      "mov z18.s, #0x0\n"
      "mov z19.s, #0x0\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel]]\n"
      "mov z20.s, #0x0\n"
      "mov z21.s, #0x0\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #16]\n"
      "mov z22.s, #0x0\n"
      "mov z23.s, #0x0\n"
      "mov z24.s, #0x0\n"
      "mov z25.s, #0x0\n"
      "mov z26.s, #0x0\n"
      "mov z27.s, #0x0\n"
      "mov z28.s, #0x0\n"
      "mov z29.s, #0x0\n"
      "mov z30.s, #0x0\n"
      "mov z31.s, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      "sdot z8.s, z4.b, z0.b[0]\n"
      "sdot z11.s, z4.b, z0.b[1]\n"
      "ld1b { z5.b }, p0/Z, [x20, #1, MUL VL]\n"
      "sdot z14.s, z4.b, z0.b[2]\n"
      "sdot z17.s, z4.b, z0.b[3]\n"
      "ld1b { z6.b }, p0/Z, [x20, #2, MUL VL]\n"
      "sdot z20.s, z4.b, z1.b[0]\n"
      "sdot z23.s, z4.b, z1.b[1]\n"
      "ld1rqb { z2.b }, p0/Z, [%x[Apanel], #32]\n"
      "sdot z26.s, z4.b, z1.b[2]\n"
      "sdot z29.s, z4.b, z1.b[3]\n"
      "ld1rqb { z3.b }, p0/Z, [%x[Apanel], #48]\n"
      "sdot z9.s, z5.b, z0.b[0]\n"
      "sdot z12.s, z5.b, z0.b[1]\n"
      "ld1b { z4.b }, p0/Z, [x20, #3, MUL VL]\n"
      "sdot z15.s, z5.b, z0.b[2]\n"
      "sdot z18.s, z5.b, z0.b[3]\n"
      "sub x19, x19, #0x2\n"
      "sdot z21.s, z5.b, z1.b[0]\n"
      "sdot z24.s, z5.b, z1.b[1]\n"
      "cmp x19, #0x2\n"
      "sdot z27.s, z5.b, z1.b[2]\n"
      "sdot z30.s, z5.b, z1.b[3]\n"
      "ld1b { z5.b }, p0/Z, [x20, #4, MUL VL]\n"
      "sdot z10.s, z6.b, z0.b[0]\n"
      "sdot z13.s, z6.b, z0.b[1]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "sdot z16.s, z6.b, z0.b[2]\n"
      "sdot z19.s, z6.b, z0.b[3]\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel]]\n"
      "sdot z22.s, z6.b, z1.b[0]\n"
      "sdot z25.s, z6.b, z1.b[1]\n"
      "sdot z28.s, z6.b, z1.b[2]\n"
      "sdot z31.s, z6.b, z1.b[3]\n"
      "ld1b { z6.b }, p0/Z, [x20, #5, MUL VL]\n"
      "sdot z8.s, z4.b, z2.b[0]\n"
      "sdot z11.s, z4.b, z2.b[1]\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #16]\n"
      "sdot z14.s, z4.b, z2.b[2]\n"
      "sdot z17.s, z4.b, z2.b[3]\n"
      "addvl x20, x20, #6\n"
      "sdot z20.s, z4.b, z3.b[0]\n"
      "sdot z23.s, z4.b, z3.b[1]\n"
      "sdot z26.s, z4.b, z3.b[2]\n"
      "sdot z29.s, z4.b, z3.b[3]\n"
      "sdot z9.s, z5.b, z2.b[0]\n"
      "sdot z12.s, z5.b, z2.b[1]\n"
      "ld1b { z4.b }, p0/Z, [x20]\n"
      "sdot z15.s, z5.b, z2.b[2]\n"
      "sdot z18.s, z5.b, z2.b[3]\n"
      "sdot z21.s, z5.b, z3.b[0]\n"
      "sdot z24.s, z5.b, z3.b[1]\n"
      "sdot z27.s, z5.b, z3.b[2]\n"
      "sdot z30.s, z5.b, z3.b[3]\n"
      "sdot z10.s, z6.b, z2.b[0]\n"
      "sdot z13.s, z6.b, z2.b[1]\n"
      "sdot z16.s, z6.b, z2.b[2]\n"
      "sdot z19.s, z6.b, z2.b[3]\n"
      "sdot z22.s, z6.b, z3.b[0]\n"
      "sdot z25.s, z6.b, z3.b[1]\n"
      "sdot z28.s, z6.b, z3.b[2]\n"
      "sdot z31.s, z6.b, z3.b[3]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "sdot z8.s, z4.b, z0.b[0]\n"
      "sdot z11.s, z4.b, z0.b[1]\n"
      "ld1b { z5.b }, p0/Z, [x20, #1, MUL VL]\n"
      "sdot z14.s, z4.b, z0.b[2]\n"
      "sdot z17.s, z4.b, z0.b[3]\n"
      "ld1b { z6.b }, p0/Z, [x20, #2, MUL VL]\n"
      "sdot z20.s, z4.b, z1.b[0]\n"
      "sdot z23.s, z4.b, z1.b[1]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "sdot z26.s, z4.b, z1.b[2]\n"
      "sdot z29.s, z4.b, z1.b[3]\n"
      "addvl x20, x20, #3\n"
      "sdot z9.s, z5.b, z0.b[0]\n"
      "sdot z12.s, z5.b, z0.b[1]\n"
      "sdot z15.s, z5.b, z0.b[2]\n"
      "sdot z18.s, z5.b, z0.b[3]\n"
      "sdot z21.s, z5.b, z1.b[0]\n"
      "sdot z24.s, z5.b, z1.b[1]\n"
      "sdot z27.s, z5.b, z1.b[2]\n"
      "sdot z30.s, z5.b, z1.b[3]\n"
      "sdot z10.s, z6.b, z0.b[0]\n"
      "sdot z13.s, z6.b, z0.b[1]\n"
      "sdot z16.s, z6.b, z0.b[2]\n"
      "sdot z19.s, z6.b, z0.b[3]\n"
      "sdot z22.s, z6.b, z1.b[0]\n"
      "sdot z25.s, z6.b, z1.b[1]\n"
      "sdot z28.s, z6.b, z1.b[2]\n"
      "sdot z31.s, z6.b, z1.b[3]\n"
      "cbz x19, 5f\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel]]\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #16]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ld1b { z7.b }, p0/Z, [x20]\n"
      "ld1b { z4.b }, p0/Z, [x20, #1, MUL VL]\n"
      "ld1b { z5.b }, p0/Z, [x20, #2, MUL VL]\n"
      "addvl x20, x20, #3\n"
      "sdot z8.s, z7.b, z0.b[0]\n"
      "sdot z11.s, z7.b, z0.b[1]\n"
      "sdot z14.s, z7.b, z0.b[2]\n"
      "sdot z17.s, z7.b, z0.b[3]\n"
      "sdot z20.s, z7.b, z1.b[0]\n"
      "sdot z23.s, z7.b, z1.b[1]\n"
      "sdot z26.s, z7.b, z1.b[2]\n"
      "sdot z29.s, z7.b, z1.b[3]\n"
      "sdot z9.s, z4.b, z0.b[0]\n"
      "sdot z12.s, z4.b, z0.b[1]\n"
      "sdot z15.s, z4.b, z0.b[2]\n"
      "sdot z18.s, z4.b, z0.b[3]\n"
      "sdot z21.s, z4.b, z1.b[0]\n"
      "sdot z24.s, z4.b, z1.b[1]\n"
      "sdot z27.s, z4.b, z1.b[2]\n"
      "sdot z30.s, z4.b, z1.b[3]\n"
      "sdot z10.s, z5.b, z0.b[0]\n"
      "sdot z13.s, z5.b, z0.b[1]\n"
      "sdot z16.s, z5.b, z0.b[2]\n"
      "sdot z19.s, z5.b, z0.b[3]\n"
      "sdot z22.s, z5.b, z1.b[0]\n"
      "sdot z25.s, z5.b, z1.b[1]\n"
      "sdot z28.s, z5.b, z1.b[2]\n"
      "sdot z31.s, z5.b, z1.b[3]\n"
      "5:"  // multiply loop done
      "st1w { z8.s }, p0, [%x[Cpanel]]\n"
      "subs x22, x22, #0x1\n"
      "st1w { z9.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z10.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z11.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z12.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z13.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z14.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z15.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "st1w { z18.s }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "st1w { z19.s }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "st1w { z20.s }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1w { z21.s }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1w { z22.s }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1w { z23.s }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1w { z24.s }, p0, [%x[Cpanel]]\n"
      "st1w { z25.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z26.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z27.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z28.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z29.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z30.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z31.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bne 1b\n"
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "p0", "x19", "x20", "x21", "x22", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // namespace arm_gemm
#endif // ARM_COMPUTE_ENABLE_SVE
