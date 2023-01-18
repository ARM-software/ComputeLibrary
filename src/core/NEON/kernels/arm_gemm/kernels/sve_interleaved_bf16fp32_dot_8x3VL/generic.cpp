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
#include "../../bfloat.hpp"

namespace arm_gemm {

void sve_interleaved_bf16fp32_dot_8x3VL(
    const bfloat16 *Apanel, const bfloat16 *Bpanel,
    float *Cpanel, int ablocks, int bblocks, int K) {

    struct KernelArgs {
        size_t bblocks = {};
        size_t K = {};
        const bfloat16 *Bpanel = {};
    } ka;

    ka.bblocks = bblocks;
    ka.K = (K/2) - 1;
    ka.Bpanel = Bpanel;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "1:"  // Height loop
      "ldr x22, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x21, %x[Apanel]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "mov z8.b, #0x0\n"
      "mov z9.b, #0x0\n"
      "ldr x19, [%x[args_ptr], %[offsetof_K]]\n"
      "mov z10.b, #0x0\n"
      "mov z11.b, #0x0\n"
      "ld1h { z4.h }, p0/Z, [x20]\n"
      "mov z12.b, #0x0\n"
      "mov z13.b, #0x0\n"
      "mov %x[Apanel], x21\n"
      "mov z14.b, #0x0\n"
      "mov z15.b, #0x0\n"
      "cmp x19, #0x2\n"
      "mov z16.b, #0x0\n"
      "mov z17.b, #0x0\n"
      "mov z18.b, #0x0\n"
      "mov z19.b, #0x0\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "mov z20.b, #0x0\n"
      "mov z21.b, #0x0\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #16]\n"
      "mov z22.b, #0x0\n"
      "mov z23.b, #0x0\n"
      "mov z24.b, #0x0\n"
      "mov z25.b, #0x0\n"
      "mov z26.b, #0x0\n"
      "mov z27.b, #0x0\n"
      "mov z28.b, #0x0\n"
      "mov z29.b, #0x0\n"
      "mov z30.b, #0x0\n"
      "mov z31.b, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      ".inst 0x64604088  // bfdot z8.s, z4.h, z0.h[0]\n"
      ".inst 0x6468408b  // bfdot z11.s, z4.h, z0.h[1]\n"
      "ld1h { z5.h }, p0/Z, [x20, #1, MUL VL]\n"
      ".inst 0x6470408e  // bfdot z14.s, z4.h, z0.h[2]\n"
      ".inst 0x64784091  // bfdot z17.s, z4.h, z0.h[3]\n"
      "ld1h { z6.h }, p0/Z, [x20, #2, MUL VL]\n"
      ".inst 0x64614094  // bfdot z20.s, z4.h, z1.h[0]\n"
      ".inst 0x64694097  // bfdot z23.s, z4.h, z1.h[1]\n"
      "ld1rqh { z2.h }, p0/Z, [%x[Apanel], #32]\n"
      ".inst 0x6471409a  // bfdot z26.s, z4.h, z1.h[2]\n"
      ".inst 0x6479409d  // bfdot z29.s, z4.h, z1.h[3]\n"
      "ld1rqh { z3.h }, p0/Z, [%x[Apanel], #48]\n"
      ".inst 0x646040a9  // bfdot z9.s, z5.h, z0.h[0]\n"
      ".inst 0x646840ac  // bfdot z12.s, z5.h, z0.h[1]\n"
      "ld1h { z4.h }, p0/Z, [x20, #3, MUL VL]\n"
      ".inst 0x647040af  // bfdot z15.s, z5.h, z0.h[2]\n"
      ".inst 0x647840b2  // bfdot z18.s, z5.h, z0.h[3]\n"
      "sub x19, x19, #0x2\n"
      ".inst 0x646140b5  // bfdot z21.s, z5.h, z1.h[0]\n"
      ".inst 0x646940b8  // bfdot z24.s, z5.h, z1.h[1]\n"
      "cmp x19, #0x2\n"
      ".inst 0x647140bb  // bfdot z27.s, z5.h, z1.h[2]\n"
      ".inst 0x647940be  // bfdot z30.s, z5.h, z1.h[3]\n"
      "ld1h { z5.h }, p0/Z, [x20, #4, MUL VL]\n"
      ".inst 0x646040ca  // bfdot z10.s, z6.h, z0.h[0]\n"
      ".inst 0x646840cd  // bfdot z13.s, z6.h, z0.h[1]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x647040d0  // bfdot z16.s, z6.h, z0.h[2]\n"
      ".inst 0x647840d3  // bfdot z19.s, z6.h, z0.h[3]\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x646140d6  // bfdot z22.s, z6.h, z1.h[0]\n"
      ".inst 0x646940d9  // bfdot z25.s, z6.h, z1.h[1]\n"
      ".inst 0x647140dc  // bfdot z28.s, z6.h, z1.h[2]\n"
      ".inst 0x647940df  // bfdot z31.s, z6.h, z1.h[3]\n"
      "ld1h { z6.h }, p0/Z, [x20, #5, MUL VL]\n"
      ".inst 0x64624088  // bfdot z8.s, z4.h, z2.h[0]\n"
      ".inst 0x646a408b  // bfdot z11.s, z4.h, z2.h[1]\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #16]\n"
      ".inst 0x6472408e  // bfdot z14.s, z4.h, z2.h[2]\n"
      ".inst 0x647a4091  // bfdot z17.s, z4.h, z2.h[3]\n"
      "addvl x20, x20, #6\n"
      ".inst 0x64634094  // bfdot z20.s, z4.h, z3.h[0]\n"
      ".inst 0x646b4097  // bfdot z23.s, z4.h, z3.h[1]\n"
      ".inst 0x6473409a  // bfdot z26.s, z4.h, z3.h[2]\n"
      ".inst 0x647b409d  // bfdot z29.s, z4.h, z3.h[3]\n"
      ".inst 0x646240a9  // bfdot z9.s, z5.h, z2.h[0]\n"
      ".inst 0x646a40ac  // bfdot z12.s, z5.h, z2.h[1]\n"
      "ld1h { z4.h }, p0/Z, [x20]\n"
      ".inst 0x647240af  // bfdot z15.s, z5.h, z2.h[2]\n"
      ".inst 0x647a40b2  // bfdot z18.s, z5.h, z2.h[3]\n"
      ".inst 0x646340b5  // bfdot z21.s, z5.h, z3.h[0]\n"
      ".inst 0x646b40b8  // bfdot z24.s, z5.h, z3.h[1]\n"
      ".inst 0x647340bb  // bfdot z27.s, z5.h, z3.h[2]\n"
      ".inst 0x647b40be  // bfdot z30.s, z5.h, z3.h[3]\n"
      ".inst 0x646240ca  // bfdot z10.s, z6.h, z2.h[0]\n"
      ".inst 0x646a40cd  // bfdot z13.s, z6.h, z2.h[1]\n"
      ".inst 0x647240d0  // bfdot z16.s, z6.h, z2.h[2]\n"
      ".inst 0x647a40d3  // bfdot z19.s, z6.h, z2.h[3]\n"
      ".inst 0x646340d6  // bfdot z22.s, z6.h, z3.h[0]\n"
      ".inst 0x646b40d9  // bfdot z25.s, z6.h, z3.h[1]\n"
      ".inst 0x647340dc  // bfdot z28.s, z6.h, z3.h[2]\n"
      ".inst 0x647b40df  // bfdot z31.s, z6.h, z3.h[3]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      ".inst 0x64604088  // bfdot z8.s, z4.h, z0.h[0]\n"
      ".inst 0x6468408b  // bfdot z11.s, z4.h, z0.h[1]\n"
      "ld1h { z5.h }, p0/Z, [x20, #1, MUL VL]\n"
      ".inst 0x6470408e  // bfdot z14.s, z4.h, z0.h[2]\n"
      ".inst 0x64784091  // bfdot z17.s, z4.h, z0.h[3]\n"
      "ld1h { z6.h }, p0/Z, [x20, #2, MUL VL]\n"
      ".inst 0x64614094  // bfdot z20.s, z4.h, z1.h[0]\n"
      ".inst 0x64694097  // bfdot z23.s, z4.h, z1.h[1]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      ".inst 0x6471409a  // bfdot z26.s, z4.h, z1.h[2]\n"
      ".inst 0x6479409d  // bfdot z29.s, z4.h, z1.h[3]\n"
      "addvl x20, x20, #3\n"
      ".inst 0x646040a9  // bfdot z9.s, z5.h, z0.h[0]\n"
      ".inst 0x646840ac  // bfdot z12.s, z5.h, z0.h[1]\n"
      ".inst 0x647040af  // bfdot z15.s, z5.h, z0.h[2]\n"
      ".inst 0x647840b2  // bfdot z18.s, z5.h, z0.h[3]\n"
      ".inst 0x646140b5  // bfdot z21.s, z5.h, z1.h[0]\n"
      ".inst 0x646940b8  // bfdot z24.s, z5.h, z1.h[1]\n"
      ".inst 0x647140bb  // bfdot z27.s, z5.h, z1.h[2]\n"
      ".inst 0x647940be  // bfdot z30.s, z5.h, z1.h[3]\n"
      ".inst 0x646040ca  // bfdot z10.s, z6.h, z0.h[0]\n"
      ".inst 0x646840cd  // bfdot z13.s, z6.h, z0.h[1]\n"
      ".inst 0x647040d0  // bfdot z16.s, z6.h, z0.h[2]\n"
      ".inst 0x647840d3  // bfdot z19.s, z6.h, z0.h[3]\n"
      ".inst 0x646140d6  // bfdot z22.s, z6.h, z1.h[0]\n"
      ".inst 0x646940d9  // bfdot z25.s, z6.h, z1.h[1]\n"
      ".inst 0x647140dc  // bfdot z28.s, z6.h, z1.h[2]\n"
      ".inst 0x647940df  // bfdot z31.s, z6.h, z1.h[3]\n"
      "cbz x19, 5f\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #16]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ld1h { z7.h }, p0/Z, [x20]\n"
      "ld1h { z4.h }, p0/Z, [x20, #1, MUL VL]\n"
      "ld1h { z5.h }, p0/Z, [x20, #2, MUL VL]\n"
      "addvl x20, x20, #3\n"
      ".inst 0x646040e8  // bfdot z8.s, z7.h, z0.h[0]\n"
      ".inst 0x646840eb  // bfdot z11.s, z7.h, z0.h[1]\n"
      ".inst 0x647040ee  // bfdot z14.s, z7.h, z0.h[2]\n"
      ".inst 0x647840f1  // bfdot z17.s, z7.h, z0.h[3]\n"
      ".inst 0x646140f4  // bfdot z20.s, z7.h, z1.h[0]\n"
      ".inst 0x646940f7  // bfdot z23.s, z7.h, z1.h[1]\n"
      ".inst 0x647140fa  // bfdot z26.s, z7.h, z1.h[2]\n"
      ".inst 0x647940fd  // bfdot z29.s, z7.h, z1.h[3]\n"
      ".inst 0x64604089  // bfdot z9.s, z4.h, z0.h[0]\n"
      ".inst 0x6468408c  // bfdot z12.s, z4.h, z0.h[1]\n"
      ".inst 0x6470408f  // bfdot z15.s, z4.h, z0.h[2]\n"
      ".inst 0x64784092  // bfdot z18.s, z4.h, z0.h[3]\n"
      ".inst 0x64614095  // bfdot z21.s, z4.h, z1.h[0]\n"
      ".inst 0x64694098  // bfdot z24.s, z4.h, z1.h[1]\n"
      ".inst 0x6471409b  // bfdot z27.s, z4.h, z1.h[2]\n"
      ".inst 0x6479409e  // bfdot z30.s, z4.h, z1.h[3]\n"
      ".inst 0x646040aa  // bfdot z10.s, z5.h, z0.h[0]\n"
      ".inst 0x646840ad  // bfdot z13.s, z5.h, z0.h[1]\n"
      ".inst 0x647040b0  // bfdot z16.s, z5.h, z0.h[2]\n"
      ".inst 0x647840b3  // bfdot z19.s, z5.h, z0.h[3]\n"
      ".inst 0x646140b6  // bfdot z22.s, z5.h, z1.h[0]\n"
      ".inst 0x646940b9  // bfdot z25.s, z5.h, z1.h[1]\n"
      ".inst 0x647140bc  // bfdot z28.s, z5.h, z1.h[2]\n"
      ".inst 0x647940bf  // bfdot z31.s, z5.h, z1.h[3]\n"
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
#endif // __ARM_FEATURE_SVE
