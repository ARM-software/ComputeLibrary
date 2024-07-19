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
#ifdef ARM_COMPUTE_ENABLE_SVE

#include <cstddef>
#include <cstdint>

namespace arm_gemm {

void sve_interleaved_u8s8s32_mmla_8x3VL(
    const uint8_t *Apanel,
    const int8_t *Bpanel,
    int32_t *Cpanel,
    int ablocks,
    int bblocks,
    int K) {

    struct KernelArgs {
        size_t K = {};
        const int8_t *Bpanel = {};
        size_t bblocks = {};
    } ka;

    ka.K = (K/8) - 1;
    ka.Bpanel = Bpanel;
    ka.bblocks = bblocks;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "1:"  // Height loop
      "ldr x23, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x21, %x[Apanel]\n"
      "2:"  // Width loop
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x21\n"
      "mov z8.s, #0x0\n"
      "mov z9.s, #0x0\n"
      "mov z10.s, #0x0\n"
      "ld1b { z4.b }, p0/Z, [x22]\n"
      "mov z11.s, #0x0\n"
      "mov z12.s, #0x0\n"
      "ld1b { z5.b }, p0/Z, [x22, #1, MUL VL]\n"
      "cmp x20, #0x2\n"
      "mov z13.s, #0x0\n"
      "mov z14.s, #0x0\n"
      "mov z15.s, #0x0\n"
      "mov z16.s, #0x0\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel]]\n"
      "mov z17.s, #0x0\n"
      "mov z18.s, #0x0\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #16]\n"
      "mov z19.s, #0x0\n"
      "mov z20.s, #0x0\n"
      "ld1rqb { z2.b }, p0/Z, [%x[Apanel], #32]\n"
      "mov z21.s, #0x0\n"
      "mov z22.s, #0x0\n"
      "addvl x22, x22, #2\n"
      "mov z23.s, #0x0\n"
      "mov z24.s, #0x0\n"
      "add %x[Apanel], %x[Apanel], #0x30\n"
      "mov z25.s, #0x0\n"
      "mov z26.s, #0x0\n"
      "mov z27.s, #0x0\n"
      "mov z28.s, #0x0\n"
      "mov z29.s, #0x0\n"
      "mov z30.s, #0x0\n"
      "mov z31.s, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      "ld1rqb { z6.b }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x45849808  // usmmla z8.s, z0.b, z4.b\n"
      ".inst 0x4585980b  // usmmla z11.s, z0.b, z5.b\n"
      ".inst 0x4584982e  // usmmla z14.s, z1.b, z4.b\n"
      ".inst 0x45859831  // usmmla z17.s, z1.b, z5.b\n"
      "ld1b { z3.b }, p0/Z, [x22]\n"
      ".inst 0x45849854  // usmmla z20.s, z2.b, z4.b\n"
      ".inst 0x45859857  // usmmla z23.s, z2.b, z5.b\n"
      "ld1b { z7.b }, p0/Z, [x22, #1, MUL VL]\n"
      ".inst 0x458498da  // usmmla z26.s, z6.b, z4.b\n"
      ".inst 0x458598dd  // usmmla z29.s, z6.b, z5.b\n"
      "ld1b { z4.b }, p0/Z, [x22, #2, MUL VL]\n"
      "ld1b { z5.b }, p0/Z, [x22, #3, MUL VL]\n"
      ".inst 0x45839809  // usmmla z9.s, z0.b, z3.b\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x4587980c  // usmmla z12.s, z0.b, z7.b\n"
      ".inst 0x4583982f  // usmmla z15.s, z1.b, z3.b\n"
      "cmp x20, #0x2\n"
      ".inst 0x45879832  // usmmla z18.s, z1.b, z7.b\n"
      ".inst 0x45839855  // usmmla z21.s, z2.b, z3.b\n"
      ".inst 0x45879858  // usmmla z24.s, z2.b, z7.b\n"
      ".inst 0x458398db  // usmmla z27.s, z6.b, z3.b\n"
      "ld1b { z3.b }, p0/Z, [x22, #4, MUL VL]\n"
      ".inst 0x458798de  // usmmla z30.s, z6.b, z7.b\n"
      ".inst 0x4584980a  // usmmla z10.s, z0.b, z4.b\n"
      "ld1b { z7.b }, p0/Z, [x22, #5, MUL VL]\n"
      ".inst 0x4585980d  // usmmla z13.s, z0.b, z5.b\n"
      ".inst 0x45849830  // usmmla z16.s, z1.b, z4.b\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel], #16]\n"
      ".inst 0x45859833  // usmmla z19.s, z1.b, z5.b\n"
      ".inst 0x45849856  // usmmla z22.s, z2.b, z4.b\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #32]\n"
      ".inst 0x45859859  // usmmla z25.s, z2.b, z5.b\n"
      ".inst 0x458498dc  // usmmla z28.s, z6.b, z4.b\n"
      "ld1rqb { z2.b }, p0/Z, [%x[Apanel], #48]\n"
      ".inst 0x458598df  // usmmla z31.s, z6.b, z5.b\n"
      "ld1rqb { z6.b }, p0/Z, [%x[Apanel], #64]\n"
      "ld1b { z4.b }, p0/Z, [x22, #6, MUL VL]\n"
      "ld1b { z5.b }, p0/Z, [x22, #7, MUL VL]\n"
      "addvl x22, x22, #16\n"
      ".inst 0x45839808  // usmmla z8.s, z0.b, z3.b\n"
      ".inst 0x4587980b  // usmmla z11.s, z0.b, z7.b\n"
      ".inst 0x4583982e  // usmmla z14.s, z1.b, z3.b\n"
      ".inst 0x45879831  // usmmla z17.s, z1.b, z7.b\n"
      ".inst 0x45839854  // usmmla z20.s, z2.b, z3.b\n"
      ".inst 0x45879857  // usmmla z23.s, z2.b, z7.b\n"
      ".inst 0x458398da  // usmmla z26.s, z6.b, z3.b\n"
      "ld1b { z3.b }, p0/Z, [x22, #-8, MUL VL]\n"
      ".inst 0x458798dd  // usmmla z29.s, z6.b, z7.b\n"
      "ld1b { z7.b }, p0/Z, [x22, #-7, MUL VL]\n"
      ".inst 0x45849809  // usmmla z9.s, z0.b, z4.b\n"
      ".inst 0x4585980c  // usmmla z12.s, z0.b, z5.b\n"
      ".inst 0x4584982f  // usmmla z15.s, z1.b, z4.b\n"
      ".inst 0x45859832  // usmmla z18.s, z1.b, z5.b\n"
      ".inst 0x45849855  // usmmla z21.s, z2.b, z4.b\n"
      ".inst 0x45859858  // usmmla z24.s, z2.b, z5.b\n"
      ".inst 0x458498db  // usmmla z27.s, z6.b, z4.b\n"
      "ld1b { z4.b }, p0/Z, [x22, #-6, MUL VL]\n"
      ".inst 0x458598de  // usmmla z30.s, z6.b, z5.b\n"
      ".inst 0x4583980a  // usmmla z10.s, z0.b, z3.b\n"
      "ld1b { z5.b }, p0/Z, [x22, #-5, MUL VL]\n"
      ".inst 0x4587980d  // usmmla z13.s, z0.b, z7.b\n"
      ".inst 0x45839830  // usmmla z16.s, z1.b, z3.b\n"
      "ld1rqb { z0.b }, p0/Z, [%x[Apanel], #80]\n"
      ".inst 0x45879833  // usmmla z19.s, z1.b, z7.b\n"
      ".inst 0x45839856  // usmmla z22.s, z2.b, z3.b\n"
      "ld1rqb { z1.b }, p0/Z, [%x[Apanel], #96]\n"
      ".inst 0x45879859  // usmmla z25.s, z2.b, z7.b\n"
      ".inst 0x458398dc  // usmmla z28.s, z6.b, z3.b\n"
      "ld1rqb { z2.b }, p0/Z, [%x[Apanel], #112]\n"
      ".inst 0x458798df  // usmmla z31.s, z6.b, z7.b\n"
      "add %x[Apanel], %x[Apanel], #0x80\n"
      "addvl x22, x22, #-4\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "ld1rqb { z3.b }, p0/Z, [%x[Apanel]]\n"
      ".inst 0x45849808  // usmmla z8.s, z0.b, z4.b\n"
      ".inst 0x4585980b  // usmmla z11.s, z0.b, z5.b\n"
      ".inst 0x4584982e  // usmmla z14.s, z1.b, z4.b\n"
      ".inst 0x45859831  // usmmla z17.s, z1.b, z5.b\n"
      "ld1b { z6.b }, p0/Z, [x22]\n"
      ".inst 0x45849854  // usmmla z20.s, z2.b, z4.b\n"
      ".inst 0x45859857  // usmmla z23.s, z2.b, z5.b\n"
      "ld1b { z7.b }, p0/Z, [x22, #1, MUL VL]\n"
      ".inst 0x4584987a  // usmmla z26.s, z3.b, z4.b\n"
      ".inst 0x4585987d  // usmmla z29.s, z3.b, z5.b\n"
      "ld1b { z5.b }, p0/Z, [x22, #2, MUL VL]\n"
      "ld1b { z4.b }, p0/Z, [x22, #3, MUL VL]\n"
      ".inst 0x45869809  // usmmla z9.s, z0.b, z6.b\n"
      "add %x[Apanel], %x[Apanel], #0x10\n"
      ".inst 0x4587980c  // usmmla z12.s, z0.b, z7.b\n"
      ".inst 0x4586982f  // usmmla z15.s, z1.b, z6.b\n"
      "addvl x22, x22, #4\n"
      ".inst 0x45879832  // usmmla z18.s, z1.b, z7.b\n"
      ".inst 0x45869855  // usmmla z21.s, z2.b, z6.b\n"
      ".inst 0x45879858  // usmmla z24.s, z2.b, z7.b\n"
      ".inst 0x4586987b  // usmmla z27.s, z3.b, z6.b\n"
      ".inst 0x4587987e  // usmmla z30.s, z3.b, z7.b\n"
      ".inst 0x4585980a  // usmmla z10.s, z0.b, z5.b\n"
      ".inst 0x4584980d  // usmmla z13.s, z0.b, z4.b\n"
      ".inst 0x45859830  // usmmla z16.s, z1.b, z5.b\n"
      ".inst 0x45849833  // usmmla z19.s, z1.b, z4.b\n"
      ".inst 0x45859856  // usmmla z22.s, z2.b, z5.b\n"
      ".inst 0x45849859  // usmmla z25.s, z2.b, z4.b\n"
      ".inst 0x4585987c  // usmmla z28.s, z3.b, z5.b\n"
      ".inst 0x4584987f  // usmmla z31.s, z3.b, z4.b\n"
      "cbz x20, 5f\n"
      "ld1b { z1.b }, p0/Z, [x22]\n"
      "ld1rqb { z7.b }, p0/Z, [%x[Apanel]]\n"
      "ld1rqb { z6.b }, p0/Z, [%x[Apanel], #16]\n"
      "ld1b { z0.b }, p0/Z, [x22, #1, MUL VL]\n"
      "ld1rqb { z5.b }, p0/Z, [%x[Apanel], #32]\n"
      "ld1rqb { z4.b }, p0/Z, [%x[Apanel], #48]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x458198e8  // usmmla z8.s, z7.b, z1.b\n"
      "ld1b { z3.b }, p0/Z, [x22, #2, MUL VL]\n"
      "ld1b { z2.b }, p0/Z, [x22, #3, MUL VL]\n"
      ".inst 0x458098eb  // usmmla z11.s, z7.b, z0.b\n"
      ".inst 0x458198ce  // usmmla z14.s, z6.b, z1.b\n"
      ".inst 0x458098d1  // usmmla z17.s, z6.b, z0.b\n"
      ".inst 0x458198b4  // usmmla z20.s, z5.b, z1.b\n"
      ".inst 0x458098b7  // usmmla z23.s, z5.b, z0.b\n"
      ".inst 0x4581989a  // usmmla z26.s, z4.b, z1.b\n"
      "ld1b { z1.b }, p0/Z, [x22, #4, MUL VL]\n"
      ".inst 0x4580989d  // usmmla z29.s, z4.b, z0.b\n"
      "ld1b { z0.b }, p0/Z, [x22, #5, MUL VL]\n"
      ".inst 0x458398e9  // usmmla z9.s, z7.b, z3.b\n"
      ".inst 0x458298ec  // usmmla z12.s, z7.b, z2.b\n"
      ".inst 0x458398cf  // usmmla z15.s, z6.b, z3.b\n"
      "addvl x22, x22, #6\n"
      ".inst 0x458298d2  // usmmla z18.s, z6.b, z2.b\n"
      ".inst 0x458398b5  // usmmla z21.s, z5.b, z3.b\n"
      ".inst 0x458298b8  // usmmla z24.s, z5.b, z2.b\n"
      ".inst 0x4583989b  // usmmla z27.s, z4.b, z3.b\n"
      ".inst 0x4582989e  // usmmla z30.s, z4.b, z2.b\n"
      ".inst 0x458198ea  // usmmla z10.s, z7.b, z1.b\n"
      ".inst 0x458098ed  // usmmla z13.s, z7.b, z0.b\n"
      ".inst 0x458198d0  // usmmla z16.s, z6.b, z1.b\n"
      ".inst 0x458098d3  // usmmla z19.s, z6.b, z0.b\n"
      ".inst 0x458198b6  // usmmla z22.s, z5.b, z1.b\n"
      ".inst 0x458098b9  // usmmla z25.s, z5.b, z0.b\n"
      ".inst 0x4581989c  // usmmla z28.s, z4.b, z1.b\n"
      ".inst 0x4580989f  // usmmla z31.s, z4.b, z0.b\n"
      "5:"  // multiply loop done
      "uzp1 z2.d, z8.d, z11.d\n"
      "uzp2 z8.d, z8.d, z11.d\n"
      "subs x23, x23, #0x1\n"
      "uzp1 z1.d, z9.d, z12.d\n"
      "uzp2 z9.d, z9.d, z12.d\n"
      "uzp1 z0.d, z10.d, z13.d\n"
      "uzp2 z10.d, z10.d, z13.d\n"
      "st1w { z2.s }, p0, [%x[Cpanel]]\n"
      "uzp1 z3.d, z14.d, z17.d\n"
      "uzp2 z14.d, z14.d, z17.d\n"
      "st1w { z1.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "uzp1 z17.d, z15.d, z18.d\n"
      "uzp2 z15.d, z15.d, z18.d\n"
      "st1w { z0.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "uzp1 z2.d, z16.d, z19.d\n"
      "uzp2 z16.d, z16.d, z19.d\n"
      "st1w { z8.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "uzp1 z1.d, z20.d, z23.d\n"
      "uzp2 z20.d, z20.d, z23.d\n"
      "st1w { z9.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "uzp1 z0.d, z21.d, z24.d\n"
      "uzp2 z21.d, z21.d, z24.d\n"
      "st1w { z10.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "uzp1 z23.d, z22.d, z25.d\n"
      "uzp2 z22.d, z22.d, z25.d\n"
      "st1w { z3.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "uzp1 z19.d, z26.d, z29.d\n"
      "uzp2 z26.d, z26.d, z29.d\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "uzp1 z18.d, z27.d, z30.d\n"
      "uzp2 z27.d, z27.d, z30.d\n"
      "uzp1 z17.d, z28.d, z31.d\n"
      "uzp2 z28.d, z28.d, z31.d\n"
      "st1w { z2.s }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "st1w { z14.s }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "st1w { z15.s }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "st1w { z1.s }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1w { z0.s }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1w { z23.s }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1w { z20.s }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1w { z21.s }, p0, [%x[Cpanel]]\n"
      "st1w { z22.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z19.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z18.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z26.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z27.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z28.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bne 1b\n"
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "p0", "x20", "x21", "x22", "x23", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // namespace arm_gemm
#endif // ARM_COMPUTE_ENABLE_SVE
