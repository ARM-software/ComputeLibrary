/*
 * Copyright (c) 2021 Arm Limited.
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
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

#include <cstddef>

namespace arm_gemm {

void a64_interleaved_fp16_mla_8x24_a55(
    const __fp16 *Apanel, const __fp16 *Bpanel,
    __fp16 *Cpanel, int ablocks, int bblocks, int K) {

    struct KernelArgs {
        size_t bblocks = {};
        size_t K = {};
        const __fp16 *Bpanel = {};
    } ka;

    ka.bblocks = bblocks;
    ka.K = (K/1) - 1;
    ka.Bpanel = Bpanel;

    __asm__ __volatile__(

      "1:"  // Height loop
      "ldr x10, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x9, %x[Apanel]\n"
      "ldr x28, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "ldr x27, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x9\n"
      "cmp x27, #0x2\n"
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "movi v10.16b, #0x0\n"
      "prfm pldl1keep, [x28, #0x0]\n"
      "movi v11.16b, #0x0\n"
      "prfm pldl1keep, [x28, #0x40]\n"
      "movi v12.16b, #0x0\n"
      "prfm pldl1keep, [x28, #0x80]\n"
      "movi v13.16b, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "movi v14.16b, #0x0\n"
      "ldr q2, [x28, #0x0]\n"
      "movi v15.16b, #0x0\n"
      "ldr q3, [x28, #0x10]\n"
      "movi v16.16b, #0x0\n"
      "ldr q4, [x28, #0x20]\n"
      "movi v17.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "movi v20.16b, #0x0\n"
      "movi v21.16b, #0x0\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      "ldr d1, [%x[Apanel], #0x10]\n"
      "fmla v8.8h, v2.8h, v0.h[0]\n"
      "ldr x26, [%x[Apanel], #0x18]\n"
      "fmla v11.8h, v2.8h, v0.h[1]\n"
      "ldr d5, [x28, #0x30]\n"
      "fmla v14.8h, v2.8h, v0.h[2]\n"
      "ldr x25, [x28, #0x38]\n"
      "fmla v17.8h, v2.8h, v0.h[3]\n"
      "ldr d6, [x28, #0x40]\n"
      "fmla v20.8h, v2.8h, v0.h[4]\n"
      "ldr x24, [x28, #0x48]\n"
      "fmla v23.8h, v2.8h, v0.h[5]\n"
      "ldr d7, [x28, #0x50]\n"
      "fmla v26.8h, v2.8h, v0.h[6]\n"
      "ldr x23, [x28, #0x58]\n"
      "fmla v29.8h, v2.8h, v0.h[7]\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "fmla v9.8h, v3.8h, v0.h[0]\n"
      "prfm pldl1keep, [x28, #0x100]\n"
      "fmla v12.8h, v3.8h, v0.h[1]\n"
      "prfm pldl1keep, [x28, #0x140]\n"
      "fmla v15.8h, v3.8h, v0.h[2]\n"
      "add x28, x28, #0x60\n"
      "fmla v18.8h, v3.8h, v0.h[3]\n"
      "ldr d2, [x28, #0x0]\n"
      "fmla v21.8h, v3.8h, v0.h[4]\n"
      "ldr x22, [x28, #0x8]\n"
      "fmla v24.8h, v3.8h, v0.h[5]\n"
      "ldr x21, [x28, #0x18]\n"
      "fmla v27.8h, v3.8h, v0.h[6]\n"
      "ldr x20, [%x[Apanel], #0x8]\n"
      "fmla v30.8h, v3.8h, v0.h[7]\n"
      "ldr d3, [x28, #0x10]\n"
      "fmla v10.8h, v4.8h, v0.h[0]\n"
      "ldr x19, [x28, #0x28]\n"
      "fmla v13.8h, v4.8h, v0.h[1]\n"
      "mov v1.d[1], x26\n"
      "fmla v16.8h, v4.8h, v0.h[2]\n"
      "mov v5.d[1], x25\n"
      "fmla v19.8h, v4.8h, v0.h[3]\n"
      "mov v6.d[1], x24\n"
      "fmla v22.8h, v4.8h, v0.h[4]\n"
      "mov v7.d[1], x23\n"
      "fmla v25.8h, v4.8h, v0.h[5]\n"
      "sub x27, x27, #0x2\n"
      "fmla v28.8h, v4.8h, v0.h[6]\n"
      "cmp x27, #0x2\n"
      "fmla v31.8h, v4.8h, v0.h[7]\n"
      "ldr d0, [%x[Apanel], #0x0]\n"
      "ldr d4, [x28, #0x20]\n"
      "mov v2.d[1], x22\n"
      "mov v3.d[1], x21\n"
      "fmla v8.8h, v5.8h, v1.h[0]\n"
      "mov v0.d[1], x20\n"
      "fmla v11.8h, v5.8h, v1.h[1]\n"
      "mov v4.d[1], x19\n"
      "fmla v14.8h, v5.8h, v1.h[2]\n"
      "fmla v17.8h, v5.8h, v1.h[3]\n"
      "fmla v20.8h, v5.8h, v1.h[4]\n"
      "fmla v23.8h, v5.8h, v1.h[5]\n"
      "fmla v26.8h, v5.8h, v1.h[6]\n"
      "fmla v29.8h, v5.8h, v1.h[7]\n"
      "fmla v9.8h, v6.8h, v1.h[0]\n"
      "fmla v12.8h, v6.8h, v1.h[1]\n"
      "fmla v15.8h, v6.8h, v1.h[2]\n"
      "fmla v18.8h, v6.8h, v1.h[3]\n"
      "fmla v21.8h, v6.8h, v1.h[4]\n"
      "fmla v24.8h, v6.8h, v1.h[5]\n"
      "fmla v27.8h, v6.8h, v1.h[6]\n"
      "fmla v30.8h, v6.8h, v1.h[7]\n"
      "fmla v10.8h, v7.8h, v1.h[0]\n"
      "fmla v13.8h, v7.8h, v1.h[1]\n"
      "fmla v16.8h, v7.8h, v1.h[2]\n"
      "fmla v19.8h, v7.8h, v1.h[3]\n"
      "fmla v22.8h, v7.8h, v1.h[4]\n"
      "fmla v25.8h, v7.8h, v1.h[5]\n"
      "fmla v28.8h, v7.8h, v1.h[6]\n"
      "fmla v31.8h, v7.8h, v1.h[7]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x10\n"
      "fmla v8.8h, v2.8h, v0.h[0]\n"
      "add x28, x28, #0x30\n"
      "fmla v11.8h, v2.8h, v0.h[1]\n"
      "fmla v14.8h, v2.8h, v0.h[2]\n"
      "fmla v17.8h, v2.8h, v0.h[3]\n"
      "fmla v20.8h, v2.8h, v0.h[4]\n"
      "fmla v23.8h, v2.8h, v0.h[5]\n"
      "fmla v26.8h, v2.8h, v0.h[6]\n"
      "fmla v29.8h, v2.8h, v0.h[7]\n"
      "fmla v9.8h, v3.8h, v0.h[0]\n"
      "fmla v12.8h, v3.8h, v0.h[1]\n"
      "fmla v15.8h, v3.8h, v0.h[2]\n"
      "fmla v18.8h, v3.8h, v0.h[3]\n"
      "fmla v21.8h, v3.8h, v0.h[4]\n"
      "fmla v24.8h, v3.8h, v0.h[5]\n"
      "fmla v27.8h, v3.8h, v0.h[6]\n"
      "fmla v30.8h, v3.8h, v0.h[7]\n"
      "fmla v10.8h, v4.8h, v0.h[0]\n"
      "fmla v13.8h, v4.8h, v0.h[1]\n"
      "fmla v16.8h, v4.8h, v0.h[2]\n"
      "fmla v19.8h, v4.8h, v0.h[3]\n"
      "fmla v22.8h, v4.8h, v0.h[4]\n"
      "fmla v25.8h, v4.8h, v0.h[5]\n"
      "fmla v28.8h, v4.8h, v0.h[6]\n"
      "fmla v31.8h, v4.8h, v0.h[7]\n"
      "cbz x27, 5f\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "add %x[Apanel], %x[Apanel], #0x10\n"
      "ldr q5, [x28, #0x0]\n"
      "fmla v8.8h, v5.8h, v0.h[0]\n"
      "ldr q6, [x28, #0x10]\n"
      "fmla v11.8h, v5.8h, v0.h[1]\n"
      "ldr q7, [x28, #0x20]\n"
      "fmla v14.8h, v5.8h, v0.h[2]\n"
      "fmla v17.8h, v5.8h, v0.h[3]\n"
      "add x28, x28, #0x30\n"
      "fmla v20.8h, v5.8h, v0.h[4]\n"
      "fmla v23.8h, v5.8h, v0.h[5]\n"
      "fmla v26.8h, v5.8h, v0.h[6]\n"
      "fmla v29.8h, v5.8h, v0.h[7]\n"
      "fmla v9.8h, v6.8h, v0.h[0]\n"
      "fmla v12.8h, v6.8h, v0.h[1]\n"
      "fmla v15.8h, v6.8h, v0.h[2]\n"
      "fmla v18.8h, v6.8h, v0.h[3]\n"
      "fmla v21.8h, v6.8h, v0.h[4]\n"
      "fmla v24.8h, v6.8h, v0.h[5]\n"
      "fmla v27.8h, v6.8h, v0.h[6]\n"
      "fmla v30.8h, v6.8h, v0.h[7]\n"
      "fmla v10.8h, v7.8h, v0.h[0]\n"
      "fmla v13.8h, v7.8h, v0.h[1]\n"
      "fmla v16.8h, v7.8h, v0.h[2]\n"
      "fmla v19.8h, v7.8h, v0.h[3]\n"
      "fmla v22.8h, v7.8h, v0.h[4]\n"
      "fmla v25.8h, v7.8h, v0.h[5]\n"
      "fmla v28.8h, v7.8h, v0.h[6]\n"
      "fmla v31.8h, v7.8h, v0.h[7]\n"
      "5:"  // multiply loop done
      "subs x10, x10, #0x1\n"
      "str q8, [%x[Cpanel], #0x0]\n"
      "str q9, [%x[Cpanel], #0x10]\n"
      "str q10, [%x[Cpanel], #0x20]\n"
      "str q11, [%x[Cpanel], #0x30]\n"
      "str q12, [%x[Cpanel], #0x40]\n"
      "str q13, [%x[Cpanel], #0x50]\n"
      "str q14, [%x[Cpanel], #0x60]\n"
      "str q15, [%x[Cpanel], #0x70]\n"
      "str q16, [%x[Cpanel], #0x80]\n"
      "str q17, [%x[Cpanel], #0x90]\n"
      "str q18, [%x[Cpanel], #0xa0]\n"
      "str q19, [%x[Cpanel], #0xb0]\n"
      "str q20, [%x[Cpanel], #0xc0]\n"
      "str q21, [%x[Cpanel], #0xd0]\n"
      "str q22, [%x[Cpanel], #0xe0]\n"
      "str q23, [%x[Cpanel], #0xf0]\n"
      "str q24, [%x[Cpanel], #0x100]\n"
      "str q25, [%x[Cpanel], #0x110]\n"
      "str q26, [%x[Cpanel], #0x120]\n"
      "str q27, [%x[Cpanel], #0x130]\n"
      "str q28, [%x[Cpanel], #0x140]\n"
      "str q29, [%x[Cpanel], #0x150]\n"
      "str q30, [%x[Cpanel], #0x160]\n"
      "str q31, [%x[Cpanel], #0x170]\n"
      "add %x[Cpanel], %x[Cpanel], #0x180\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bne 1b\n"
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // namespace arm_gemm
#endif // __aarch64__
