/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#ifdef __aarch64__

#include <cstddef>
#include "../../bfloat.hpp"

namespace arm_gemm {

void a64_interleaved_bf16fp32_dot_8x12(
    const bfloat16 *Apanel,
    const bfloat16 *Bpanel,
    float *Cpanel,
    int ablocks,
    int bblocks,
    int K) {

    struct KernelArgs {
        size_t K = {};
        const bfloat16 *Bpanel = {};
        size_t bblocks = {};
    } ka;

    ka.K = (K/2) - 1;
    ka.Bpanel = Bpanel;
    ka.bblocks = bblocks;

    __asm__ __volatile__(
      "1:"  // Height loop
      "ldr x23, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x21, %x[Apanel]\n"
      "2:"  // Width loop
      "ldr q4, [x22, #0x0]\n"
      "ldr q5, [x22, #0x10]\n"
      "mov %x[Apanel], x21\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v8.16b, #0x0\n"
      "ldr q6, [x22, #0x20]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "cmp x20, #0x2\n"
      "movi v9.16b, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "movi v14.16b, #0x0\n"
      "movi v15.16b, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x40]\n"
      "movi v16.16b, #0x0\n"
      "movi v17.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x80]\n"
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
      "ldr q3, [%x[Apanel], #0x20]\n"
      "ldr q7, [%x[Apanel], #0x30]\n"
      ".inst 0x4f40f088  // bfdot v8.4s, v4.8h, v0.h[0]\n"
      ".inst 0x4f60f08b  // bfdot v11.4s, v4.8h, v0.h[1]\n"
      ".inst 0x4f40f88e  // bfdot v14.4s, v4.8h, v0.h[2]\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x4f60f891  // bfdot v17.4s, v4.8h, v0.h[3]\n"
      ".inst 0x4f41f094  // bfdot v20.4s, v4.8h, v1.h[0]\n"
      "cmp x20, #0x2\n"
      ".inst 0x4f61f097  // bfdot v23.4s, v4.8h, v1.h[1]\n"
      ".inst 0x4f41f89a  // bfdot v26.4s, v4.8h, v1.h[2]\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      ".inst 0x4f61f89d  // bfdot v29.4s, v4.8h, v1.h[3]\n"
      "ldr q4, [x22, #0x30]\n"
      ".inst 0x4f40f0a9  // bfdot v9.4s, v5.8h, v0.h[0]\n"
      ".inst 0x4f60f0ac  // bfdot v12.4s, v5.8h, v0.h[1]\n"
      ".inst 0x4f40f8af  // bfdot v15.4s, v5.8h, v0.h[2]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x4f60f8b2  // bfdot v18.4s, v5.8h, v0.h[3]\n"
      ".inst 0x4f41f0b5  // bfdot v21.4s, v5.8h, v1.h[0]\n"
      "prfm pldl1keep, [x22, #0x100]\n"
      ".inst 0x4f61f0b8  // bfdot v24.4s, v5.8h, v1.h[1]\n"
      ".inst 0x4f41f8bb  // bfdot v27.4s, v5.8h, v1.h[2]\n"
      "prfm pldl1keep, [x22, #0x140]\n"
      ".inst 0x4f61f8be  // bfdot v30.4s, v5.8h, v1.h[3]\n"
      "ldr q5, [x22, #0x40]\n"
      ".inst 0x4f40f0ca  // bfdot v10.4s, v6.8h, v0.h[0]\n"
      ".inst 0x4f60f0cd  // bfdot v13.4s, v6.8h, v0.h[1]\n"
      ".inst 0x4f40f8d0  // bfdot v16.4s, v6.8h, v0.h[2]\n"
      ".inst 0x4f60f8d3  // bfdot v19.4s, v6.8h, v0.h[3]\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      ".inst 0x4f41f0d6  // bfdot v22.4s, v6.8h, v1.h[0]\n"
      ".inst 0x4f61f0d9  // bfdot v25.4s, v6.8h, v1.h[1]\n"
      ".inst 0x4f41f8dc  // bfdot v28.4s, v6.8h, v1.h[2]\n"
      ".inst 0x4f61f8df  // bfdot v31.4s, v6.8h, v1.h[3]\n"
      "ldr q2, [x22, #0x50]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add x22, x22, #0x60\n"
      ".inst 0x4f43f088  // bfdot v8.4s, v4.8h, v3.h[0]\n"
      ".inst 0x4f63f08b  // bfdot v11.4s, v4.8h, v3.h[1]\n"
      ".inst 0x4f43f88e  // bfdot v14.4s, v4.8h, v3.h[2]\n"
      ".inst 0x4f63f891  // bfdot v17.4s, v4.8h, v3.h[3]\n"
      ".inst 0x4f47f094  // bfdot v20.4s, v4.8h, v7.h[0]\n"
      ".inst 0x4f67f097  // bfdot v23.4s, v4.8h, v7.h[1]\n"
      ".inst 0x4f47f89a  // bfdot v26.4s, v4.8h, v7.h[2]\n"
      ".inst 0x4f67f89d  // bfdot v29.4s, v4.8h, v7.h[3]\n"
      "ldr q4, [x22, #0x0]\n"
      ".inst 0x4f43f0a9  // bfdot v9.4s, v5.8h, v3.h[0]\n"
      ".inst 0x4f63f0ac  // bfdot v12.4s, v5.8h, v3.h[1]\n"
      ".inst 0x4f43f8af  // bfdot v15.4s, v5.8h, v3.h[2]\n"
      ".inst 0x4f63f8b2  // bfdot v18.4s, v5.8h, v3.h[3]\n"
      ".inst 0x4f47f0b5  // bfdot v21.4s, v5.8h, v7.h[0]\n"
      ".inst 0x4f67f0b8  // bfdot v24.4s, v5.8h, v7.h[1]\n"
      ".inst 0x4f47f8bb  // bfdot v27.4s, v5.8h, v7.h[2]\n"
      ".inst 0x4f67f8be  // bfdot v30.4s, v5.8h, v7.h[3]\n"
      "ldr q5, [x22, #0x10]\n"
      ".inst 0x4f43f04a  // bfdot v10.4s, v2.8h, v3.h[0]\n"
      ".inst 0x4f63f04d  // bfdot v13.4s, v2.8h, v3.h[1]\n"
      ".inst 0x4f43f850  // bfdot v16.4s, v2.8h, v3.h[2]\n"
      ".inst 0x4f63f853  // bfdot v19.4s, v2.8h, v3.h[3]\n"
      ".inst 0x4f47f056  // bfdot v22.4s, v2.8h, v7.h[0]\n"
      ".inst 0x4f67f059  // bfdot v25.4s, v2.8h, v7.h[1]\n"
      ".inst 0x4f47f85c  // bfdot v28.4s, v2.8h, v7.h[2]\n"
      ".inst 0x4f67f85f  // bfdot v31.4s, v2.8h, v7.h[3]\n"
      "ldr q6, [x22, #0x20]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      ".inst 0x4f40f088  // bfdot v8.4s, v4.8h, v0.h[0]\n"
      ".inst 0x4f60f08b  // bfdot v11.4s, v4.8h, v0.h[1]\n"
      "add x22, x22, #0x30\n"
      ".inst 0x4f40f88e  // bfdot v14.4s, v4.8h, v0.h[2]\n"
      ".inst 0x4f60f891  // bfdot v17.4s, v4.8h, v0.h[3]\n"
      ".inst 0x4f41f094  // bfdot v20.4s, v4.8h, v1.h[0]\n"
      ".inst 0x4f61f097  // bfdot v23.4s, v4.8h, v1.h[1]\n"
      ".inst 0x4f41f89a  // bfdot v26.4s, v4.8h, v1.h[2]\n"
      ".inst 0x4f61f89d  // bfdot v29.4s, v4.8h, v1.h[3]\n"
      ".inst 0x4f40f0a9  // bfdot v9.4s, v5.8h, v0.h[0]\n"
      ".inst 0x4f60f0ac  // bfdot v12.4s, v5.8h, v0.h[1]\n"
      ".inst 0x4f40f8af  // bfdot v15.4s, v5.8h, v0.h[2]\n"
      ".inst 0x4f60f8b2  // bfdot v18.4s, v5.8h, v0.h[3]\n"
      ".inst 0x4f41f0b5  // bfdot v21.4s, v5.8h, v1.h[0]\n"
      ".inst 0x4f61f0b8  // bfdot v24.4s, v5.8h, v1.h[1]\n"
      ".inst 0x4f41f8bb  // bfdot v27.4s, v5.8h, v1.h[2]\n"
      ".inst 0x4f61f8be  // bfdot v30.4s, v5.8h, v1.h[3]\n"
      ".inst 0x4f40f0ca  // bfdot v10.4s, v6.8h, v0.h[0]\n"
      ".inst 0x4f60f0cd  // bfdot v13.4s, v6.8h, v0.h[1]\n"
      ".inst 0x4f40f8d0  // bfdot v16.4s, v6.8h, v0.h[2]\n"
      ".inst 0x4f60f8d3  // bfdot v19.4s, v6.8h, v0.h[3]\n"
      ".inst 0x4f41f0d6  // bfdot v22.4s, v6.8h, v1.h[0]\n"
      ".inst 0x4f61f0d9  // bfdot v25.4s, v6.8h, v1.h[1]\n"
      ".inst 0x4f41f8dc  // bfdot v28.4s, v6.8h, v1.h[2]\n"
      ".inst 0x4f61f8df  // bfdot v31.4s, v6.8h, v1.h[3]\n"
      "cbz x20, 5f\n"
      "ldr q4, [%x[Apanel], #0x0]\n"
      "ldr q3, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q2, [x22, #0x0]\n"
      "ldr q1, [x22, #0x10]\n"
      ".inst 0x4f44f048  // bfdot v8.4s, v2.8h, v4.h[0]\n"
      "ldr q0, [x22, #0x20]\n"
      ".inst 0x4f64f04b  // bfdot v11.4s, v2.8h, v4.h[1]\n"
      ".inst 0x4f44f84e  // bfdot v14.4s, v2.8h, v4.h[2]\n"
      ".inst 0x4f64f851  // bfdot v17.4s, v2.8h, v4.h[3]\n"
      ".inst 0x4f43f054  // bfdot v20.4s, v2.8h, v3.h[0]\n"
      "add x22, x22, #0x30\n"
      ".inst 0x4f63f057  // bfdot v23.4s, v2.8h, v3.h[1]\n"
      ".inst 0x4f43f85a  // bfdot v26.4s, v2.8h, v3.h[2]\n"
      ".inst 0x4f63f85d  // bfdot v29.4s, v2.8h, v3.h[3]\n"
      ".inst 0x4f44f029  // bfdot v9.4s, v1.8h, v4.h[0]\n"
      ".inst 0x4f64f02c  // bfdot v12.4s, v1.8h, v4.h[1]\n"
      ".inst 0x4f44f82f  // bfdot v15.4s, v1.8h, v4.h[2]\n"
      ".inst 0x4f64f832  // bfdot v18.4s, v1.8h, v4.h[3]\n"
      ".inst 0x4f43f035  // bfdot v21.4s, v1.8h, v3.h[0]\n"
      ".inst 0x4f63f038  // bfdot v24.4s, v1.8h, v3.h[1]\n"
      ".inst 0x4f43f83b  // bfdot v27.4s, v1.8h, v3.h[2]\n"
      ".inst 0x4f63f83e  // bfdot v30.4s, v1.8h, v3.h[3]\n"
      ".inst 0x4f44f00a  // bfdot v10.4s, v0.8h, v4.h[0]\n"
      ".inst 0x4f64f00d  // bfdot v13.4s, v0.8h, v4.h[1]\n"
      ".inst 0x4f44f810  // bfdot v16.4s, v0.8h, v4.h[2]\n"
      ".inst 0x4f64f813  // bfdot v19.4s, v0.8h, v4.h[3]\n"
      ".inst 0x4f43f016  // bfdot v22.4s, v0.8h, v3.h[0]\n"
      ".inst 0x4f63f019  // bfdot v25.4s, v0.8h, v3.h[1]\n"
      ".inst 0x4f43f81c  // bfdot v28.4s, v0.8h, v3.h[2]\n"
      ".inst 0x4f63f81f  // bfdot v31.4s, v0.8h, v3.h[3]\n"
      "5:"  // multiply loop done
      "subs x23, x23, #0x1\n"
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
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23"
    );
}

} // namespace arm_gemm
#endif // __aarch64__
