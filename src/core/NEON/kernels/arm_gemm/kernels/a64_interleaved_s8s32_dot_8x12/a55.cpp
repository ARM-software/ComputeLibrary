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
#ifdef __aarch64__

#include <cstddef>
#include <cstdint>

namespace arm_gemm {

void a64_interleaved_s8s32_dot_8x12_a55(
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

      "1:"  // Height loop
      "ldr x27, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x26, %x[Apanel]\n"
      "ldr x25, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "ldr x24, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x26\n"
      "cmp x24, #0x2\n"
      "movi v8.4s, #0x0\n"
      "movi v9.4s, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "movi v10.4s, #0x0\n"
      "prfm pldl1keep, [x25, #0x0]\n"
      "movi v11.4s, #0x0\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "movi v12.4s, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x40]\n"
      "movi v13.4s, #0x0\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "movi v14.4s, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "movi v15.4s, #0x0\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v16.4s, #0x0\n"
      "ldr q4, [x25, #0x0]\n"
      "movi v17.4s, #0x0\n"
      "ldr q5, [x25, #0x10]\n"
      "movi v18.4s, #0x0\n"
      "ldr q6, [x25, #0x20]\n"
      "movi v19.4s, #0x0\n"
      "movi v20.4s, #0x0\n"
      "movi v21.4s, #0x0\n"
      "movi v22.4s, #0x0\n"
      "movi v23.4s, #0x0\n"
      "movi v24.4s, #0x0\n"
      "movi v25.4s, #0x0\n"
      "movi v26.4s, #0x0\n"
      "movi v27.4s, #0x0\n"
      "movi v28.4s, #0x0\n"
      "movi v29.4s, #0x0\n"
      "movi v30.4s, #0x0\n"
      "movi v31.4s, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      ".inst 0x4f80e088  // sdot v8.4s, v4.16b, v0.4b[0]\n"
      "ldr d2, [%x[Apanel], #0x20]\n"
      "ldr x23, [%x[Apanel], #0x28]\n"
      ".inst 0x4fa0e08b  // sdot v11.4s, v4.16b, v0.4b[1]\n"
      "ldr d3, [%x[Apanel], #0x30]\n"
      ".inst 0x4f80e88e  // sdot v14.4s, v4.16b, v0.4b[2]\n"
      "ldr x19, [%x[Apanel], #0x38]\n"
      ".inst 0x4fa0e891  // sdot v17.4s, v4.16b, v0.4b[3]\n"
      ".inst 0x4f81e094  // sdot v20.4s, v4.16b, v1.4b[0]\n"
      "ldr x22, [x25, #0x38]\n"
      ".inst 0x4fa1e097  // sdot v23.4s, v4.16b, v1.4b[1]\n"
      "ldr x20, [x25, #0x48]\n"
      ".inst 0x4f81e89a  // sdot v26.4s, v4.16b, v1.4b[2]\n"
      "ldr x21, [x25, #0x58]\n"
      ".inst 0x4fa1e89d  // sdot v29.4s, v4.16b, v1.4b[3]\n"
      "ldr d4, [x25, #0x30]\n"
      ".inst 0x4f80e0a9  // sdot v9.4s, v5.16b, v0.4b[0]\n"
      "mov v2.d[1], x23\n"
      ".inst 0x4fa0e0ac  // sdot v12.4s, v5.16b, v0.4b[1]\n"
      "mov v3.d[1], x19\n"
      ".inst 0x4f80e8af  // sdot v15.4s, v5.16b, v0.4b[2]\n"
      "mov v4.d[1], x22\n"
      ".inst 0x4fa0e8b2  // sdot v18.4s, v5.16b, v0.4b[3]\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      ".inst 0x4f81e0b5  // sdot v21.4s, v5.16b, v1.4b[0]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x4fa1e0b8  // sdot v24.4s, v5.16b, v1.4b[1]\n"
      "prfm pldl1keep, [x25, #0x100]\n"
      ".inst 0x4f81e8bb  // sdot v27.4s, v5.16b, v1.4b[2]\n"
      "prfm pldl1keep, [x25, #0x140]\n"
      ".inst 0x4fa1e8be  // sdot v30.4s, v5.16b, v1.4b[3]\n"
      "ldr d5, [x25, #0x40]\n"
      ".inst 0x4f80e0ca  // sdot v10.4s, v6.16b, v0.4b[0]\n"
      "mov v5.d[1], x20\n"
      ".inst 0x4fa0e0cd  // sdot v13.4s, v6.16b, v0.4b[1]\n"
      "ldr x20, [%x[Apanel], #0x8]\n"
      ".inst 0x4f80e8d0  // sdot v16.4s, v6.16b, v0.4b[2]\n"
      "ldr x19, [%x[Apanel], #0x18]\n"
      ".inst 0x4fa0e8d3  // sdot v19.4s, v6.16b, v0.4b[3]\n"
      "ldr d0, [%x[Apanel], #0x0]\n"
      ".inst 0x4f81e0d6  // sdot v22.4s, v6.16b, v1.4b[0]\n"
      "sub x24, x24, #0x2\n"
      ".inst 0x4fa1e0d9  // sdot v25.4s, v6.16b, v1.4b[1]\n"
      "cmp x24, #0x2\n"
      ".inst 0x4f81e8dc  // sdot v28.4s, v6.16b, v1.4b[2]\n"
      "mov v0.d[1], x20\n"
      ".inst 0x4fa1e8df  // sdot v31.4s, v6.16b, v1.4b[3]\n"
      "ldr d6, [x25, #0x50]\n"
      "mov v6.d[1], x21\n"
      "add x25, x25, #0x60\n"
      ".inst 0x4f82e088  // sdot v8.4s, v4.16b, v2.4b[0]\n"
      "ldr d1, [%x[Apanel], #0x10]\n"
      ".inst 0x4fa2e08b  // sdot v11.4s, v4.16b, v2.4b[1]\n"
      "ldr x22, [x25, #0x8]\n"
      ".inst 0x4f82e88e  // sdot v14.4s, v4.16b, v2.4b[2]\n"
      "ldr x20, [x25, #0x18]\n"
      ".inst 0x4fa2e891  // sdot v17.4s, v4.16b, v2.4b[3]\n"
      "ldr x21, [x25, #0x28]\n"
      ".inst 0x4f83e094  // sdot v20.4s, v4.16b, v3.4b[0]\n"
      "mov v1.d[1], x19\n"
      ".inst 0x4fa3e097  // sdot v23.4s, v4.16b, v3.4b[1]\n"
      ".inst 0x4f83e89a  // sdot v26.4s, v4.16b, v3.4b[2]\n"
      ".inst 0x4fa3e89d  // sdot v29.4s, v4.16b, v3.4b[3]\n"
      "ldr d4, [x25, #0x0]\n"
      ".inst 0x4f82e0a9  // sdot v9.4s, v5.16b, v2.4b[0]\n"
      "mov v4.d[1], x22\n"
      ".inst 0x4fa2e0ac  // sdot v12.4s, v5.16b, v2.4b[1]\n"
      ".inst 0x4f82e8af  // sdot v15.4s, v5.16b, v2.4b[2]\n"
      ".inst 0x4fa2e8b2  // sdot v18.4s, v5.16b, v2.4b[3]\n"
      ".inst 0x4f83e0b5  // sdot v21.4s, v5.16b, v3.4b[0]\n"
      ".inst 0x4fa3e0b8  // sdot v24.4s, v5.16b, v3.4b[1]\n"
      ".inst 0x4f83e8bb  // sdot v27.4s, v5.16b, v3.4b[2]\n"
      ".inst 0x4fa3e8be  // sdot v30.4s, v5.16b, v3.4b[3]\n"
      "ldr d5, [x25, #0x10]\n"
      ".inst 0x4f82e0ca  // sdot v10.4s, v6.16b, v2.4b[0]\n"
      "mov v5.d[1], x20\n"
      ".inst 0x4fa2e0cd  // sdot v13.4s, v6.16b, v2.4b[1]\n"
      ".inst 0x4f82e8d0  // sdot v16.4s, v6.16b, v2.4b[2]\n"
      ".inst 0x4fa2e8d3  // sdot v19.4s, v6.16b, v2.4b[3]\n"
      ".inst 0x4f83e0d6  // sdot v22.4s, v6.16b, v3.4b[0]\n"
      ".inst 0x4fa3e0d9  // sdot v25.4s, v6.16b, v3.4b[1]\n"
      ".inst 0x4f83e8dc  // sdot v28.4s, v6.16b, v3.4b[2]\n"
      ".inst 0x4fa3e8df  // sdot v31.4s, v6.16b, v3.4b[3]\n"
      "ldr d6, [x25, #0x20]\n"
      "mov v6.d[1], x21\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      ".inst 0x4f80e088  // sdot v8.4s, v4.16b, v0.4b[0]\n"
      "add x25, x25, #0x30\n"
      ".inst 0x4fa0e08b  // sdot v11.4s, v4.16b, v0.4b[1]\n"
      ".inst 0x4f80e88e  // sdot v14.4s, v4.16b, v0.4b[2]\n"
      ".inst 0x4fa0e891  // sdot v17.4s, v4.16b, v0.4b[3]\n"
      ".inst 0x4f81e094  // sdot v20.4s, v4.16b, v1.4b[0]\n"
      ".inst 0x4fa1e097  // sdot v23.4s, v4.16b, v1.4b[1]\n"
      ".inst 0x4f81e89a  // sdot v26.4s, v4.16b, v1.4b[2]\n"
      ".inst 0x4fa1e89d  // sdot v29.4s, v4.16b, v1.4b[3]\n"
      ".inst 0x4f80e0a9  // sdot v9.4s, v5.16b, v0.4b[0]\n"
      ".inst 0x4fa0e0ac  // sdot v12.4s, v5.16b, v0.4b[1]\n"
      ".inst 0x4f80e8af  // sdot v15.4s, v5.16b, v0.4b[2]\n"
      ".inst 0x4fa0e8b2  // sdot v18.4s, v5.16b, v0.4b[3]\n"
      ".inst 0x4f81e0b5  // sdot v21.4s, v5.16b, v1.4b[0]\n"
      ".inst 0x4fa1e0b8  // sdot v24.4s, v5.16b, v1.4b[1]\n"
      ".inst 0x4f81e8bb  // sdot v27.4s, v5.16b, v1.4b[2]\n"
      ".inst 0x4fa1e8be  // sdot v30.4s, v5.16b, v1.4b[3]\n"
      ".inst 0x4f80e0ca  // sdot v10.4s, v6.16b, v0.4b[0]\n"
      ".inst 0x4fa0e0cd  // sdot v13.4s, v6.16b, v0.4b[1]\n"
      ".inst 0x4f80e8d0  // sdot v16.4s, v6.16b, v0.4b[2]\n"
      ".inst 0x4fa0e8d3  // sdot v19.4s, v6.16b, v0.4b[3]\n"
      ".inst 0x4f81e0d6  // sdot v22.4s, v6.16b, v1.4b[0]\n"
      ".inst 0x4fa1e0d9  // sdot v25.4s, v6.16b, v1.4b[1]\n"
      ".inst 0x4f81e8dc  // sdot v28.4s, v6.16b, v1.4b[2]\n"
      ".inst 0x4fa1e8df  // sdot v31.4s, v6.16b, v1.4b[3]\n"
      "cbz x24, 5f\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q7, [x25, #0x0]\n"
      ".inst 0x4f80e0e8  // sdot v8.4s, v7.16b, v0.4b[0]\n"
      "ldr q4, [x25, #0x10]\n"
      ".inst 0x4fa0e0eb  // sdot v11.4s, v7.16b, v0.4b[1]\n"
      "ldr q5, [x25, #0x20]\n"
      ".inst 0x4f80e8ee  // sdot v14.4s, v7.16b, v0.4b[2]\n"
      ".inst 0x4fa0e8f1  // sdot v17.4s, v7.16b, v0.4b[3]\n"
      "add x25, x25, #0x30\n"
      ".inst 0x4f81e0f4  // sdot v20.4s, v7.16b, v1.4b[0]\n"
      ".inst 0x4fa1e0f7  // sdot v23.4s, v7.16b, v1.4b[1]\n"
      ".inst 0x4f81e8fa  // sdot v26.4s, v7.16b, v1.4b[2]\n"
      ".inst 0x4fa1e8fd  // sdot v29.4s, v7.16b, v1.4b[3]\n"
      ".inst 0x4f80e089  // sdot v9.4s, v4.16b, v0.4b[0]\n"
      ".inst 0x4fa0e08c  // sdot v12.4s, v4.16b, v0.4b[1]\n"
      ".inst 0x4f80e88f  // sdot v15.4s, v4.16b, v0.4b[2]\n"
      ".inst 0x4fa0e892  // sdot v18.4s, v4.16b, v0.4b[3]\n"
      ".inst 0x4f81e095  // sdot v21.4s, v4.16b, v1.4b[0]\n"
      ".inst 0x4fa1e098  // sdot v24.4s, v4.16b, v1.4b[1]\n"
      ".inst 0x4f81e89b  // sdot v27.4s, v4.16b, v1.4b[2]\n"
      ".inst 0x4fa1e89e  // sdot v30.4s, v4.16b, v1.4b[3]\n"
      ".inst 0x4f80e0aa  // sdot v10.4s, v5.16b, v0.4b[0]\n"
      ".inst 0x4fa0e0ad  // sdot v13.4s, v5.16b, v0.4b[1]\n"
      ".inst 0x4f80e8b0  // sdot v16.4s, v5.16b, v0.4b[2]\n"
      ".inst 0x4fa0e8b3  // sdot v19.4s, v5.16b, v0.4b[3]\n"
      ".inst 0x4f81e0b6  // sdot v22.4s, v5.16b, v1.4b[0]\n"
      ".inst 0x4fa1e0b9  // sdot v25.4s, v5.16b, v1.4b[1]\n"
      ".inst 0x4f81e8bc  // sdot v28.4s, v5.16b, v1.4b[2]\n"
      ".inst 0x4fa1e8bf  // sdot v31.4s, v5.16b, v1.4b[3]\n"
      "5:"  // multiply loop done
      "subs x27, x27, #0x1\n"
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
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27"
    );
}

} // namespace arm_gemm
#endif // __aarch64__
