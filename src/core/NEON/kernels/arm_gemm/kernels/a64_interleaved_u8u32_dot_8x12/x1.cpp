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

void a64_interleaved_u8u32_dot_8x12_x1(
    const uint8_t *Apanel, const uint8_t *Bpanel,
    uint32_t *Cpanel, int ablocks, int bblocks, int K) {

    struct KernelArgs {
        size_t bblocks = {};
        size_t K = {};
        const uint8_t *Bpanel = {};
    } ka;

    ka.bblocks = bblocks;
    ka.K = (K/4) - 1;
    ka.Bpanel = Bpanel;

    __asm__ __volatile__(

      "1:"  // Height loop
      "ldr x22, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "mov x21, %x[Apanel]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "2:"  // Width loop
      "ldr x19, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x21\n"
      "cmp x19, #0x2\n"
      "movi v8.4s, #0x0\n"
      "movi v9.4s, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x0]\n"
      "movi v10.4s, #0x0\n"
      "movi v11.4s, #0x0\n"
      "prfm pldl1keep, [x20, #0x0]\n"
      "movi v12.4s, #0x0\n"
      "movi v13.4s, #0x0\n"
      "prfm pldl1keep, [x20, #0x40]\n"
      "movi v14.4s, #0x0\n"
      "movi v15.4s, #0x0\n"
      "prfm pldl1keep, [%x[Apanel], #0x40]\n"
      "movi v16.4s, #0x0\n"
      "movi v17.4s, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "movi v18.4s, #0x0\n"
      "movi v19.4s, #0x0\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "movi v20.4s, #0x0\n"
      "movi v21.4s, #0x0\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v22.4s, #0x0\n"
      "movi v23.4s, #0x0\n"
      "ldr q2, [x20, #0x0]\n"
      "movi v24.4s, #0x0\n"
      "movi v25.4s, #0x0\n"
      "ldr q3, [x20, #0x10]\n"
      "movi v26.4s, #0x0\n"
      "movi v27.4s, #0x0\n"
      "ldr q4, [x20, #0x20]\n"
      "movi v28.4s, #0x0\n"
      "movi v29.4s, #0x0\n"
      "movi v30.4s, #0x0\n"
      "movi v31.4s, #0x0\n"
      "blt 4f\n"
      "3:"  // main loop head
      ".inst 0x6f80e048  // udot v8.4s, v2.16b, v0.4b[0]\n"
      ".inst 0x6fa0e04b  // udot v11.4s, v2.16b, v0.4b[1]\n"
      "sub x19, x19, #0x2\n"
      ".inst 0x6f80e84e  // udot v14.4s, v2.16b, v0.4b[2]\n"
      ".inst 0x6fa0e851  // udot v17.4s, v2.16b, v0.4b[3]\n"
      "cmp x19, #0x2\n"
      ".inst 0x6f81e054  // udot v20.4s, v2.16b, v1.4b[0]\n"
      ".inst 0x6fa1e057  // udot v23.4s, v2.16b, v1.4b[1]\n"
      "prfm pldl1keep, [%x[Apanel], #0x80]\n"
      ".inst 0x6f81e85a  // udot v26.4s, v2.16b, v1.4b[2]\n"
      ".inst 0x6fa1e85d  // udot v29.4s, v2.16b, v1.4b[3]\n"
      "ldr q2, [x20, #0x30]\n"
      ".inst 0x6f80e069  // udot v9.4s, v3.16b, v0.4b[0]\n"
      ".inst 0x6fa0e06c  // udot v12.4s, v3.16b, v0.4b[1]\n"
      "prfm pldl1keep, [x20, #0x100]\n"
      ".inst 0x6f80e86f  // udot v15.4s, v3.16b, v0.4b[2]\n"
      ".inst 0x6fa0e872  // udot v18.4s, v3.16b, v0.4b[3]\n"
      "prfm pldl1keep, [x20, #0x140]\n"
      ".inst 0x6f81e075  // udot v21.4s, v3.16b, v1.4b[0]\n"
      ".inst 0x6fa1e078  // udot v24.4s, v3.16b, v1.4b[1]\n"
      ".inst 0x6f81e87b  // udot v27.4s, v3.16b, v1.4b[2]\n"
      ".inst 0x6fa1e87e  // udot v30.4s, v3.16b, v1.4b[3]\n"
      "ldr q3, [x20, #0x40]\n"
      ".inst 0x6f80e08a  // udot v10.4s, v4.16b, v0.4b[0]\n"
      ".inst 0x6fa0e08d  // udot v13.4s, v4.16b, v0.4b[1]\n"
      ".inst 0x6f80e890  // udot v16.4s, v4.16b, v0.4b[2]\n"
      ".inst 0x6fa0e893  // udot v19.4s, v4.16b, v0.4b[3]\n"
      "ldr q0, [%x[Apanel], #0x20]\n"
      ".inst 0x6f81e096  // udot v22.4s, v4.16b, v1.4b[0]\n"
      ".inst 0x6fa1e099  // udot v25.4s, v4.16b, v1.4b[1]\n"
      ".inst 0x6f81e89c  // udot v28.4s, v4.16b, v1.4b[2]\n"
      ".inst 0x6fa1e89f  // udot v31.4s, v4.16b, v1.4b[3]\n"
      "ldr q1, [%x[Apanel], #0x30]\n"
      "ldr q4, [x20, #0x50]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "add x20, x20, #0x60\n"
      ".inst 0x6f80e048  // udot v8.4s, v2.16b, v0.4b[0]\n"
      ".inst 0x6fa0e04b  // udot v11.4s, v2.16b, v0.4b[1]\n"
      ".inst 0x6f80e84e  // udot v14.4s, v2.16b, v0.4b[2]\n"
      ".inst 0x6fa0e851  // udot v17.4s, v2.16b, v0.4b[3]\n"
      ".inst 0x6f81e054  // udot v20.4s, v2.16b, v1.4b[0]\n"
      ".inst 0x6fa1e057  // udot v23.4s, v2.16b, v1.4b[1]\n"
      ".inst 0x6f81e85a  // udot v26.4s, v2.16b, v1.4b[2]\n"
      ".inst 0x6fa1e85d  // udot v29.4s, v2.16b, v1.4b[3]\n"
      "ldr q2, [x20, #0x0]\n"
      ".inst 0x6f80e069  // udot v9.4s, v3.16b, v0.4b[0]\n"
      ".inst 0x6fa0e06c  // udot v12.4s, v3.16b, v0.4b[1]\n"
      ".inst 0x6f80e86f  // udot v15.4s, v3.16b, v0.4b[2]\n"
      ".inst 0x6fa0e872  // udot v18.4s, v3.16b, v0.4b[3]\n"
      ".inst 0x6f81e075  // udot v21.4s, v3.16b, v1.4b[0]\n"
      ".inst 0x6fa1e078  // udot v24.4s, v3.16b, v1.4b[1]\n"
      ".inst 0x6f81e87b  // udot v27.4s, v3.16b, v1.4b[2]\n"
      ".inst 0x6fa1e87e  // udot v30.4s, v3.16b, v1.4b[3]\n"
      "ldr q3, [x20, #0x10]\n"
      ".inst 0x6f80e08a  // udot v10.4s, v4.16b, v0.4b[0]\n"
      ".inst 0x6fa0e08d  // udot v13.4s, v4.16b, v0.4b[1]\n"
      ".inst 0x6f80e890  // udot v16.4s, v4.16b, v0.4b[2]\n"
      ".inst 0x6fa0e893  // udot v19.4s, v4.16b, v0.4b[3]\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      ".inst 0x6f81e096  // udot v22.4s, v4.16b, v1.4b[0]\n"
      ".inst 0x6fa1e099  // udot v25.4s, v4.16b, v1.4b[1]\n"
      ".inst 0x6f81e89c  // udot v28.4s, v4.16b, v1.4b[2]\n"
      ".inst 0x6fa1e89f  // udot v31.4s, v4.16b, v1.4b[3]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "ldr q4, [x20, #0x20]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      ".inst 0x6f80e048  // udot v8.4s, v2.16b, v0.4b[0]\n"
      ".inst 0x6fa0e04b  // udot v11.4s, v2.16b, v0.4b[1]\n"
      "add x20, x20, #0x30\n"
      ".inst 0x6f80e84e  // udot v14.4s, v2.16b, v0.4b[2]\n"
      ".inst 0x6fa0e851  // udot v17.4s, v2.16b, v0.4b[3]\n"
      ".inst 0x6f81e054  // udot v20.4s, v2.16b, v1.4b[0]\n"
      ".inst 0x6fa1e057  // udot v23.4s, v2.16b, v1.4b[1]\n"
      ".inst 0x6f81e85a  // udot v26.4s, v2.16b, v1.4b[2]\n"
      ".inst 0x6fa1e85d  // udot v29.4s, v2.16b, v1.4b[3]\n"
      ".inst 0x6f80e069  // udot v9.4s, v3.16b, v0.4b[0]\n"
      ".inst 0x6fa0e06c  // udot v12.4s, v3.16b, v0.4b[1]\n"
      ".inst 0x6f80e86f  // udot v15.4s, v3.16b, v0.4b[2]\n"
      ".inst 0x6fa0e872  // udot v18.4s, v3.16b, v0.4b[3]\n"
      ".inst 0x6f81e075  // udot v21.4s, v3.16b, v1.4b[0]\n"
      ".inst 0x6fa1e078  // udot v24.4s, v3.16b, v1.4b[1]\n"
      ".inst 0x6f81e87b  // udot v27.4s, v3.16b, v1.4b[2]\n"
      ".inst 0x6fa1e87e  // udot v30.4s, v3.16b, v1.4b[3]\n"
      ".inst 0x6f80e08a  // udot v10.4s, v4.16b, v0.4b[0]\n"
      ".inst 0x6fa0e08d  // udot v13.4s, v4.16b, v0.4b[1]\n"
      ".inst 0x6f80e890  // udot v16.4s, v4.16b, v0.4b[2]\n"
      ".inst 0x6fa0e893  // udot v19.4s, v4.16b, v0.4b[3]\n"
      ".inst 0x6f81e096  // udot v22.4s, v4.16b, v1.4b[0]\n"
      ".inst 0x6fa1e099  // udot v25.4s, v4.16b, v1.4b[1]\n"
      ".inst 0x6f81e89c  // udot v28.4s, v4.16b, v1.4b[2]\n"
      ".inst 0x6fa1e89f  // udot v31.4s, v4.16b, v1.4b[3]\n"
      "cbz x19, 5f\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q5, [x20, #0x0]\n"
      "ldr q6, [x20, #0x10]\n"
      ".inst 0x6f80e0a8  // udot v8.4s, v5.16b, v0.4b[0]\n"
      "ldr q7, [x20, #0x20]\n"
      ".inst 0x6fa0e0ab  // udot v11.4s, v5.16b, v0.4b[1]\n"
      ".inst 0x6f80e8ae  // udot v14.4s, v5.16b, v0.4b[2]\n"
      "add x20, x20, #0x30\n"
      ".inst 0x6fa0e8b1  // udot v17.4s, v5.16b, v0.4b[3]\n"
      ".inst 0x6f81e0b4  // udot v20.4s, v5.16b, v1.4b[0]\n"
      ".inst 0x6fa1e0b7  // udot v23.4s, v5.16b, v1.4b[1]\n"
      ".inst 0x6f81e8ba  // udot v26.4s, v5.16b, v1.4b[2]\n"
      ".inst 0x6fa1e8bd  // udot v29.4s, v5.16b, v1.4b[3]\n"
      ".inst 0x6f80e0c9  // udot v9.4s, v6.16b, v0.4b[0]\n"
      ".inst 0x6fa0e0cc  // udot v12.4s, v6.16b, v0.4b[1]\n"
      ".inst 0x6f80e8cf  // udot v15.4s, v6.16b, v0.4b[2]\n"
      ".inst 0x6fa0e8d2  // udot v18.4s, v6.16b, v0.4b[3]\n"
      ".inst 0x6f81e0d5  // udot v21.4s, v6.16b, v1.4b[0]\n"
      ".inst 0x6fa1e0d8  // udot v24.4s, v6.16b, v1.4b[1]\n"
      ".inst 0x6f81e8db  // udot v27.4s, v6.16b, v1.4b[2]\n"
      ".inst 0x6fa1e8de  // udot v30.4s, v6.16b, v1.4b[3]\n"
      ".inst 0x6f80e0ea  // udot v10.4s, v7.16b, v0.4b[0]\n"
      ".inst 0x6fa0e0ed  // udot v13.4s, v7.16b, v0.4b[1]\n"
      ".inst 0x6f80e8f0  // udot v16.4s, v7.16b, v0.4b[2]\n"
      ".inst 0x6fa0e8f3  // udot v19.4s, v7.16b, v0.4b[3]\n"
      ".inst 0x6f81e0f6  // udot v22.4s, v7.16b, v1.4b[0]\n"
      ".inst 0x6fa1e0f9  // udot v25.4s, v7.16b, v1.4b[1]\n"
      ".inst 0x6f81e8fc  // udot v28.4s, v7.16b, v1.4b[2]\n"
      ".inst 0x6fa1e8ff  // udot v31.4s, v7.16b, v1.4b[3]\n"
      "5:"  // multiply loop done
      "subs x22, x22, #0x1\n"
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
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22"
    );
}

} // namespace arm_gemm
#endif // __aarch64__
