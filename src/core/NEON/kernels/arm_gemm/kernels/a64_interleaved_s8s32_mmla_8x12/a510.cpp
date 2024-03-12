/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#include <cstdint>

namespace arm_gemm {

void a64_interleaved_s8s32_mmla_8x12_a510(
    const int8_t *Apanel,
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
      "1:"  // Height loop
      "ldr x23, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x21, %x[Apanel]\n"
      "2:"  // Width loop
      "ldp q4, q5, [x22], #0x20\n"
      "mov %x[Apanel], x21\n"
      "ld1 { v0.16b }, [%x[Apanel]], #0x10\n"
      "ld1 { v1.16b }, [%x[Apanel]], #0x10\n"
      "movi v8.4s, #0x0\n"
      "ld1 { v2.16b }, [%x[Apanel]], #0x10\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "cmp x20, #0x2\n"
      "movi v9.4s, #0x0\n"
      "movi v10.4s, #0x0\n"
      "movi v11.4s, #0x0\n"
      "movi v12.4s, #0x0\n"
      "movi v13.4s, #0x0\n"
      "movi v14.4s, #0x0\n"
      "movi v15.4s, #0x0\n"
      "movi v16.4s, #0x0\n"
      "movi v17.4s, #0x0\n"
      "movi v18.4s, #0x0\n"
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
      "ld1 { v6.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e84a408  // smmla v8.4s, v0.16b, v4.16b\n"
      ".inst 0x4e85a40b  // smmla v11.4s, v0.16b, v5.16b\n"
      "ldp q3, q7, [x22], #0x20\n"
      ".inst 0x4e84a42e  // smmla v14.4s, v1.16b, v4.16b\n"
      ".inst 0x4e85a431  // smmla v17.4s, v1.16b, v5.16b\n"
      ".inst 0x4e84a454  // smmla v20.4s, v2.16b, v4.16b\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x4e85a457  // smmla v23.4s, v2.16b, v5.16b\n"
      ".inst 0x4e84a4da  // smmla v26.4s, v6.16b, v4.16b\n"
      "cmp x20, #0x2\n"
      ".inst 0x4e85a4dd  // smmla v29.4s, v6.16b, v5.16b\n"
      "ldp q4, q5, [x22], #0x20\n"
      ".inst 0x4e83a409  // smmla v9.4s, v0.16b, v3.16b\n"
      ".inst 0x4e87a40c  // smmla v12.4s, v0.16b, v7.16b\n"
      ".inst 0x4e83a42f  // smmla v15.4s, v1.16b, v3.16b\n"
      ".inst 0x4e87a432  // smmla v18.4s, v1.16b, v7.16b\n"
      ".inst 0x4e83a455  // smmla v21.4s, v2.16b, v3.16b\n"
      ".inst 0x4e87a458  // smmla v24.4s, v2.16b, v7.16b\n"
      ".inst 0x4e83a4db  // smmla v27.4s, v6.16b, v3.16b\n"
      ".inst 0x4e87a4de  // smmla v30.4s, v6.16b, v7.16b\n"
      "ldp q7, q3, [x22], #0x20\n"
      ".inst 0x4e84a40a  // smmla v10.4s, v0.16b, v4.16b\n"
      ".inst 0x4e85a40d  // smmla v13.4s, v0.16b, v5.16b\n"
      "ld1 { v0.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e84a430  // smmla v16.4s, v1.16b, v4.16b\n"
      ".inst 0x4e85a433  // smmla v19.4s, v1.16b, v5.16b\n"
      "ld1 { v1.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e84a456  // smmla v22.4s, v2.16b, v4.16b\n"
      ".inst 0x4e85a459  // smmla v25.4s, v2.16b, v5.16b\n"
      "ld1 { v2.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e84a4dc  // smmla v28.4s, v6.16b, v4.16b\n"
      ".inst 0x4e85a4df  // smmla v31.4s, v6.16b, v5.16b\n"
      "ld1 { v6.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e87a408  // smmla v8.4s, v0.16b, v7.16b\n"
      "ldp q4, q5, [x22], #0x20\n"
      ".inst 0x4e83a40b  // smmla v11.4s, v0.16b, v3.16b\n"
      ".inst 0x4e87a42e  // smmla v14.4s, v1.16b, v7.16b\n"
      ".inst 0x4e83a431  // smmla v17.4s, v1.16b, v3.16b\n"
      ".inst 0x4e87a454  // smmla v20.4s, v2.16b, v7.16b\n"
      ".inst 0x4e83a457  // smmla v23.4s, v2.16b, v3.16b\n"
      ".inst 0x4e87a4da  // smmla v26.4s, v6.16b, v7.16b\n"
      ".inst 0x4e83a4dd  // smmla v29.4s, v6.16b, v3.16b\n"
      "ldp q7, q3, [x22], #0x20\n"
      ".inst 0x4e84a409  // smmla v9.4s, v0.16b, v4.16b\n"
      ".inst 0x4e85a40c  // smmla v12.4s, v0.16b, v5.16b\n"
      ".inst 0x4e84a42f  // smmla v15.4s, v1.16b, v4.16b\n"
      ".inst 0x4e85a432  // smmla v18.4s, v1.16b, v5.16b\n"
      ".inst 0x4e84a455  // smmla v21.4s, v2.16b, v4.16b\n"
      ".inst 0x4e85a458  // smmla v24.4s, v2.16b, v5.16b\n"
      ".inst 0x4e84a4db  // smmla v27.4s, v6.16b, v4.16b\n"
      ".inst 0x4e85a4de  // smmla v30.4s, v6.16b, v5.16b\n"
      "ldp q4, q5, [x22], #0x20\n"
      ".inst 0x4e87a40a  // smmla v10.4s, v0.16b, v7.16b\n"
      ".inst 0x4e83a40d  // smmla v13.4s, v0.16b, v3.16b\n"
      "ld1 { v0.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e87a430  // smmla v16.4s, v1.16b, v7.16b\n"
      ".inst 0x4e83a433  // smmla v19.4s, v1.16b, v3.16b\n"
      "ld1 { v1.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e87a456  // smmla v22.4s, v2.16b, v7.16b\n"
      ".inst 0x4e83a459  // smmla v25.4s, v2.16b, v3.16b\n"
      "ld1 { v2.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e87a4dc  // smmla v28.4s, v6.16b, v7.16b\n"
      ".inst 0x4e83a4df  // smmla v31.4s, v6.16b, v3.16b\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "ld1 { v3.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e84a408  // smmla v8.4s, v0.16b, v4.16b\n"
      ".inst 0x4e85a40b  // smmla v11.4s, v0.16b, v5.16b\n"
      "ldp q6, q7, [x22], #0x20\n"
      ".inst 0x4e84a42e  // smmla v14.4s, v1.16b, v4.16b\n"
      ".inst 0x4e85a431  // smmla v17.4s, v1.16b, v5.16b\n"
      ".inst 0x4e84a454  // smmla v20.4s, v2.16b, v4.16b\n"
      ".inst 0x4e85a457  // smmla v23.4s, v2.16b, v5.16b\n"
      ".inst 0x4e84a47a  // smmla v26.4s, v3.16b, v4.16b\n"
      ".inst 0x4e85a47d  // smmla v29.4s, v3.16b, v5.16b\n"
      "ldp q5, q4, [x22], #0x20\n"
      ".inst 0x4e86a409  // smmla v9.4s, v0.16b, v6.16b\n"
      ".inst 0x4e87a40c  // smmla v12.4s, v0.16b, v7.16b\n"
      ".inst 0x4e86a42f  // smmla v15.4s, v1.16b, v6.16b\n"
      ".inst 0x4e87a432  // smmla v18.4s, v1.16b, v7.16b\n"
      ".inst 0x4e86a455  // smmla v21.4s, v2.16b, v6.16b\n"
      ".inst 0x4e87a458  // smmla v24.4s, v2.16b, v7.16b\n"
      ".inst 0x4e86a47b  // smmla v27.4s, v3.16b, v6.16b\n"
      ".inst 0x4e87a47e  // smmla v30.4s, v3.16b, v7.16b\n"
      ".inst 0x4e85a40a  // smmla v10.4s, v0.16b, v5.16b\n"
      ".inst 0x4e84a40d  // smmla v13.4s, v0.16b, v4.16b\n"
      ".inst 0x4e85a430  // smmla v16.4s, v1.16b, v5.16b\n"
      ".inst 0x4e84a433  // smmla v19.4s, v1.16b, v4.16b\n"
      ".inst 0x4e85a456  // smmla v22.4s, v2.16b, v5.16b\n"
      ".inst 0x4e84a459  // smmla v25.4s, v2.16b, v4.16b\n"
      ".inst 0x4e85a47c  // smmla v28.4s, v3.16b, v5.16b\n"
      ".inst 0x4e84a47f  // smmla v31.4s, v3.16b, v4.16b\n"
      "cbz x20, 5f\n"
      "ldp q1, q0, [x22], #0x20\n"
      "ld1 { v7.16b }, [%x[Apanel]], #0x10\n"
      "ld1 { v6.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e81a4e8  // smmla v8.4s, v7.16b, v1.16b\n"
      "ld1 { v5.16b }, [%x[Apanel]], #0x10\n"
      "ld1 { v4.16b }, [%x[Apanel]], #0x10\n"
      ".inst 0x4e80a4eb  // smmla v11.4s, v7.16b, v0.16b\n"
      "ldp q3, q2, [x22], #0x20\n"
      ".inst 0x4e81a4ce  // smmla v14.4s, v6.16b, v1.16b\n"
      ".inst 0x4e80a4d1  // smmla v17.4s, v6.16b, v0.16b\n"
      ".inst 0x4e81a4b4  // smmla v20.4s, v5.16b, v1.16b\n"
      ".inst 0x4e80a4b7  // smmla v23.4s, v5.16b, v0.16b\n"
      ".inst 0x4e81a49a  // smmla v26.4s, v4.16b, v1.16b\n"
      ".inst 0x4e80a49d  // smmla v29.4s, v4.16b, v0.16b\n"
      "ldp q1, q0, [x22], #0x20\n"
      ".inst 0x4e83a4e9  // smmla v9.4s, v7.16b, v3.16b\n"
      ".inst 0x4e82a4ec  // smmla v12.4s, v7.16b, v2.16b\n"
      ".inst 0x4e83a4cf  // smmla v15.4s, v6.16b, v3.16b\n"
      ".inst 0x4e82a4d2  // smmla v18.4s, v6.16b, v2.16b\n"
      ".inst 0x4e83a4b5  // smmla v21.4s, v5.16b, v3.16b\n"
      ".inst 0x4e82a4b8  // smmla v24.4s, v5.16b, v2.16b\n"
      ".inst 0x4e83a49b  // smmla v27.4s, v4.16b, v3.16b\n"
      ".inst 0x4e82a49e  // smmla v30.4s, v4.16b, v2.16b\n"
      ".inst 0x4e81a4ea  // smmla v10.4s, v7.16b, v1.16b\n"
      ".inst 0x4e80a4ed  // smmla v13.4s, v7.16b, v0.16b\n"
      ".inst 0x4e81a4d0  // smmla v16.4s, v6.16b, v1.16b\n"
      ".inst 0x4e80a4d3  // smmla v19.4s, v6.16b, v0.16b\n"
      ".inst 0x4e81a4b6  // smmla v22.4s, v5.16b, v1.16b\n"
      ".inst 0x4e80a4b9  // smmla v25.4s, v5.16b, v0.16b\n"
      ".inst 0x4e81a49c  // smmla v28.4s, v4.16b, v1.16b\n"
      ".inst 0x4e80a49f  // smmla v31.4s, v4.16b, v0.16b\n"
      "5:"  // multiply loop done
      "subs x23, x23, #0x1\n"
      "uzp1 v0.2d, v8.2d, v11.2d\n"
      "uzp2 v8.2d, v8.2d, v11.2d\n"
      "uzp1 v1.2d, v9.2d, v12.2d\n"
      "uzp2 v9.2d, v9.2d, v12.2d\n"
      "str q0, [%x[Cpanel], #0x0]\n"
      "uzp1 v0.2d, v10.2d, v13.2d\n"
      "uzp2 v10.2d, v10.2d, v13.2d\n"
      "str q1, [%x[Cpanel], #0x10]\n"
      "str q0, [%x[Cpanel], #0x20]\n"
      "uzp1 v0.2d, v14.2d, v17.2d\n"
      "uzp2 v14.2d, v14.2d, v17.2d\n"
      "str q8, [%x[Cpanel], #0x30]\n"
      "uzp1 v2.2d, v15.2d, v18.2d\n"
      "uzp2 v15.2d, v15.2d, v18.2d\n"
      "str q9, [%x[Cpanel], #0x40]\n"
      "uzp1 v17.2d, v16.2d, v19.2d\n"
      "uzp2 v16.2d, v16.2d, v19.2d\n"
      "str q10, [%x[Cpanel], #0x50]\n"
      "uzp1 v1.2d, v20.2d, v23.2d\n"
      "uzp2 v20.2d, v20.2d, v23.2d\n"
      "str q0, [%x[Cpanel], #0x60]\n"
      "uzp1 v0.2d, v21.2d, v24.2d\n"
      "uzp2 v21.2d, v21.2d, v24.2d\n"
      "str q2, [%x[Cpanel], #0x70]\n"
      "uzp1 v23.2d, v22.2d, v25.2d\n"
      "uzp2 v22.2d, v22.2d, v25.2d\n"
      "str q17, [%x[Cpanel], #0x80]\n"
      "uzp1 v19.2d, v26.2d, v29.2d\n"
      "uzp2 v26.2d, v26.2d, v29.2d\n"
      "str q14, [%x[Cpanel], #0x90]\n"
      "uzp1 v18.2d, v27.2d, v30.2d\n"
      "uzp2 v27.2d, v27.2d, v30.2d\n"
      "str q15, [%x[Cpanel], #0xa0]\n"
      "uzp1 v17.2d, v28.2d, v31.2d\n"
      "uzp2 v28.2d, v28.2d, v31.2d\n"
      "str q16, [%x[Cpanel], #0xb0]\n"
      "str q1, [%x[Cpanel], #0xc0]\n"
      "str q0, [%x[Cpanel], #0xd0]\n"
      "str q23, [%x[Cpanel], #0xe0]\n"
      "str q20, [%x[Cpanel], #0xf0]\n"
      "str q21, [%x[Cpanel], #0x100]\n"
      "str q22, [%x[Cpanel], #0x110]\n"
      "str q19, [%x[Cpanel], #0x120]\n"
      "str q18, [%x[Cpanel], #0x130]\n"
      "str q17, [%x[Cpanel], #0x140]\n"
      "str q26, [%x[Cpanel], #0x150]\n"
      "str q27, [%x[Cpanel], #0x160]\n"
      "str q28, [%x[Cpanel], #0x170]\n"
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
