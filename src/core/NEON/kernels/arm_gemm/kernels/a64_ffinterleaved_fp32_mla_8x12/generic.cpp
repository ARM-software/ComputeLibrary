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
#ifdef __aarch64__

#include <cstddef>

namespace arm_gemm {

void a64_ffinterleaved_fp32_mla_8x12(
    const float *Apanel,
    const float *Bpanel,
    size_t B_stride,
    float *Cpanel,
    int ablocks,
    size_t N,
    int K) {

    struct KernelArgs {
        size_t K = {};
        const float *Bpanel = {};
        size_t N = {};
        size_t B_stride = {};
        const float *cur_B_ptr = {};
    } ka;

    ka.K = (K/1) - 1;
    ka.Bpanel = Bpanel;
    ka.N = N;
    ka.B_stride = B_stride;

    __asm__ __volatile__(
      "1:"  // Height loop
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "ldr x25, [%x[args_ptr], %[offsetof_N]]\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "mov x24, %x[Apanel]\n"
      "2:"  // Width loop
      "ldr x23, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_B_stride]]\n"
      "add x22, x23, x20, LSL #2\n"
      "add x21, x22, x20, LSL #2\n"
      "add x20, x21, x20, LSL #2\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "cmp x25, #0x8\n"
      "mov %x[Apanel], x24\n"
      "bgt 3f\n"
      "cmp x25, #0x4\n"
      "mov x21, x23\n"
      "bgt 3f\n"
      "mov x22, x23\n"
      "3:"  // B setup done
      "ldr q0, [%x[Apanel], #0x0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v8.16b, #0x0\n"
      "ldr q4, [x23, #0x0]\n"
      "ldr q5, [x22, #0x0]\n"
      "movi v9.16b, #0x0\n"
      "ldr q6, [x21, #0x0]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "cmp x20, #0x4\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "movi v14.16b, #0x0\n"
      "movi v15.16b, #0x0\n"
      "movi v16.16b, #0x0\n"
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
      "blt 5f\n"
      "4:"  // main loop head
      "ldr q3, [%x[Apanel], #0x20]\n"
      "ldr q7, [%x[Apanel], #0x30]\n"
      "fmla v8.4s, v4.4s, v0.s[0]\n"
      "ldr q2, [x23, #0x10]\n"
      "fmla v11.4s, v4.4s, v0.s[1]\n"
      "fmla v14.4s, v4.4s, v0.s[2]\n"
      "fmla v17.4s, v4.4s, v0.s[3]\n"
      "fmla v20.4s, v4.4s, v1.s[0]\n"
      "sub x20, x20, #0x4\n"
      "fmla v23.4s, v4.4s, v1.s[1]\n"
      "fmla v26.4s, v4.4s, v1.s[2]\n"
      "cmp x20, #0x4\n"
      "fmla v29.4s, v4.4s, v1.s[3]\n"
      "ldr q4, [x22, #0x10]\n"
      "fmla v9.4s, v5.4s, v0.s[0]\n"
      "fmla v12.4s, v5.4s, v0.s[1]\n"
      "fmla v15.4s, v5.4s, v0.s[2]\n"
      "fmla v18.4s, v5.4s, v0.s[3]\n"
      "fmla v21.4s, v5.4s, v1.s[0]\n"
      "fmla v24.4s, v5.4s, v1.s[1]\n"
      "fmla v27.4s, v5.4s, v1.s[2]\n"
      "fmla v30.4s, v5.4s, v1.s[3]\n"
      "ldr q5, [x21, #0x10]\n"
      "fmla v10.4s, v6.4s, v0.s[0]\n"
      "fmla v13.4s, v6.4s, v0.s[1]\n"
      "fmla v16.4s, v6.4s, v0.s[2]\n"
      "fmla v19.4s, v6.4s, v0.s[3]\n"
      "ldr q0, [%x[Apanel], #0x40]\n"
      "fmla v22.4s, v6.4s, v1.s[0]\n"
      "fmla v25.4s, v6.4s, v1.s[1]\n"
      "fmla v28.4s, v6.4s, v1.s[2]\n"
      "fmla v31.4s, v6.4s, v1.s[3]\n"
      "ldr q1, [%x[Apanel], #0x50]\n"
      "ldr q6, [x23, #0x20]\n"
      "fmla v8.4s, v2.4s, v3.s[0]\n"
      "fmla v11.4s, v2.4s, v3.s[1]\n"
      "fmla v14.4s, v2.4s, v3.s[2]\n"
      "fmla v17.4s, v2.4s, v3.s[3]\n"
      "fmla v20.4s, v2.4s, v7.s[0]\n"
      "fmla v23.4s, v2.4s, v7.s[1]\n"
      "fmla v26.4s, v2.4s, v7.s[2]\n"
      "fmla v29.4s, v2.4s, v7.s[3]\n"
      "ldr q2, [x22, #0x20]\n"
      "fmla v9.4s, v4.4s, v3.s[0]\n"
      "fmla v12.4s, v4.4s, v3.s[1]\n"
      "fmla v15.4s, v4.4s, v3.s[2]\n"
      "fmla v18.4s, v4.4s, v3.s[3]\n"
      "fmla v21.4s, v4.4s, v7.s[0]\n"
      "fmla v24.4s, v4.4s, v7.s[1]\n"
      "fmla v27.4s, v4.4s, v7.s[2]\n"
      "fmla v30.4s, v4.4s, v7.s[3]\n"
      "ldr q4, [x21, #0x20]\n"
      "fmla v10.4s, v5.4s, v3.s[0]\n"
      "fmla v13.4s, v5.4s, v3.s[1]\n"
      "fmla v16.4s, v5.4s, v3.s[2]\n"
      "fmla v19.4s, v5.4s, v3.s[3]\n"
      "ldr q3, [%x[Apanel], #0x60]\n"
      "fmla v22.4s, v5.4s, v7.s[0]\n"
      "fmla v25.4s, v5.4s, v7.s[1]\n"
      "fmla v28.4s, v5.4s, v7.s[2]\n"
      "fmla v31.4s, v5.4s, v7.s[3]\n"
      "ldr q7, [%x[Apanel], #0x70]\n"
      "ldr q5, [x23, #0x30]\n"
      "fmla v8.4s, v6.4s, v0.s[0]\n"
      "fmla v11.4s, v6.4s, v0.s[1]\n"
      "fmla v14.4s, v6.4s, v0.s[2]\n"
      "fmla v17.4s, v6.4s, v0.s[3]\n"
      "add %x[Apanel], %x[Apanel], #0x80\n"
      "fmla v20.4s, v6.4s, v1.s[0]\n"
      "fmla v23.4s, v6.4s, v1.s[1]\n"
      "add x23, x23, #0x40\n"
      "fmla v26.4s, v6.4s, v1.s[2]\n"
      "fmla v29.4s, v6.4s, v1.s[3]\n"
      "ldr q6, [x22, #0x30]\n"
      "fmla v9.4s, v2.4s, v0.s[0]\n"
      "fmla v12.4s, v2.4s, v0.s[1]\n"
      "add x22, x22, #0x40\n"
      "fmla v15.4s, v2.4s, v0.s[2]\n"
      "fmla v18.4s, v2.4s, v0.s[3]\n"
      "fmla v21.4s, v2.4s, v1.s[0]\n"
      "fmla v24.4s, v2.4s, v1.s[1]\n"
      "fmla v27.4s, v2.4s, v1.s[2]\n"
      "fmla v30.4s, v2.4s, v1.s[3]\n"
      "ldr q2, [x21, #0x30]\n"
      "fmla v10.4s, v4.4s, v0.s[0]\n"
      "fmla v13.4s, v4.4s, v0.s[1]\n"
      "add x21, x21, #0x40\n"
      "fmla v16.4s, v4.4s, v0.s[2]\n"
      "fmla v19.4s, v4.4s, v0.s[3]\n"
      "ldr q0, [%x[Apanel], #0x0]\n"
      "fmla v22.4s, v4.4s, v1.s[0]\n"
      "fmla v25.4s, v4.4s, v1.s[1]\n"
      "fmla v28.4s, v4.4s, v1.s[2]\n"
      "fmla v31.4s, v4.4s, v1.s[3]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "ldr q4, [x23, #0x0]\n"
      "fmla v8.4s, v5.4s, v3.s[0]\n"
      "fmla v11.4s, v5.4s, v3.s[1]\n"
      "fmla v14.4s, v5.4s, v3.s[2]\n"
      "fmla v17.4s, v5.4s, v3.s[3]\n"
      "fmla v20.4s, v5.4s, v7.s[0]\n"
      "fmla v23.4s, v5.4s, v7.s[1]\n"
      "fmla v26.4s, v5.4s, v7.s[2]\n"
      "fmla v29.4s, v5.4s, v7.s[3]\n"
      "ldr q5, [x22, #0x0]\n"
      "fmla v9.4s, v6.4s, v3.s[0]\n"
      "fmla v12.4s, v6.4s, v3.s[1]\n"
      "fmla v15.4s, v6.4s, v3.s[2]\n"
      "fmla v18.4s, v6.4s, v3.s[3]\n"
      "fmla v21.4s, v6.4s, v7.s[0]\n"
      "fmla v24.4s, v6.4s, v7.s[1]\n"
      "fmla v27.4s, v6.4s, v7.s[2]\n"
      "fmla v30.4s, v6.4s, v7.s[3]\n"
      "ldr q6, [x21, #0x0]\n"
      "fmla v10.4s, v2.4s, v3.s[0]\n"
      "fmla v13.4s, v2.4s, v3.s[1]\n"
      "fmla v16.4s, v2.4s, v3.s[2]\n"
      "fmla v19.4s, v2.4s, v3.s[3]\n"
      "fmla v22.4s, v2.4s, v7.s[0]\n"
      "fmla v25.4s, v2.4s, v7.s[1]\n"
      "fmla v28.4s, v2.4s, v7.s[2]\n"
      "fmla v31.4s, v2.4s, v7.s[3]\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "fmla v8.4s, v4.4s, v0.s[0]\n"
      "fmla v11.4s, v4.4s, v0.s[1]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "fmla v14.4s, v4.4s, v0.s[2]\n"
      "fmla v17.4s, v4.4s, v0.s[3]\n"
      "add x23, x23, #0x10\n"
      "fmla v20.4s, v4.4s, v1.s[0]\n"
      "fmla v23.4s, v4.4s, v1.s[1]\n"
      "add x22, x22, #0x10\n"
      "fmla v26.4s, v4.4s, v1.s[2]\n"
      "fmla v29.4s, v4.4s, v1.s[3]\n"
      "add x21, x21, #0x10\n"
      "fmla v9.4s, v5.4s, v0.s[0]\n"
      "fmla v12.4s, v5.4s, v0.s[1]\n"
      "fmla v15.4s, v5.4s, v0.s[2]\n"
      "fmla v18.4s, v5.4s, v0.s[3]\n"
      "fmla v21.4s, v5.4s, v1.s[0]\n"
      "fmla v24.4s, v5.4s, v1.s[1]\n"
      "fmla v27.4s, v5.4s, v1.s[2]\n"
      "fmla v30.4s, v5.4s, v1.s[3]\n"
      "fmla v10.4s, v6.4s, v0.s[0]\n"
      "fmla v13.4s, v6.4s, v0.s[1]\n"
      "fmla v16.4s, v6.4s, v0.s[2]\n"
      "fmla v19.4s, v6.4s, v0.s[3]\n"
      "fmla v22.4s, v6.4s, v1.s[0]\n"
      "fmla v25.4s, v6.4s, v1.s[1]\n"
      "fmla v28.4s, v6.4s, v1.s[2]\n"
      "fmla v31.4s, v6.4s, v1.s[3]\n"
      "cbz x20, 7f\n"
      "6:"  // odd loop
      "ldr q4, [%x[Apanel], #0x0]\n"
      "ldr q3, [%x[Apanel], #0x10]\n"
      "subs x20, x20, #0x1\n"
      "ldr q2, [x23, #0x0]\n"
      "ldr q1, [x22, #0x0]\n"
      "fmla v8.4s, v2.4s, v4.s[0]\n"
      "ldr q0, [x21, #0x0]\n"
      "fmla v11.4s, v2.4s, v4.s[1]\n"
      "fmla v14.4s, v2.4s, v4.s[2]\n"
      "fmla v17.4s, v2.4s, v4.s[3]\n"
      "fmla v20.4s, v2.4s, v3.s[0]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "fmla v23.4s, v2.4s, v3.s[1]\n"
      "fmla v26.4s, v2.4s, v3.s[2]\n"
      "add x23, x23, #0x10\n"
      "fmla v29.4s, v2.4s, v3.s[3]\n"
      "fmla v9.4s, v1.4s, v4.s[0]\n"
      "add x22, x22, #0x10\n"
      "fmla v12.4s, v1.4s, v4.s[1]\n"
      "fmla v15.4s, v1.4s, v4.s[2]\n"
      "add x21, x21, #0x10\n"
      "fmla v18.4s, v1.4s, v4.s[3]\n"
      "fmla v21.4s, v1.4s, v3.s[0]\n"
      "fmla v24.4s, v1.4s, v3.s[1]\n"
      "fmla v27.4s, v1.4s, v3.s[2]\n"
      "fmla v30.4s, v1.4s, v3.s[3]\n"
      "fmla v10.4s, v0.4s, v4.s[0]\n"
      "fmla v13.4s, v0.4s, v4.s[1]\n"
      "fmla v16.4s, v0.4s, v4.s[2]\n"
      "fmla v19.4s, v0.4s, v4.s[3]\n"
      "fmla v22.4s, v0.4s, v3.s[0]\n"
      "fmla v25.4s, v0.4s, v3.s[1]\n"
      "fmla v28.4s, v0.4s, v3.s[2]\n"
      "fmla v31.4s, v0.4s, v3.s[3]\n"
      "bne 6b\n"
      "7:"  // multiply loop done
      "subs x25, x25, #0xc\n"
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
      : [args_ptr] "r" (&ka), [offsetof_B_stride] "I" (offsetof(KernelArgs, B_stride)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_cur_B_ptr] "I" (offsetof(KernelArgs, cur_B_ptr))
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25"
    );
}

} // namespace arm_gemm
#endif // __aarch64__
