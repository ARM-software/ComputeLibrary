//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16, FEAT_FP16.
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 8;
static const size_t kai_nr = 12;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_mr;
}

size_t kai_get_n_step_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_nr;
}

size_t kai_get_mr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_mr == 0);

    return m_idx * kai_roundup(k, kai_kr) * sizeof(uint16_t);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx * (sizeof(uint16_t) + kai_roundup(k, kai_kr) * sizeof(uint16_t));
}

size_t kai_get_dst_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
    size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    KAI_ASSUME(n_idx % kai_nr == 0);

    return m_idx * stride + n_idx * sizeof(uint16_t);
}

size_t kai_get_dst_size_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(size_t m, size_t n) {
    return m * n * sizeof(uint16_t);
}

void kai_run_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs_packed,                                   //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max) {
    KAI_ASSERT(dst_stride_col == sizeof(uint16_t));

    const void* Apanel = lhs_packed;
    void* Cpanel = dst;
    size_t ldc = dst_stride_row / sizeof(uint16_t);

    size_t M = m;

    typedef struct {
        float maxval;
        float minval;
        size_t N;
        size_t K;
        const void* Bpanel;
        void* output_ptr;
    } KernelArgs;

    KernelArgs ka;

    ka.N = n;
    ka.K = kai_roundup(k, kai_kr) / kai_kr - 1;

    ka.Bpanel = rhs_packed;

    // Direct output.
    ka.output_ptr = dst;

    // Clamping output.
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    __asm__ __volatile__(
        "1:"  // Height loop
        "add x11, %x[Cpanel], %x[ldc], LSL #2\n"
        "add x10, %x[Cpanel], %x[ldc], LSL #1\n"
        "add x9, x11, %x[ldc], LSL #1\n"
        "cmp %x[M], #0x8\n"
        "add x28, %x[Cpanel], %x[ldc], LSL #3\n"
        "add x27, %x[Cpanel], %x[ldc]\n"
        "add x26, x10, %x[ldc]\n"
        "add x25, x11, %x[ldc]\n"
        "add x24, x9, %x[ldc]\n"
        "bge 2f\n"
        "cmp %x[M], #0x2\n"
        "mov x24, %x[Cpanel]\n"
        "csel x27, x27, %x[Cpanel], GE\n"
        "csel x10, x10, %x[Cpanel], GT\n"
        "cmp %x[M], #0x4\n"
        "csel x26, x26, %x[Cpanel], GE\n"
        "csel x11, x11, %x[Cpanel], GT\n"
        "cmp %x[M], #0x6\n"
        "csel x25, x25, %x[Cpanel], GE\n"
        "csel x9, x9, %x[Cpanel], GT\n"
        "2:"  // all rows valid
        "ldr x23, [%x[args_ptr], %[offsetof_N]]\n"
        "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
        "mov x21, %x[Apanel]\n"
        "3:"  // Width loop
        "ldr q4, [x22, #0x0]\n"
        "ldr d5, [x22, #0x10]\n"
        "mov %x[Apanel], x21\n"
        "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
        "add x22, x22, #0x18\n"
        "ldr q7, [x22, #0x0]\n"
        "ldr q0, [%x[Apanel], #0x0]\n"
        "ldr q1, [%x[Apanel], #0x10]\n"
        "fcvtl v6.4s, v5.4h\n"
        "ldr q2, [%x[Apanel], #0x20]\n"
        "fcvtl2 v5.4s, v4.8h\n"
        "fcvtl v4.4s, v4.4h\n"
        "cmp x20, #0x2\n"
        "prfm pldl1keep, [%x[Apanel], #0x0]\n"
        "prfm pldl1keep, [x22, #0x0]\n"
        "zip1 v10.2d, v6.2d, v6.2d\n"
        "zip2 v13.2d, v6.2d, v6.2d\n"
        "prfm pldl1keep, [x22, #0x40]\n"
        "zip1 v8.2d, v4.2d, v4.2d\n"
        "zip2 v11.2d, v4.2d, v4.2d\n"
        "ldr q4, [x22, #0x10]\n"
        "zip1 v9.2d, v5.2d, v5.2d\n"
        "zip2 v12.2d, v5.2d, v5.2d\n"
        "prfm pldl1keep, [%x[Apanel], #0x40]\n"
        "mov v16.16b, v10.16b\n"
        "mov v19.16b, v13.16b\n"
        "prfm pldl1keep, [x22, #0x80]\n"
        "mov v14.16b, v8.16b\n"
        "mov v17.16b, v11.16b\n"
        "prfm pldl1keep, [%x[Apanel], #0x80]\n"
        "mov v15.16b, v9.16b\n"
        "mov v18.16b, v12.16b\n"
        "prfm pldl1keep, [x22, #0xc0]\n"
        "mov v20.16b, v8.16b\n"
        "mov v21.16b, v9.16b\n"
        "prfm pldl1keep, [x22, #0x100]\n"
        "mov v22.16b, v10.16b\n"
        "mov v23.16b, v11.16b\n"
        "prfm pldl1keep, [%x[Apanel], #0xc0]\n"
        "mov v24.16b, v12.16b\n"
        "mov v25.16b, v13.16b\n"
        "prfm pldl1keep, [x22, #0x140]\n"
        "mov v26.16b, v8.16b\n"
        "mov v27.16b, v9.16b\n"
        "add x22, x22, #0x20\n"
        "mov v28.16b, v10.16b\n"
        "mov v29.16b, v11.16b\n"
        "add %x[Apanel], %x[Apanel], #0x30\n"
        "mov v30.16b, v12.16b\n"
        "mov v31.16b, v13.16b\n"
        "blt 5f\n"
        "4:"  // main loop head
        "ldr q3, [%x[Apanel], #0x0]\n"
        "ldr q5, [x22, #0x0]\n"
        ".inst 0x6e47ec08  // bfmmla v8.4s, v0.8h, v7.8h\n"
        "ldr q6, [x22, #0x10]\n"
        ".inst 0x6e44ec0b  // bfmmla v11.4s, v0.8h, v4.8h\n"
        ".inst 0x6e47ec2e  // bfmmla v14.4s, v1.8h, v7.8h\n"
        ".inst 0x6e44ec31  // bfmmla v17.4s, v1.8h, v4.8h\n"
        ".inst 0x6e47ec54  // bfmmla v20.4s, v2.8h, v7.8h\n"
        "sub x20, x20, #0x2\n"
        ".inst 0x6e44ec57  // bfmmla v23.4s, v2.8h, v4.8h\n"
        ".inst 0x6e47ec7a  // bfmmla v26.4s, v3.8h, v7.8h\n"
        "ldr q7, [x22, #0x20]\n"
        ".inst 0x6e44ec7d  // bfmmla v29.4s, v3.8h, v4.8h\n"
        "ldr q4, [x22, #0x30]\n"
        ".inst 0x6e45ec09  // bfmmla v9.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0c  // bfmmla v12.4s, v0.8h, v6.8h\n"
        ".inst 0x6e45ec2f  // bfmmla v15.4s, v1.8h, v5.8h\n"
        "cmp x20, #0x2\n"
        ".inst 0x6e46ec32  // bfmmla v18.4s, v1.8h, v6.8h\n"
        ".inst 0x6e45ec55  // bfmmla v21.4s, v2.8h, v5.8h\n"
        "prfm pldl1keep, [%x[Apanel], #0x100]\n"
        ".inst 0x6e46ec58  // bfmmla v24.4s, v2.8h, v6.8h\n"
        ".inst 0x6e45ec7b  // bfmmla v27.4s, v3.8h, v5.8h\n"
        "ldr q5, [x22, #0x40]\n"
        ".inst 0x6e46ec7e  // bfmmla v30.4s, v3.8h, v6.8h\n"
        "ldr q6, [x22, #0x50]\n"
        ".inst 0x6e47ec0a  // bfmmla v10.4s, v0.8h, v7.8h\n"
        ".inst 0x6e44ec0d  // bfmmla v13.4s, v0.8h, v4.8h\n"
        "ldr q0, [%x[Apanel], #0x10]\n"
        ".inst 0x6e47ec30  // bfmmla v16.4s, v1.8h, v7.8h\n"
        ".inst 0x6e44ec33  // bfmmla v19.4s, v1.8h, v4.8h\n"
        "ldr q1, [%x[Apanel], #0x20]\n"
        ".inst 0x6e47ec56  // bfmmla v22.4s, v2.8h, v7.8h\n"
        ".inst 0x6e44ec59  // bfmmla v25.4s, v2.8h, v4.8h\n"
        "ldr q2, [%x[Apanel], #0x30]\n"
        ".inst 0x6e47ec7c  // bfmmla v28.4s, v3.8h, v7.8h\n"
        "ldr q7, [x22, #0x60]\n"
        ".inst 0x6e44ec7f  // bfmmla v31.4s, v3.8h, v4.8h\n"
        "ldr q3, [%x[Apanel], #0x40]\n"
        "ldr q4, [x22, #0x70]\n"
        ".inst 0x6e45ec08  // bfmmla v8.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0b  // bfmmla v11.4s, v0.8h, v6.8h\n"
        ".inst 0x6e45ec2e  // bfmmla v14.4s, v1.8h, v5.8h\n"
        ".inst 0x6e46ec31  // bfmmla v17.4s, v1.8h, v6.8h\n"
        "prfm pldl1keep, [x22, #0x180]\n"
        ".inst 0x6e45ec54  // bfmmla v20.4s, v2.8h, v5.8h\n"
        ".inst 0x6e46ec57  // bfmmla v23.4s, v2.8h, v6.8h\n"
        "prfm pldl1keep, [x22, #0x1c0]\n"
        ".inst 0x6e45ec7a  // bfmmla v26.4s, v3.8h, v5.8h\n"
        "ldr q5, [x22, #0x80]\n"
        ".inst 0x6e46ec7d  // bfmmla v29.4s, v3.8h, v6.8h\n"
        "ldr q6, [x22, #0x90]\n"
        "prfm pldl1keep, [%x[Apanel], #0x140]\n"
        ".inst 0x6e47ec09  // bfmmla v9.4s, v0.8h, v7.8h\n"
        "prfm pldl1keep, [x22, #0x200]\n"
        ".inst 0x6e44ec0c  // bfmmla v12.4s, v0.8h, v4.8h\n"
        ".inst 0x6e47ec2f  // bfmmla v15.4s, v1.8h, v7.8h\n"
        ".inst 0x6e44ec32  // bfmmla v18.4s, v1.8h, v4.8h\n"
        ".inst 0x6e47ec55  // bfmmla v21.4s, v2.8h, v7.8h\n"
        ".inst 0x6e44ec58  // bfmmla v24.4s, v2.8h, v4.8h\n"
        ".inst 0x6e47ec7b  // bfmmla v27.4s, v3.8h, v7.8h\n"
        "ldr q7, [x22, #0xa0]\n"
        ".inst 0x6e44ec7e  // bfmmla v30.4s, v3.8h, v4.8h\n"
        "ldr q4, [x22, #0xb0]\n"
        ".inst 0x6e45ec0a  // bfmmla v10.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0d  // bfmmla v13.4s, v0.8h, v6.8h\n"
        "ldr q0, [%x[Apanel], #0x50]\n"
        ".inst 0x6e45ec30  // bfmmla v16.4s, v1.8h, v5.8h\n"
        ".inst 0x6e46ec33  // bfmmla v19.4s, v1.8h, v6.8h\n"
        "ldr q1, [%x[Apanel], #0x60]\n"
        ".inst 0x6e45ec56  // bfmmla v22.4s, v2.8h, v5.8h\n"
        ".inst 0x6e46ec59  // bfmmla v25.4s, v2.8h, v6.8h\n"
        "ldr q2, [%x[Apanel], #0x70]\n"
        ".inst 0x6e45ec7c  // bfmmla v28.4s, v3.8h, v5.8h\n"
        ".inst 0x6e46ec7f  // bfmmla v31.4s, v3.8h, v6.8h\n"
        "add %x[Apanel], %x[Apanel], #0x80\n"
        "add x22, x22, #0xc0\n"
        "bge 4b\n"
        "5:"  // main loop skip
        "ldr q3, [%x[Apanel], #0x0]\n"
        "ldr q5, [x22, #0x0]\n"
        ".inst 0x6e47ec08  // bfmmla v8.4s, v0.8h, v7.8h\n"
        "ldr q6, [x22, #0x10]\n"
        ".inst 0x6e44ec0b  // bfmmla v11.4s, v0.8h, v4.8h\n"
        ".inst 0x6e47ec2e  // bfmmla v14.4s, v1.8h, v7.8h\n"
        ".inst 0x6e44ec31  // bfmmla v17.4s, v1.8h, v4.8h\n"
        ".inst 0x6e47ec54  // bfmmla v20.4s, v2.8h, v7.8h\n"
        "add %x[Apanel], %x[Apanel], #0x10\n"
        ".inst 0x6e44ec57  // bfmmla v23.4s, v2.8h, v4.8h\n"
        ".inst 0x6e47ec7a  // bfmmla v26.4s, v3.8h, v7.8h\n"
        "ldr q7, [x22, #0x20]\n"
        ".inst 0x6e44ec7d  // bfmmla v29.4s, v3.8h, v4.8h\n"
        "ldr q4, [x22, #0x30]\n"
        ".inst 0x6e45ec09  // bfmmla v9.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0c  // bfmmla v12.4s, v0.8h, v6.8h\n"
        ".inst 0x6e45ec2f  // bfmmla v15.4s, v1.8h, v5.8h\n"
        "add x22, x22, #0x40\n"
        ".inst 0x6e46ec32  // bfmmla v18.4s, v1.8h, v6.8h\n"
        ".inst 0x6e45ec55  // bfmmla v21.4s, v2.8h, v5.8h\n"
        ".inst 0x6e46ec58  // bfmmla v24.4s, v2.8h, v6.8h\n"
        ".inst 0x6e45ec7b  // bfmmla v27.4s, v3.8h, v5.8h\n"
        ".inst 0x6e46ec7e  // bfmmla v30.4s, v3.8h, v6.8h\n"
        ".inst 0x6e47ec0a  // bfmmla v10.4s, v0.8h, v7.8h\n"
        ".inst 0x6e44ec0d  // bfmmla v13.4s, v0.8h, v4.8h\n"
        ".inst 0x6e47ec30  // bfmmla v16.4s, v1.8h, v7.8h\n"
        ".inst 0x6e44ec33  // bfmmla v19.4s, v1.8h, v4.8h\n"
        ".inst 0x6e47ec56  // bfmmla v22.4s, v2.8h, v7.8h\n"
        ".inst 0x6e44ec59  // bfmmla v25.4s, v2.8h, v4.8h\n"
        ".inst 0x6e47ec7c  // bfmmla v28.4s, v3.8h, v7.8h\n"
        ".inst 0x6e44ec7f  // bfmmla v31.4s, v3.8h, v4.8h\n"
        "cbz x20, 6f\n"
        "ldr q5, [x22, #0x0]\n"
        "ldr q0, [%x[Apanel], #0x0]\n"
        "ldr q1, [%x[Apanel], #0x10]\n"
        "ldr q6, [x22, #0x10]\n"
        "ldr q2, [%x[Apanel], #0x20]\n"
        "ldr q3, [%x[Apanel], #0x30]\n"
        "add %x[Apanel], %x[Apanel], #0x40\n"
        "ldr q7, [x22, #0x20]\n"
        "ldr q4, [x22, #0x30]\n"
        ".inst 0x6e45ec08  // bfmmla v8.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0b  // bfmmla v11.4s, v0.8h, v6.8h\n"
        ".inst 0x6e45ec2e  // bfmmla v14.4s, v1.8h, v5.8h\n"
        ".inst 0x6e46ec31  // bfmmla v17.4s, v1.8h, v6.8h\n"
        ".inst 0x6e45ec54  // bfmmla v20.4s, v2.8h, v5.8h\n"
        ".inst 0x6e46ec57  // bfmmla v23.4s, v2.8h, v6.8h\n"
        ".inst 0x6e45ec7a  // bfmmla v26.4s, v3.8h, v5.8h\n"
        "ldr q5, [x22, #0x40]\n"
        ".inst 0x6e46ec7d  // bfmmla v29.4s, v3.8h, v6.8h\n"
        "ldr q6, [x22, #0x50]\n"
        ".inst 0x6e47ec09  // bfmmla v9.4s, v0.8h, v7.8h\n"
        ".inst 0x6e44ec0c  // bfmmla v12.4s, v0.8h, v4.8h\n"
        ".inst 0x6e47ec2f  // bfmmla v15.4s, v1.8h, v7.8h\n"
        "add x22, x22, #0x60\n"
        ".inst 0x6e44ec32  // bfmmla v18.4s, v1.8h, v4.8h\n"
        ".inst 0x6e47ec55  // bfmmla v21.4s, v2.8h, v7.8h\n"
        ".inst 0x6e44ec58  // bfmmla v24.4s, v2.8h, v4.8h\n"
        ".inst 0x6e47ec7b  // bfmmla v27.4s, v3.8h, v7.8h\n"
        ".inst 0x6e44ec7e  // bfmmla v30.4s, v3.8h, v4.8h\n"
        ".inst 0x6e45ec0a  // bfmmla v10.4s, v0.8h, v5.8h\n"
        ".inst 0x6e46ec0d  // bfmmla v13.4s, v0.8h, v6.8h\n"
        ".inst 0x6e45ec30  // bfmmla v16.4s, v1.8h, v5.8h\n"
        ".inst 0x6e46ec33  // bfmmla v19.4s, v1.8h, v6.8h\n"
        ".inst 0x6e45ec56  // bfmmla v22.4s, v2.8h, v5.8h\n"
        ".inst 0x6e46ec59  // bfmmla v25.4s, v2.8h, v6.8h\n"
        ".inst 0x6e45ec7c  // bfmmla v28.4s, v3.8h, v5.8h\n"
        ".inst 0x6e46ec7f  // bfmmla v31.4s, v3.8h, v6.8h\n"
        "6:"  // multiply loop done
        "add x20, %x[args_ptr], %[offset_max]\n"
        "uzp1 v7.2d, v8.2d, v11.2d\n"
        "uzp2 v8.2d, v8.2d, v11.2d\n"
        "ld1r { v1.4s }, [x20]\n"
        "uzp1 v11.2d, v9.2d, v12.2d\n"
        "uzp2 v9.2d, v9.2d, v12.2d\n"
        "uzp1 v12.2d, v10.2d, v13.2d\n"
        "uzp2 v10.2d, v10.2d, v13.2d\n"
        "add x20, %x[args_ptr], %[offset_min]\n"
        "ld1r { v0.4s }, [x20]\n"
        "uzp1 v13.2d, v14.2d, v17.2d\n"
        "uzp2 v14.2d, v14.2d, v17.2d\n"
        "uzp1 v17.2d, v15.2d, v18.2d\n"
        "uzp2 v15.2d, v15.2d, v18.2d\n"
        "cmp x23, #0xc\n"
        "uzp1 v18.2d, v16.2d, v19.2d\n"
        "uzp2 v16.2d, v16.2d, v19.2d\n"
        "uzp1 v19.2d, v20.2d, v23.2d\n"
        "uzp2 v20.2d, v20.2d, v23.2d\n"
        "uzp1 v23.2d, v21.2d, v24.2d\n"
        "uzp2 v21.2d, v21.2d, v24.2d\n"
        "uzp1 v24.2d, v22.2d, v25.2d\n"
        "uzp2 v22.2d, v22.2d, v25.2d\n"
        "uzp1 v25.2d, v26.2d, v29.2d\n"
        "uzp2 v26.2d, v26.2d, v29.2d\n"
        "uzp1 v29.2d, v27.2d, v30.2d\n"
        "uzp2 v27.2d, v27.2d, v30.2d\n"
        "uzp1 v30.2d, v28.2d, v31.2d\n"
        "uzp2 v28.2d, v28.2d, v31.2d\n"
        "fmin v7.4s, v7.4s, v1.4s\n"
        "fmin v8.4s, v8.4s, v1.4s\n"
        "fmin v13.4s, v13.4s, v1.4s\n"
        "fmin v14.4s, v14.4s, v1.4s\n"
        "fmin v19.4s, v19.4s, v1.4s\n"
        "fmin v20.4s, v20.4s, v1.4s\n"
        "fmin v25.4s, v25.4s, v1.4s\n"
        "fmin v26.4s, v26.4s, v1.4s\n"
        "fmax v7.4s, v7.4s, v0.4s\n"
        "fmin v11.4s, v11.4s, v1.4s\n"
        "fmin v12.4s, v12.4s, v1.4s\n"
        "fmax v8.4s, v8.4s, v0.4s\n"
        "fmin v9.4s, v9.4s, v1.4s\n"
        "fmin v10.4s, v10.4s, v1.4s\n"
        "fmax v13.4s, v13.4s, v0.4s\n"
        "fmin v17.4s, v17.4s, v1.4s\n"
        "fmin v18.4s, v18.4s, v1.4s\n"
        "fmax v14.4s, v14.4s, v0.4s\n"
        "fmin v15.4s, v15.4s, v1.4s\n"
        "fmin v16.4s, v16.4s, v1.4s\n"
        "fmax v19.4s, v19.4s, v0.4s\n"
        "fmin v23.4s, v23.4s, v1.4s\n"
        "fmin v24.4s, v24.4s, v1.4s\n"
        "fmax v20.4s, v20.4s, v0.4s\n"
        "fmin v21.4s, v21.4s, v1.4s\n"
        "fmin v22.4s, v22.4s, v1.4s\n"
        "fmax v25.4s, v25.4s, v0.4s\n"
        "fmin v29.4s, v29.4s, v1.4s\n"
        "fmin v30.4s, v30.4s, v1.4s\n"
        "fmax v26.4s, v26.4s, v0.4s\n"
        "fmin v27.4s, v27.4s, v1.4s\n"
        "fmin v28.4s, v28.4s, v1.4s\n"
        "fmax v11.4s, v11.4s, v0.4s\n"
        "fmax v12.4s, v12.4s, v0.4s\n"
        "fmax v9.4s, v9.4s, v0.4s\n"
        "fmax v10.4s, v10.4s, v0.4s\n"
        "fmax v17.4s, v17.4s, v0.4s\n"
        "fmax v18.4s, v18.4s, v0.4s\n"
        "fmax v15.4s, v15.4s, v0.4s\n"
        "fmax v16.4s, v16.4s, v0.4s\n"
        "fmax v23.4s, v23.4s, v0.4s\n"
        "fmax v24.4s, v24.4s, v0.4s\n"
        "fmax v21.4s, v21.4s, v0.4s\n"
        "fmax v22.4s, v22.4s, v0.4s\n"
        "fmax v29.4s, v29.4s, v0.4s\n"
        "fmax v30.4s, v30.4s, v0.4s\n"
        "fmax v27.4s, v27.4s, v0.4s\n"
        "fmax v28.4s, v28.4s, v0.4s\n"
        "fcvtn v7.4h, v7.4s\n"
        "fcvtn v8.4h, v8.4s\n"
        "fcvtn v13.4h, v13.4s\n"
        "fcvtn v14.4h, v14.4s\n"
        "fcvtn v19.4h, v19.4s\n"
        "fcvtn v20.4h, v20.4s\n"
        "fcvtn v25.4h, v25.4s\n"
        "fcvtn v26.4h, v26.4s\n"
        "fcvtn2 v7.8h, v11.4s\n"
        "fcvtn v11.4h, v12.4s\n"
        "fcvtn2 v8.8h, v9.4s\n"
        "fcvtn v9.4h, v10.4s\n"
        "fcvtn2 v13.8h, v17.4s\n"
        "fcvtn v17.4h, v18.4s\n"
        "fcvtn2 v14.8h, v15.4s\n"
        "fcvtn v15.4h, v16.4s\n"
        "fcvtn2 v19.8h, v23.4s\n"
        "fcvtn v23.4h, v24.4s\n"
        "fcvtn2 v20.8h, v21.4s\n"
        "fcvtn v21.4h, v22.4s\n"
        "fcvtn2 v25.8h, v29.4s\n"
        "fcvtn v29.4h, v30.4s\n"
        "fcvtn2 v26.8h, v27.4s\n"
        "fcvtn v27.4h, v28.4s\n"
        "blt 7f\n"
        "str q26, [x24, #0x0]\n"
        "str d27, [x24, #0x10]\n"
        "add x24, x24, #0x18\n"
        "str q25, [x9, #0x0]\n"
        "str d29, [x9, #0x10]\n"
        "add x9, x9, #0x18\n"
        "str q20, [x25, #0x0]\n"
        "str d21, [x25, #0x10]\n"
        "add x25, x25, #0x18\n"
        "str q19, [x11, #0x0]\n"
        "str d23, [x11, #0x10]\n"
        "add x11, x11, #0x18\n"
        "str q14, [x26, #0x0]\n"
        "str d15, [x26, #0x10]\n"
        "add x26, x26, #0x18\n"
        "str q13, [x10, #0x0]\n"
        "str d17, [x10, #0x10]\n"
        "add x10, x10, #0x18\n"
        "str q8, [x27, #0x0]\n"
        "str d9, [x27, #0x10]\n"
        "add x27, x27, #0x18\n"
        "str q7, [%x[Cpanel], #0x0]\n"
        "str d11, [%x[Cpanel], #0x10]\n"
        "add %x[Cpanel], %x[Cpanel], #0x18\n"
        "b 14f\n"
        "7:"  // partial output
        "tbz x23, #3, 9f\n"
        "st1 { v26.8h }, [x24], #0x10\n"
        "st1 { v25.8h }, [x9], #0x10\n"
        "st1 { v20.8h }, [x25], #0x10\n"
        "st1 { v19.8h }, [x11], #0x10\n"
        "st1 { v14.8h }, [x26], #0x10\n"
        "st1 { v13.8h }, [x10], #0x10\n"
        "st1 { v8.8h }, [x27], #0x10\n"
        "st1 { v7.8h }, [%x[Cpanel]], #0x10\n"
        "tbz x23, #1, 8f\n"
        "str s27, [x24], #0x4\n"
        "str s29, [x9], #0x4\n"
        "str s21, [x25], #0x4\n"
        "str s23, [x11], #0x4\n"
        "str s15, [x26], #0x4\n"
        "str s17, [x10], #0x4\n"
        "str s9, [x27], #0x4\n"
        "str s11, [%x[Cpanel]], #0x4\n"
        "tbz x23, #0, 13f\n"
        "st1 { v27.h }[2], [x24]\n"
        "st1 { v29.h }[2], [x9]\n"
        "st1 { v21.h }[2], [x25]\n"
        "st1 { v23.h }[2], [x11]\n"
        "st1 { v15.h }[2], [x26]\n"
        "st1 { v17.h }[2], [x10]\n"
        "st1 { v9.h }[2], [x27]\n"
        "st1 { v11.h }[2], [%x[Cpanel]]\n"
        "b 13f\n"
        "8:"  // partial result store: partial_1_8
        "tbz x23, #0, 13f\n"
        "str h27, [x24, #0x0]\n"
        "str h29, [x9, #0x0]\n"
        "str h21, [x25, #0x0]\n"
        "str h23, [x11, #0x0]\n"
        "str h15, [x26, #0x0]\n"
        "str h17, [x10, #0x0]\n"
        "str h9, [x27, #0x0]\n"
        "str h11, [%x[Cpanel], #0x0]\n"
        "b 13f\n"
        "9:"  // partial result store: partial_4_0
        "tbz x23, #2, 11f\n"
        "str d26, [x24], #0x8\n"
        "str d25, [x9], #0x8\n"
        "str d20, [x25], #0x8\n"
        "str d19, [x11], #0x8\n"
        "str d14, [x26], #0x8\n"
        "str d13, [x10], #0x8\n"
        "str d8, [x27], #0x8\n"
        "str d7, [%x[Cpanel]], #0x8\n"
        "tbz x23, #1, 10f\n"
        "st1 { v26.s }[2], [x24], #0x4\n"
        "st1 { v25.s }[2], [x9], #0x4\n"
        "st1 { v20.s }[2], [x25], #0x4\n"
        "st1 { v19.s }[2], [x11], #0x4\n"
        "st1 { v14.s }[2], [x26], #0x4\n"
        "st1 { v13.s }[2], [x10], #0x4\n"
        "st1 { v8.s }[2], [x27], #0x4\n"
        "st1 { v7.s }[2], [%x[Cpanel]], #0x4\n"
        "tbz x23, #0, 13f\n"
        "st1 { v26.h }[6], [x24]\n"
        "st1 { v25.h }[6], [x9]\n"
        "st1 { v20.h }[6], [x25]\n"
        "st1 { v19.h }[6], [x11]\n"
        "st1 { v14.h }[6], [x26]\n"
        "st1 { v13.h }[6], [x10]\n"
        "st1 { v8.h }[6], [x27]\n"
        "st1 { v7.h }[6], [%x[Cpanel]]\n"
        "b 13f\n"
        "10:"  // partial result store: partial_1_4
        "tbz x23, #0, 13f\n"
        "st1 { v26.h }[4], [x24]\n"
        "st1 { v25.h }[4], [x9]\n"
        "st1 { v20.h }[4], [x25]\n"
        "st1 { v19.h }[4], [x11]\n"
        "st1 { v14.h }[4], [x26]\n"
        "st1 { v13.h }[4], [x10]\n"
        "st1 { v8.h }[4], [x27]\n"
        "st1 { v7.h }[4], [%x[Cpanel]]\n"
        "b 13f\n"
        "11:"  // partial result store: partial_2_0
        "tbz x23, #1, 12f\n"
        "str s26, [x24], #0x4\n"
        "str s25, [x9], #0x4\n"
        "str s20, [x25], #0x4\n"
        "str s19, [x11], #0x4\n"
        "str s14, [x26], #0x4\n"
        "str s13, [x10], #0x4\n"
        "str s8, [x27], #0x4\n"
        "str s7, [%x[Cpanel]], #0x4\n"
        "tbz x23, #0, 13f\n"
        "st1 { v26.h }[2], [x24]\n"
        "st1 { v25.h }[2], [x9]\n"
        "st1 { v20.h }[2], [x25]\n"
        "st1 { v19.h }[2], [x11]\n"
        "st1 { v14.h }[2], [x26]\n"
        "st1 { v13.h }[2], [x10]\n"
        "st1 { v8.h }[2], [x27]\n"
        "st1 { v7.h }[2], [%x[Cpanel]]\n"
        "b 13f\n"
        "12:"  // partial result store: partial_1_0
        "str h26, [x24, #0x0]\n"
        "str h25, [x9, #0x0]\n"
        "str h20, [x25, #0x0]\n"
        "str h19, [x11, #0x0]\n"
        "str h14, [x26, #0x0]\n"
        "str h13, [x10, #0x0]\n"
        "str h8, [x27, #0x0]\n"
        "str h7, [%x[Cpanel], #0x0]\n"
        "13:"  // partial result store: Done
        "14:"  // store done
        "subs x23, x23, #0xc\n"
        "bgt 3b\n"
        "subs %x[M], %x[M], #0x8\n"
        "mov %x[Cpanel], x28\n"
        "bgt 1b\n"
        : [Apanel] "+&r"(Apanel), [Cpanel] "+&r"(Cpanel), [M] "+&r"(M)
        : [args_ptr] "r"(&ka), [ldc] "r"(ldc * sizeof(uint16_t)), [offset_max] "I"(offsetof(KernelArgs, maxval)),
          [offset_min] "I"(offsetof(KernelArgs, minval)), [offsetof_Bpanel] "I"(offsetof(KernelArgs, Bpanel)),
          [offsetof_K] "I"(offsetof(KernelArgs, K)), [offsetof_N] "I"(offsetof(KernelArgs, N))
        : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v2", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7",
          "v8", "v9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9");
}

#endif  // Architectural features check.
