//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 2;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(__fp16);
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(size_t k) {
    return kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() *
        (sizeof(__fp16) + kai_roundup(k, kai_kr) * sizeof(__fp16));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(k);
}

size_t kai_get_dst_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa() == 0);

    return m_idx * dst_stride + n_idx * sizeof(__fp16);
}

size_t kai_get_dst_size_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(__fp16);
}

void kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, __fp16 clamp_min, __fp16 clamp_max) {
    KAI_ASSUME(dst_stride_col == sizeof(__fp16));

    typedef struct {
        const void* A;
        const void* B;

        void* C;
        uint64_t ldcb;
        uint64_t M, N, K;
        __fp16 min;
        __fp16 max;

        void* accumulator_buffer;
        uint64_t flags;
    } KernelArgs;

    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;

    args.C = dst;
    args.ldcb = dst_stride_row;
    args.M = m;
    args.N = n;
    args.K = k;
    args.min = clamp_min;
    args.max = clamp_max;

    args.accumulator_buffer = NULL;
    args.flags = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "ldr w13, [%x[args], %[offsetof_M]]\n"
        "mov x11, #0x0\n"
        "mov x10, #0x0\n"
        "ptrue p1.b\n"
        ".inst 0x25207810  // ptrue pn8.b\n"
        "ldr w9, [%x[args], %[offsetof_N]]\n"
        "ldr x28, [%x[args], %[offsetof_A]]\n"
        "1:"  // M loop
        "ldr x27, [%x[args], %[offsetof_B]]\n"
        "2:"  // N loop
        "fmov z24.h, #0.0\n"
        "ld1h { z5.h }, p1/Z, [x27]\n"
        "fmov z27.h, #1.0\n"
        "mov x26, x28\n"
        ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
        "inch x27, ALL, MUL #2\n"
        "zip1 z30.h, z5.h, z24.h\n"
        "zip2 z20.h, z5.h, z24.h\n"
        ".inst 0x81be2760  // fmopa za0.s, p1/M, p1/M, z27.h, z30.h\n"
        ".inst 0x81b42761  // fmopa za1.s, p1/M, p1/M, z27.h, z20.h\n"
        ".inst 0x81be2762  // fmopa za2.s, p1/M, p1/M, z27.h, z30.h\n"
        ".inst 0x81b42763  // fmopa za3.s, p1/M, p1/M, z27.h, z20.h\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "add x20, x20, #0x1\n"
        "lsr x20, x20, #0x1\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 6f\n"
        "subs x21, x21, #0x1\n"
        ".inst 0xa0402352  // ld1h { z18.h-z19.h }, pn8.b/Z, [x26]\n"
        ".inst 0xa0402370  // ld1h { z16.h-z17.h }, pn8.b/Z, [x27]\n"
        ".inst 0xa1412342  // ld1h { z2.h, z10.h }, pn8.b/Z, [x26, #0x2, MUL VL]\n"
        ".inst 0xa041237e  // ld1h { z30.h-z31.h }, pn8.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0xa042235c  // ld1h { z28.h-z29.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
        ".inst 0xa1422366  // ld1h { z6.h, z14.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa1432345  // ld1h { z5.h, z13.h }, pn8.b/Z, [x26, #0x6, MUL VL]\n"
        "addvl x26, x26, #8\n"
        ".inst 0xa1432367  // ld1h { z7.h, z15.h }, pn8.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        "ble 5f\n"
        "4:"  // K loop
        ".inst 0x81b02640  // fmopa za0.s, p1/M, p1/M, z18.h, z16.h\n"
        "subs x21, x21, #0x1\n"
        ".inst 0x81b12641  // fmopa za1.s, p1/M, p1/M, z18.h, z17.h\n"
        ".inst 0x81b02662  // fmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
        ".inst 0x81b12663  // fmopa za3.s, p1/M, p1/M, z19.h, z17.h\n"
        ".inst 0xa0402352  // ld1h { z18.h-z19.h }, pn8.b/Z, [x26]\n"
        ".inst 0x81be2440  // fmopa za0.s, p1/M, p1/M, z2.h, z30.h\n"
        ".inst 0xa0402370  // ld1h { z16.h-z17.h }, pn8.b/Z, [x27]\n"
        ".inst 0x81bf2441  // fmopa za1.s, p1/M, p1/M, z2.h, z31.h\n"
        ".inst 0x81be2542  // fmopa za2.s, p1/M, p1/M, z10.h, z30.h\n"
        ".inst 0x81bf2543  // fmopa za3.s, p1/M, p1/M, z10.h, z31.h\n"
        ".inst 0xa1412342  // ld1h { z2.h, z10.h }, pn8.b/Z, [x26, #0x2, MUL VL]\n"
        ".inst 0x81a62780  // fmopa za0.s, p1/M, p1/M, z28.h, z6.h\n"
        ".inst 0xa041237e  // ld1h { z30.h-z31.h }, pn8.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0x81ae2781  // fmopa za1.s, p1/M, p1/M, z28.h, z14.h\n"
        ".inst 0x81a627a2  // fmopa za2.s, p1/M, p1/M, z29.h, z6.h\n"
        ".inst 0x81ae27a3  // fmopa za3.s, p1/M, p1/M, z29.h, z14.h\n"
        ".inst 0xa042235c  // ld1h { z28.h-z29.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
        ".inst 0xa1422366  // ld1h { z6.h, z14.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0x81a724a0  // fmopa za0.s, p1/M, p1/M, z5.h, z7.h\n"
        ".inst 0x81af24a1  // fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        ".inst 0x81a725a2  // fmopa za2.s, p1/M, p1/M, z13.h, z7.h\n"
        ".inst 0x81af25a3  // fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        ".inst 0xa1432345  // ld1h { z5.h, z13.h }, pn8.b/Z, [x26, #0x6, MUL VL]\n"
        "addvl x26, x26, #8\n"
        ".inst 0xa1432367  // ld1h { z7.h, z15.h }, pn8.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        "bgt 4b\n"
        "5:"  // K loop tail
        ".inst 0x81b02640  // fmopa za0.s, p1/M, p1/M, z18.h, z16.h\n"
        ".inst 0x81b12641  // fmopa za1.s, p1/M, p1/M, z18.h, z17.h\n"
        ".inst 0x81b02662  // fmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
        ".inst 0x81b12663  // fmopa za3.s, p1/M, p1/M, z19.h, z17.h\n"
        ".inst 0x81be2440  // fmopa za0.s, p1/M, p1/M, z2.h, z30.h\n"
        ".inst 0x81bf2441  // fmopa za1.s, p1/M, p1/M, z2.h, z31.h\n"
        ".inst 0x81be2542  // fmopa za2.s, p1/M, p1/M, z10.h, z30.h\n"
        ".inst 0x81bf2543  // fmopa za3.s, p1/M, p1/M, z10.h, z31.h\n"
        ".inst 0x81a62780  // fmopa za0.s, p1/M, p1/M, z28.h, z6.h\n"
        ".inst 0x81ae2781  // fmopa za1.s, p1/M, p1/M, z28.h, z14.h\n"
        ".inst 0x81a627a2  // fmopa za2.s, p1/M, p1/M, z29.h, z6.h\n"
        ".inst 0x81ae27a3  // fmopa za3.s, p1/M, p1/M, z29.h, z14.h\n"
        ".inst 0x81a724a0  // fmopa za0.s, p1/M, p1/M, z5.h, z7.h\n"
        ".inst 0x81af24a1  // fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        ".inst 0x81a725a2  // fmopa za2.s, p1/M, p1/M, z13.h, z7.h\n"
        ".inst 0x81af25a3  // fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        "6:"  // K oddments
        "cbz x20, 8f\n"
        "7:"  // K oddments: Loop
        ".inst 0xa1402345  // ld1h { z5.h, z13.h }, pn8.b/Z, [x26]\n"
        "subs x20, x20, #0x1\n"
        "addvl x26, x26, #2\n"
        ".inst 0xa040236e  // ld1h { z14.h-z15.h }, pn8.b/Z, [x27]\n"
        "addvl x27, x27, #2\n"
        ".inst 0x81ae24a0  // fmopa za0.s, p1/M, p1/M, z5.h, z14.h\n"
        ".inst 0x81af24a1  // fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        ".inst 0x81ae25a2  // fmopa za2.s, p1/M, p1/M, z13.h, z14.h\n"
        ".inst 0x81af25a3  // fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        "bgt 7b\n"
        "8:"  // K oddments: End
        "ldr x25, [%x[args], %[offsetof_C]]\n"
        "sub x24, x13, x11\n"
        "cntw x23, ALL, MUL #2\n"
        "ld1rh { z17.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
        "whilelt p0.h, x10, x9\n"
        "cmp x24, x23\n"
        "ld1rh { z16.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "mov x12, #0x0\n"
        "mov x21, #0x0\n"
        "add x25, x25, x10, LSL #1\n"  // C += n
        "mov x20, #0x2\n"
        "madd x25, x11, x22, x25\n"  // C += m * ldc
        "csel x24, x24, x23, LT\n"
        "10:"  // Store to output array: Accumulator loop
        ".inst 0xc006000e  // mova { z14.b-z15.b }, za0h.b[x12, 0:1]\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x23, LSL #1\n"
        "add x21, x21, #0x1\n"
        ".inst 0xc120e1cc  // fcvt z12.h, { z14.s-z15.s }\n"
        "csel x12, x12, x20, LT\n"
        "cmp x21, x24\n"
        ".inst 0x6470262c  // fclamp z12.h, z17.h, z16.h\n"
        "st1h { z12.h }, p0, [x25]\n"
        "add x25, x25, x22\n"
        "blt 10b\n"
        "incw x10, ALL, MUL #2\n"
        "cmp x10, x9\n"
        "blt 2b\n"
        "incw x11, ALL, MUL #2\n"
        "mov x10, #0x0\n"
        "cmp x11, x13\n"
        "mov x28, x26\n"
        "blt 1b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)), [offsetof_M] "I"(offsetof(KernelArgs, M)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)), [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9",
          "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22",
          "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
