//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() == 0);
    return m_idx * k * sizeof(float);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() == 0);
    return n_idx * (k * sizeof(float) + sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() == 0);

    return m_idx * dst_stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    typedef struct {
        const void* A;
        const void* B;

        void* C;
        uint64_t ldcb;
        uint64_t M, N, K;
        float min;
        float max;

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
        "ldr x17, [%x[args], %[offsetof_flags]]\n"
        ".inst 0xd503477f  // SMSTART ZA\n"
        "ptrue p0.b\n"
        ".inst 0x25207811  // ptrue pn9.b\n"
        "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "tbz x17, #0, 2f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "1:"  // Initial accumulator load from buffer: Loop
        ".inst 0xa040c618  // ld1w { z24.s-z27.s }, pn9.b/Z, [x16]\n"
        ".inst 0xa041c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
        ".inst 0xa042c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
        ".inst 0xa043c610  // ld1w { z16.s-z19.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
        ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
        "addvl x16, x16, #16\n"
        ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
        ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
        ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x20\n"
        "blt 1b\n"
        "2:"  // Initial accumulator load from buffer: End
        "ldr w14, [%x[args], %[offsetof_M]]\n"
        "mov x13, #0x0\n"
        "mov x11, #0x0\n"
        "ldr w10, [%x[args], %[offsetof_N]]\n"
        "ldr x9, [%x[args], %[offsetof_A]]\n"
        "3:"  // M loop
        "ldr x28, [%x[args], %[offsetof_B]]\n"
        "4:"  // N loop
        "mov x27, x9\n"
        ".inst 0x25aa4570  // whilelt pn8.s, x11, x10, VLx2\n"
        "tbnz x17, #0, 5f\n"
        "fmov z17.s, #1.0\n"
        ".inst 0xa040438a  // ld1w { z10.s-z11.s }, p8/Z, [x28]\n"  // Load bias
        ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
        "addvl x28, x28, #2\n"
        ".inst 0x808a0220  // fmopa za0.s, p0/M, p0/M, z17.s, z10.s\n"
        ".inst 0x808b0221  // fmopa za1.s, p0/M, p0/M, z17.s, z11.s\n"
        ".inst 0x808a0222  // fmopa za2.s, p0/M, p0/M, z17.s, z10.s\n"
        ".inst 0x808b0223  // fmopa za3.s, p0/M, p0/M, z17.s, z11.s\n"
        "5:"  // Prepare accumulators: Test for last block
        "mov x20, x11\n"
        "mov x21, x13\n"
        "incw x20, ALL, MUL #2\n"
        "incw x21, ALL, MUL #2\n"
        "cmp x20, x10\n"
        "mov x20, x17\n"
        "csel x21, x13, x21, LT\n"
        "bfm x17, XZR, #0x0, #0x0  // bfc x17, #0x0, #0x1\n"
        "cmp x21, x14\n"
        "csel x17, x20, x17, LT\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 9f\n"
        "subs x21, x21, #0x1\n"
        ".inst 0xa0404776  // ld1w { z22.s-z23.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa1404787  // ld1w { z7.s, z15.s }, pn9.b/Z, [x28]\n"
        ".inst 0xa1414766  // ld1w { z6.s, z14.s }, pn9.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0xa0414794  // ld1w { z20.s-z21.s }, pn9.b/Z, [x28, #0x2, MUL VL]\n"
        ".inst 0xa1424762  // ld1w { z2.s, z10.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa1424783  // ld1w { z3.s, z11.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
        ".inst 0xa1434761  // ld1w { z1.s, z9.s }, pn9.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        ".inst 0xa0434784  // ld1w { z4.s-z5.s }, pn9.b/Z, [x28, #0x6, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "ble 8f\n"
        "7:"  // K loop
        ".inst 0x808702c0  // fmopa za0.s, p0/M, p0/M, z22.s, z7.s\n"
        "subs x21, x21, #0x1\n"
        ".inst 0x808f02c1  // fmopa za1.s, p0/M, p0/M, z22.s, z15.s\n"
        ".inst 0x808702e2  // fmopa za2.s, p0/M, p0/M, z23.s, z7.s\n"
        ".inst 0x808f02e3  // fmopa za3.s, p0/M, p0/M, z23.s, z15.s\n"
        ".inst 0xa0404776  // ld1w { z22.s-z23.s }, pn9.b/Z, [x27]\n"
        ".inst 0x809400c0  // fmopa za0.s, p0/M, p0/M, z6.s, z20.s\n"
        ".inst 0xa1404787  // ld1w { z7.s, z15.s }, pn9.b/Z, [x28]\n"
        ".inst 0x809500c1  // fmopa za1.s, p0/M, p0/M, z6.s, z21.s\n"
        ".inst 0x809401c2  // fmopa za2.s, p0/M, p0/M, z14.s, z20.s\n"
        ".inst 0x809501c3  // fmopa za3.s, p0/M, p0/M, z14.s, z21.s\n"
        ".inst 0xa1414766  // ld1w { z6.s, z14.s }, pn9.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0x80830040  // fmopa za0.s, p0/M, p0/M, z2.s, z3.s\n"
        ".inst 0xa0414794  // ld1w { z20.s-z21.s }, pn9.b/Z, [x28, #0x2, MUL VL]\n"
        ".inst 0x808b0041  // fmopa za1.s, p0/M, p0/M, z2.s, z11.s\n"
        ".inst 0x80830142  // fmopa za2.s, p0/M, p0/M, z10.s, z3.s\n"
        ".inst 0x808b0143  // fmopa za3.s, p0/M, p0/M, z10.s, z11.s\n"
        ".inst 0xa1424762  // ld1w { z2.s, z10.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa1424783  // ld1w { z3.s, z11.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
        ".inst 0x80840020  // fmopa za0.s, p0/M, p0/M, z1.s, z4.s\n"
        ".inst 0x80850021  // fmopa za1.s, p0/M, p0/M, z1.s, z5.s\n"
        ".inst 0x80840122  // fmopa za2.s, p0/M, p0/M, z9.s, z4.s\n"
        ".inst 0x80850123  // fmopa za3.s, p0/M, p0/M, z9.s, z5.s\n"
        ".inst 0xa1434761  // ld1w { z1.s, z9.s }, pn9.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        ".inst 0xa0434784  // ld1w { z4.s-z5.s }, pn9.b/Z, [x28, #0x6, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "bgt 7b\n"
        "8:"  // K loop tail
        ".inst 0x808702c0  // fmopa za0.s, p0/M, p0/M, z22.s, z7.s\n"
        ".inst 0x808f02c1  // fmopa za1.s, p0/M, p0/M, z22.s, z15.s\n"
        ".inst 0x808702e2  // fmopa za2.s, p0/M, p0/M, z23.s, z7.s\n"
        ".inst 0x808f02e3  // fmopa za3.s, p0/M, p0/M, z23.s, z15.s\n"
        ".inst 0x809400c0  // fmopa za0.s, p0/M, p0/M, z6.s, z20.s\n"
        ".inst 0x809500c1  // fmopa za1.s, p0/M, p0/M, z6.s, z21.s\n"
        ".inst 0x809401c2  // fmopa za2.s, p0/M, p0/M, z14.s, z20.s\n"
        ".inst 0x809501c3  // fmopa za3.s, p0/M, p0/M, z14.s, z21.s\n"
        ".inst 0x80830040  // fmopa za0.s, p0/M, p0/M, z2.s, z3.s\n"
        ".inst 0x808b0041  // fmopa za1.s, p0/M, p0/M, z2.s, z11.s\n"
        ".inst 0x80830142  // fmopa za2.s, p0/M, p0/M, z10.s, z3.s\n"
        ".inst 0x808b0143  // fmopa za3.s, p0/M, p0/M, z10.s, z11.s\n"
        ".inst 0x80840020  // fmopa za0.s, p0/M, p0/M, z1.s, z4.s\n"
        ".inst 0x80850021  // fmopa za1.s, p0/M, p0/M, z1.s, z5.s\n"
        ".inst 0x80840122  // fmopa za2.s, p0/M, p0/M, z9.s, z4.s\n"
        ".inst 0x80850123  // fmopa za3.s, p0/M, p0/M, z9.s, z5.s\n"
        "9:"  // K oddments
        "cbz x20, 11f\n"
        "10:"  // K oddments: Loop
        ".inst 0xa040476a  // ld1w { z10.s-z11.s }, pn9.b/Z, [x27]\n"
        "subs x20, x20, #0x1\n"
        "addvl x27, x27, #2\n"
        ".inst 0xa040478e  // ld1w { z14.s-z15.s }, pn9.b/Z, [x28]\n"
        "addvl x28, x28, #2\n"
        ".inst 0x808e0140  // fmopa za0.s, p0/M, p0/M, z10.s, z14.s\n"
        ".inst 0x808f0141  // fmopa za1.s, p0/M, p0/M, z10.s, z15.s\n"
        ".inst 0x808e0162  // fmopa za2.s, p0/M, p0/M, z11.s, z14.s\n"
        ".inst 0x808f0163  // fmopa za3.s, p0/M, p0/M, z11.s, z15.s\n"
        "bgt 10b\n"
        "11:"  // K oddments: End
        "tbz x17, #1, 15f\n"
        "tbz x17, #0, 13f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "12:"  // Store to partial result buffer: Store and refill: Loop
        ".inst 0xa040c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16]\n"
        ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
        ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
        ".inst 0xa041c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
        ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
        ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
        ".inst 0xa042c610  // ld1w { z16.s-z19.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
        ".inst 0xa043c618  // ld1w { z24.s-z27.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
        ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
        "addvl x16, x16, #16\n"
        ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
        ".inst 0xa060c5f4  // st1w { z20.s-z23.s }, pn9.b, [x15]\n"
        ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
        ".inst 0xa061c5fc  // st1w { z28.s-z31.s }, pn9.b, [x15, #0x4, MUL VL]\n"
        ".inst 0xc0840703  // mova za3h.s[x12], { z24.s-z27.s }\n"
        "add x12, x12, #0x4\n"
        ".inst 0xa062c5e8  // st1w { z8.s-z11.s }, pn9.b, [x15, #0x8, MUL VL]\n"
        "cmp x12, x20\n"
        ".inst 0xa063c5ec  // st1w { z12.s-z15.s }, pn9.b, [x15, #0xc, MUL VL]\n"
        "addvl x15, x15, #16\n"
        "blt 12b\n"
        "b 31f\n"
        "13:"  // Store to partial result buffer: Store only
        "mov x12, #0x0\n"
        "cntw x20\n"
        "14:"  // Store to partial result buffer: Store only: Loop
        ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
        ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
        ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
        ".inst 0xc0860474  // mova { z20.s-z23.s }, za3h.s[x12]\n"
        ".inst 0xa060c5e0  // st1w { z0.s-z3.s }, pn9.b, [x15]\n"
        "add x12, x12, #0x4\n"
        ".inst 0xa061c5f0  // st1w { z16.s-z19.s }, pn9.b, [x15, #0x4, MUL VL]\n"
        "cmp x12, x20\n"
        ".inst 0xa062c5fc  // st1w { z28.s-z31.s }, pn9.b, [x15, #0x8, MUL VL]\n"
        ".inst 0xa063c5f4  // st1w { z20.s-z23.s }, pn9.b, [x15, #0xc, MUL VL]\n"
        "addvl x15, x15, #16\n"
        "blt 14b\n"
        "b 31f\n"
        "15:"  // Store to output array
        "ldr x26, [%x[args], %[offsetof_C]]\n"
        "sub x25, x14, x13\n"
        "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
        "add x26, x26, x11, LSL #2\n"  // C += n
        "madd x26, x13, x24, x26\n"    // C += m * ldc
        "tbz x17, #2, 22f\n"
        "cntw x23\n"
        "mov x12, #0x0\n"
        "cmp x25, x23\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 17f\n"
        "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
        ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
        ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
        ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "add x12, x12, #0x4\n"
        ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "blt 16b\n"
        "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
        "cbz x20, 18f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
        ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
        ".inst 0xa1604340  // st1w { z0.s, z8.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 18f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xa1604341  // st1w { z1.s, z9.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 18f\n"
        ".inst 0xa1604342  // st1w { z2.s, z10.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 22f\n"
        "cmp x25, x23\n"
        "mov x12, #0x0\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 20f\n"
        "19:"  // Store to output array: Skip activation: Accumulator row 1 loop
        ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
        ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
        ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "add x12, x12, #0x4\n"
        ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "blt 19b\n"
        "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments
        "cbz x20, 21f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
        ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
        ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 21f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 21f\n"
        ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "21:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
        "subs x25, x25, x22\n"
        "beq 22f\n"
        "b 29f\n"
        "22:"  // Store to output array: Skip activation: End
        "cntw x23\n"
        "ld1rw { z21.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "mov x12, #0x0\n"
        "cmp x25, x23\n"
        "ld1rw { z20.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 24f\n"
        "23:"  // Store to output array: Accumulator row 0 loop
        ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
        ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
        ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
        ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x21, LSL #2\n"
        ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604353  // st1w { z19.s, z27.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "blt 23b\n"
        "24:"  // Store to output array: Accumulator row 0 oddments
        "cbz x20, 25f\n"
        ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
        ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
        ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
        ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 25f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 25f\n"
        ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "25:"  // Store to output array: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 29f\n"
        "cmp x25, x23\n"
        "mov x12, #0x0\n"
        "csel x20, x25, x23, LT\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 27f\n"
        "26:"  // Store to output array: Accumulator row 1 loop
        ".inst 0xc0860440  // mova { z0.s-z3.s }, za2h.s[x12]\n"
        ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
        ".inst 0xc1b4caa0  // fclamp { z0.s-z3.s }, z21.s, z20.s\n"
        ".inst 0xc1b4caa8  // fclamp { z8.s-z11.s }, z21.s, z20.s\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x21, LSL #2\n"
        ".inst 0xa1604340  // st1w { z0.s, z8.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604341  // st1w { z1.s, z9.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604342  // st1w { z2.s, z10.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        ".inst 0xa1604343  // st1w { z3.s, z11.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "blt 26b\n"
        "27:"  // Store to output array: Accumulator row 1 oddments
        "cbz x20, 28f\n"
        ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
        ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
        ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
        ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 28f\n"
        "subs x20, x20, #0x1\n"
        ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
        "add x26, x26, x24\n"
        "beq 28f\n"
        ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
        "28:"  // Store to output array: Accumulator row 1 oddments: End
        "29:"  // Store to output array: End
        "tbz x17, #0, 31f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "30:"  // Store to output array: Refill accumulators: Loop
        ".inst 0xa040c608  // ld1w { z8.s-z11.s }, pn9.b/Z, [x16]\n"
        ".inst 0xa041c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
        ".inst 0xa042c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
        ".inst 0xa043c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
        ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
        "addvl x16, x16, #16\n"
        ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
        ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
        ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x20\n"
        "blt 30b\n"
        "31:"  // End block
        "incw x11, ALL, MUL #2\n"
        "cmp x11, x10\n"
        "blt 4b\n"
        "incw x13, ALL, MUL #2\n"
        "mov x11, #0x0\n"
        "cmp x13, x14\n"
        "mov x9, x27\n"
        "blt 3b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)), [offsetof_M] "I"(offsetof(KernelArgs, M)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)),
          [offsetof_accumulator_buffer] "I"(offsetof(KernelArgs, accumulator_buffer)),
          [offsetof_flags] "I"(offsetof(KernelArgs, flags)), [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25",
          "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
          "z29", "z30", "z31");
}

#endif  // Architectural features check.
