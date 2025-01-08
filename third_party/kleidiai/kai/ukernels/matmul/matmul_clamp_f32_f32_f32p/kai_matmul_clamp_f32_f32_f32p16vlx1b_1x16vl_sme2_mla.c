//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;
static const size_t kai_nr = 16;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);

    return m_idx * lhs_stride;
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla() == 0);
    return n_idx * (k * sizeof(float) + sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla() == 0);

    return (m_idx * dst_stride) + (n_idx * sizeof(float));
}

size_t kai_get_dst_size_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla(
    size_t m, size_t n, size_t k, const void* lhs, size_t lhs_stride, const void* rhs_packed, void* dst,
    size_t dst_stride_row, size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_UNUSED(lhs_stride);
    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(dst_stride_col);

    KAI_ASSUME(m == 1);

    typedef struct {
        float maxval;
        float minval;
    } KernelArgs;

    KernelArgs ka;
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    size_t N = n;
    size_t K = k;

    const void* A_ptr = lhs;
    const void* B_ptr = rhs_packed;
    void* output_ptr = dst;

    uint64_t flags = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x9, #0x0\n"
        "mov x27, %x[B_ptr]\n"
        "cntw x26, ALL, MUL #4\n"
        "mov x25, %x[output_ptr]\n"
        "add x24, %x[N], x26\n"
        "ptrue p1.b\n"
        "sub x24, x24, #0x1\n"
        ".inst 0x25207811  // ptrue pn9.b\n"
        "udiv x24, x24, x26\n"
        "mov x22, #0x1\n"
        "add x21, x24, #0x3\n"
        "and x21, x21, #0xfffffffffffffffc\n"
        "mul x21, x21, x26\n"
        "mul x21, x21, %x[K]\n"
        "lsl x21, x21, #0x2\n"
        "1:"  // RHS size check loop
        "cmp x21, #0x200000\n"
        "blt 2f\n"
        "tbnz x21, #0, 3f\n"
        "lsr x21, x21, #0x1\n"
        "lsl x22, x22, #0x1\n"
        "b 1b\n"
        "2:"  // RHS do prefetch
        "lsl x20, x21, #0x26\n"
        "sub x22, x22, #0x1\n"
        "lsl x22, x22, #0x16\n"
        "orr x21, x21, x20\n"
        "orr x21, x21, x22\n"
        ".inst 0xf8b54b7a  // rprfm pldonce, x21, [x27]\n"
        "3:"  // RHS prefetch exit
        "4:"  // Column loop
        "cmp x24, #0x4\n"
        "bge 22f\n"
        "cmp x24, #0x2\n"
        "bgt 16f\n"
        "beq 10f\n"
        ".inst 0xa040c774  // ld1w { z20.s-z23.s }, pn9.b/Z, [x27]\n"
        "mov x23, %x[K]\n"
        "mov x21, %x[N]\n"
        "mov x22, %x[A_ptr]\n"
        "lsl x20, %x[K], #0x2\n"
        ".inst 0x25b567f0  // whilelt p8.s, XZR, x21, VLx4\n"
        "cmp x23, #0x4\n"
        ".inst 0xf8b44ad8  // rprfm pldmany, x20, [x22]\n"
        ".inst 0xc0042e80  // mova za.d[x9, #0], { z20.d-z23.d }\n"
        "addvl x27, x27, #16\n"
        "ble 6f\n"
        "5:"  // Width 1: Multiply loop: Main loop head
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        "addvl x27, x27, #16\n"
        "ld1rqw { z2.s }, p0/Z, [x22]\n"
        "sub x23, x23, #0x4\n"
        "add x22, x22, #0x10\n"
        ".inst 0xa040c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27]\n"
        "addvl x27, x27, #16\n"
        "cmp x23, #0x4\n"
        ".inst 0xa040c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc152a380  // fmla za.s[x9, 0], { z28.s-z31.s }, z2.s[0]\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc152a600  // fmla za.s[x9, 0], { z16.s-z19.s }, z2.s[1]\n"
        ".inst 0xc152ab00  // fmla za.s[x9, 0], { z24.s-z27.s }, z2.s[2]\n"
        ".inst 0xc152ad80  // fmla za.s[x9, 0], { z12.s-z15.s }, z2.s[3]\n"
        "bgt 5b\n"
        "6:"  // Width 1: Multiply loop: Single iteration only
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a180  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[0]\n"
        "ble 7f\n"
        ".inst 0xa040c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a480  // fmla za.s[x9, 0], { z4.s-z7.s }, z3.s[1]\n"
        "ble 7f\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a980  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[2]\n"
        "ble 7f\n"
        ".inst 0xa040c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153ad00  // fmla za.s[x9, 0], { z8.s-z11.s }, z3.s[3]\n"
        "7:"  // Width 1: Multiply loop: multiply skip
        "tbz %x[flags], #1, 8f\n"
        "add x21, %x[args_ptr], %[offset_min]\n"
        "add x20, %x[args_ptr], %[offset_max]\n"
        ".inst 0xc0062c00  // mova { z0.d-z3.d }, za.d[x9, #0]\n"
        "ld1rw { z23.s }, p1/Z, [x21]\n"
        "ld1rw { z22.s }, p1/Z, [x20]\n"
        ".inst 0xc1b6cae0  // fclamp { z0.s-z3.s }, z23.s, z22.s\n"
        ".inst 0xa060c320  // st1w { z0.s-z3.s }, p8, [x25]\n"
        "b 9f\n"
        "8:"  // Width 1: No activation
        ".inst 0xc0062c00  // mova { z0.d-z3.d }, za.d[x9, #0]\n"
        ".inst 0xa060c320  // st1w { z0.s-z3.s }, p8, [x25]\n"
        "9:"  // Width 1: Output done
        "b 28f\n"
        "10:"  // Width 2
        ".inst 0xa040c77c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        "mov x23, %x[K]\n"
        "sub x21, %x[N], x26\n"
        ".inst 0xa041c764  // ld1w { z4.s-z7.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "mov x22, %x[A_ptr]\n"
        "lsl x20, %x[K], #0x2\n"
        ".inst 0x25b567f0  // whilelt p8.s, XZR, x21, VLx4\n"
        "cmp x23, #0x4\n"
        ".inst 0xf8b44ad8  // rprfm pldmany, x20, [x22]\n"
        ".inst 0xc0042f80  // mova za.d[x9, #0], { z28.d-z31.d }\n"
        "addvl x27, x27, #8\n"
        ".inst 0xc0042c81  // mova za.d[x9, #1], { z4.d-z7.d }\n"
        "ble 12f\n"
        "11:"  // Width 2: Multiply loop: Main loop head
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        "sub x23, x23, #0x4\n"
        "ld1rqw { z1.s }, p0/Z, [x22]\n"
        "cmp x23, #0x4\n"
        "add x22, x22, #0x10\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xa040c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc151a380  // fmla za.s[x9, 0], { z28.s-z31.s }, z1.s[0]\n"
        ".inst 0xa041c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc151a181  // fmla za.s[x9, 1], { z12.s-z15.s }, z1.s[0]\n"
        ".inst 0xa040c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xa040c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc151a600  // fmla za.s[x9, 0], { z16.s-z19.s }, z1.s[1]\n"
        ".inst 0xa041c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc151a701  // fmla za.s[x9, 1], { z24.s-z27.s }, z1.s[1]\n"
        ".inst 0xc151ab80  // fmla za.s[x9, 0], { z28.s-z31.s }, z1.s[2]\n"
        ".inst 0xc151a981  // fmla za.s[x9, 1], { z12.s-z15.s }, z1.s[2]\n"
        ".inst 0xc151ad00  // fmla za.s[x9, 0], { z8.s-z11.s }, z1.s[3]\n"
        ".inst 0xc151ae81  // fmla za.s[x9, 1], { z20.s-z23.s }, z1.s[3]\n"
        "bgt 11b\n"
        "12:"  // Width 2: Multiply loop: Single iteration only
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        ".inst 0xa041c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a200  // fmla za.s[x9, 0], { z16.s-z19.s }, z3.s[0]\n"
        ".inst 0xc153a381  // fmla za.s[x9, 1], { z28.s-z31.s }, z3.s[0]\n"
        "ble 13f\n"
        ".inst 0xa040c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a680  // fmla za.s[x9, 0], { z20.s-z23.s }, z3.s[1]\n"
        ".inst 0xc153a601  // fmla za.s[x9, 1], { z16.s-z19.s }, z3.s[1]\n"
        "ble 13f\n"
        ".inst 0xa040c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a880  // fmla za.s[x9, 0], { z4.s-z7.s }, z3.s[2]\n"
        ".inst 0xc153aa01  // fmla za.s[x9, 1], { z16.s-z19.s }, z3.s[2]\n"
        "ble 13f\n"
        ".inst 0xa040c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153af80  // fmla za.s[x9, 0], { z28.s-z31.s }, z3.s[3]\n"
        ".inst 0xc153ad81  // fmla za.s[x9, 1], { z12.s-z15.s }, z3.s[3]\n"
        "13:"  // Width 2: Multiply loop: multiply skip
        "tbz %x[flags], #1, 14f\n"
        "add x21, %x[args_ptr], %[offset_min]\n"
        "add x20, %x[args_ptr], %[offset_max]\n"
        ".inst 0xc0062c04  // mova { z4.d-z7.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c28  // mova { z8.d-z11.d }, za.d[x9, #1]\n"
        "ld1rw { z17.s }, p1/Z, [x21]\n"
        "ld1rw { z23.s }, p1/Z, [x20]\n"
        ".inst 0xc1b7ca24  // fclamp { z4.s-z7.s }, z17.s, z23.s\n"
        ".inst 0xc1b7ca28  // fclamp { z8.s-z11.s }, z17.s, z23.s\n"
        ".inst 0xa060c724  // st1w { z4.s-z7.s }, pn9.b, [x25]\n"
        ".inst 0xa061c328  // st1w { z8.s-z11.s }, p8, [x25, #0x4, MUL VL]\n"
        "b 15f\n"
        "14:"  // Width 2: No activation
        ".inst 0xc0062c08  // mova { z8.d-z11.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c30  // mova { z16.d-z19.d }, za.d[x9, #1]\n"
        ".inst 0xa060c728  // st1w { z8.s-z11.s }, pn9.b, [x25]\n"
        ".inst 0xa061c330  // st1w { z16.s-z19.s }, p8, [x25, #0x4, MUL VL]\n"
        "15:"  // Width 2: Output done
        "b 28f\n"
        "16:"  // Width 3
        "mov x20, #0x2\n"
        ".inst 0xa040c768  // ld1w { z8.s-z11.s }, pn9.b/Z, [x27]\n"
        "mov x23, %x[K]\n"
        ".inst 0xa041c760  // ld1w { z0.s-z3.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "msub x21, x26, x20, %x[N]\n"
        "mov x22, %x[A_ptr]\n"
        ".inst 0xa042c764  // ld1w { z4.s-z7.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "lsl x20, %x[K], #0x2\n"
        ".inst 0x25b567f0  // whilelt p8.s, XZR, x21, VLx4\n"
        "cmp x23, #0x4\n"
        ".inst 0xf8b44ad8  // rprfm pldmany, x20, [x22]\n"
        ".inst 0xc0042d00  // mova za.d[x9, #0], { z8.d-z11.d }\n"
        ".inst 0xc0042c01  // mova za.d[x9, #1], { z0.d-z3.d }\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc0042c82  // mova za.d[x9, #2], { z4.d-z7.d }\n"
        "ble 18f\n"
        "17:"  // Width 3: Multiply loop: Main loop head
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "sub x23, x23, #0x4\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        "cmp x23, #0x4\n"
        "add x22, x22, #0x10\n"
        ".inst 0xa041c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a180  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[0]\n"
        ".inst 0xa040c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153a101  // fmla za.s[x9, 1], { z8.s-z11.s }, z3.s[0]\n"
        ".inst 0xa041c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a082  // fmla za.s[x9, 2], { z4.s-z7.s }, z3.s[0]\n"
        ".inst 0xa042c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153a600  // fmla za.s[x9, 0], { z16.s-z19.s }, z3.s[1]\n"
        ".inst 0xa041c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a681  // fmla za.s[x9, 1], { z20.s-z23.s }, z3.s[1]\n"
        ".inst 0xa042c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a702  // fmla za.s[x9, 2], { z24.s-z27.s }, z3.s[1]\n"
        ".inst 0xa040c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa041c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a980  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[2]\n"
        ".inst 0xa042c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153ab81  // fmla za.s[x9, 1], { z28.s-z31.s }, z3.s[2]\n"
        ".inst 0xc153a902  // fmla za.s[x9, 2], { z8.s-z11.s }, z3.s[2]\n"
        ".inst 0xc153ac80  // fmla za.s[x9, 0], { z4.s-z7.s }, z3.s[3]\n"
        ".inst 0xc153ae81  // fmla za.s[x9, 1], { z20.s-z23.s }, z3.s[3]\n"
        ".inst 0xc153af02  // fmla za.s[x9, 2], { z24.s-z27.s }, z3.s[3]\n"
        "bgt 17b\n"
        "18:"  // Width 3: Multiply loop: Single iteration only
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a280  // fmla za.s[x9, 0], { z20.s-z23.s }, z3.s[0]\n"
        ".inst 0xc153a181  // fmla za.s[x9, 1], { z12.s-z15.s }, z3.s[0]\n"
        ".inst 0xc153a082  // fmla za.s[x9, 2], { z4.s-z7.s }, z3.s[0]\n"
        "ble 19f\n"
        ".inst 0xa040c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a680  // fmla za.s[x9, 0], { z20.s-z23.s }, z3.s[1]\n"
        ".inst 0xc153a501  // fmla za.s[x9, 1], { z8.s-z11.s }, z3.s[1]\n"
        ".inst 0xc153a602  // fmla za.s[x9, 2], { z16.s-z19.s }, z3.s[1]\n"
        "ble 19f\n"
        ".inst 0xa040c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153ab80  // fmla za.s[x9, 0], { z28.s-z31.s }, z3.s[2]\n"
        ".inst 0xc153ab01  // fmla za.s[x9, 1], { z24.s-z27.s }, z3.s[2]\n"
        ".inst 0xc153a982  // fmla za.s[x9, 2], { z12.s-z15.s }, z3.s[2]\n"
        "ble 19f\n"
        ".inst 0xa040c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa041c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xc153ad00  // fmla za.s[x9, 0], { z8.s-z11.s }, z3.s[3]\n"
        ".inst 0xc153af81  // fmla za.s[x9, 1], { z28.s-z31.s }, z3.s[3]\n"
        ".inst 0xc153ad82  // fmla za.s[x9, 2], { z12.s-z15.s }, z3.s[3]\n"
        "19:"  // Width 3: Multiply loop: multiply skip
        "tbz %x[flags], #1, 20f\n"
        "add x21, %x[args_ptr], %[offset_min]\n"
        "add x20, %x[args_ptr], %[offset_max]\n"
        ".inst 0xc0062c08  // mova { z8.d-z11.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c2c  // mova { z12.d-z15.d }, za.d[x9, #1]\n"
        "ld1rw { z21.s }, p1/Z, [x21]\n"
        ".inst 0xc0062c50  // mova { z16.d-z19.d }, za.d[x9, #2]\n"
        "ld1rw { z20.s }, p1/Z, [x20]\n"
        ".inst 0xc1b4caa8  // fclamp { z8.s-z11.s }, z21.s, z20.s\n"
        ".inst 0xc1b4caac  // fclamp { z12.s-z15.s }, z21.s, z20.s\n"
        ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
        ".inst 0xa060c728  // st1w { z8.s-z11.s }, pn9.b, [x25]\n"
        ".inst 0xa061c72c  // st1w { z12.s-z15.s }, pn9.b, [x25, #0x4, MUL VL]\n"
        ".inst 0xa062c330  // st1w { z16.s-z19.s }, p8, [x25, #0x8, MUL VL]\n"
        "b 21f\n"
        "20:"  // Width 3: No activation
        ".inst 0xc0062c04  // mova { z4.d-z7.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c2c  // mova { z12.d-z15.d }, za.d[x9, #1]\n"
        ".inst 0xc0062c5c  // mova { z28.d-z31.d }, za.d[x9, #2]\n"
        ".inst 0xa060c724  // st1w { z4.s-z7.s }, pn9.b, [x25]\n"
        ".inst 0xa061c72c  // st1w { z12.s-z15.s }, pn9.b, [x25, #0x4, MUL VL]\n"
        ".inst 0xa062c33c  // st1w { z28.s-z31.s }, p8, [x25, #0x8, MUL VL]\n"
        "21:"  // Width 3: Output done
        "b 28f\n"
        "22:"  // Width 4
        "mov x20, #0x3\n"
        ".inst 0xa040c764  // ld1w { z4.s-z7.s }, pn9.b/Z, [x27]\n"
        "mov x23, %x[K]\n"
        ".inst 0xa041c76c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        "msub x21, x26, x20, %x[N]\n"
        "mov x22, %x[A_ptr]\n"
        ".inst 0xa042c77c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        "lsl x20, %x[K], #0x2\n"
        ".inst 0x25b567f0  // whilelt p8.s, XZR, x21, VLx4\n"
        ".inst 0xa043c770  // ld1w { z16.s-z19.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        "cmp x23, #0x4\n"
        ".inst 0xf8b44ad8  // rprfm pldmany, x20, [x22]\n"
        ".inst 0xc0042c80  // mova za.d[x9, #0], { z4.d-z7.d }\n"
        ".inst 0xc0042d81  // mova za.d[x9, #1], { z12.d-z15.d }\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc0042f82  // mova za.d[x9, #2], { z28.d-z31.d }\n"
        ".inst 0xc0042e03  // mova za.d[x9, #3], { z16.d-z19.d }\n"
        "ble 24f\n"
        "23:"  // Width 4: Multiply loop: Main loop head
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "sub x23, x23, #0x4\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        "cmp x23, #0x4\n"
        "add x22, x22, #0x10\n"
        ".inst 0xa041c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153a180  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[0]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a281  // fmla za.s[x9, 1], { z20.s-z23.s }, z3.s[0]\n"
        ".inst 0xa040c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153a202  // fmla za.s[x9, 2], { z16.s-z19.s }, z3.s[0]\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a103  // fmla za.s[x9, 3], { z8.s-z11.s }, z3.s[0]\n"
        ".inst 0xa042c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153a700  // fmla za.s[x9, 0], { z24.s-z27.s }, z3.s[1]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a581  // fmla za.s[x9, 1], { z12.s-z15.s }, z3.s[1]\n"
        ".inst 0xa040c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153a502  // fmla za.s[x9, 2], { z8.s-z11.s }, z3.s[1]\n"
        ".inst 0xa041c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a483  // fmla za.s[x9, 3], { z4.s-z7.s }, z3.s[1]\n"
        ".inst 0xa042c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153ab00  // fmla za.s[x9, 0], { z24.s-z27.s }, z3.s[2]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a901  // fmla za.s[x9, 1], { z8.s-z11.s }, z3.s[2]\n"
        ".inst 0xa040c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27]\n"
        ".inst 0xc153aa02  // fmla za.s[x9, 2], { z16.s-z19.s }, z3.s[2]\n"
        ".inst 0xa041c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xc153a883  // fmla za.s[x9, 3], { z4.s-z7.s }, z3.s[2]\n"
        ".inst 0xa042c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153ad00  // fmla za.s[x9, 0], { z8.s-z11.s }, z3.s[3]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153af81  // fmla za.s[x9, 1], { z28.s-z31.s }, z3.s[3]\n"
        ".inst 0xc153ad82  // fmla za.s[x9, 2], { z12.s-z15.s }, z3.s[3]\n"
        ".inst 0xc153ae83  // fmla za.s[x9, 3], { z20.s-z23.s }, z3.s[3]\n"
        "bgt 23b\n"
        "24:"  // Width 4: Multiply loop: Single iteration only
        "whilelt p0.s, XZR, x23\n"
        ".inst 0xa040c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        "ld1rqw { z3.s }, p0/Z, [x22]\n"
        ".inst 0xa041c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c77d  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153a200  // fmla za.s[x9, 0], { z16.s-z19.s }, z3.s[0]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a181  // fmla za.s[x9, 1], { z12.s-z15.s }, z3.s[0]\n"
        ".inst 0xc153a382  // fmla za.s[x9, 2], { z28.s-z31.s }, z3.s[0]\n"
        ".inst 0xc153a283  // fmla za.s[x9, 3], { z20.s-z23.s }, z3.s[0]\n"
        "ble 25f\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c765  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c779  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153a580  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[1]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a481  // fmla za.s[x9, 1], { z4.s-z7.s }, z3.s[1]\n"
        ".inst 0xc153a702  // fmla za.s[x9, 2], { z24.s-z27.s }, z3.s[1]\n"
        ".inst 0xc153a683  // fmla za.s[x9, 3], { z20.s-z23.s }, z3.s[1]\n"
        "ble 25f\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        "subs x23, x23, #0x1\n"
        ".inst 0xa041c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153a980  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[2]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153a901  // fmla za.s[x9, 1], { z8.s-z11.s }, z3.s[2]\n"
        ".inst 0xc153aa82  // fmla za.s[x9, 2], { z20.s-z23.s }, z3.s[2]\n"
        ".inst 0xc153aa03  // fmla za.s[x9, 3], { z16.s-z19.s }, z3.s[2]\n"
        "ble 25f\n"
        ".inst 0xa040c76d  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x27]\n"
        ".inst 0xa041c769  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa042c775  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
        ".inst 0xa043c771  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
        ".inst 0xc153ad80  // fmla za.s[x9, 0], { z12.s-z15.s }, z3.s[3]\n"
        "addvl x27, x27, #16\n"
        ".inst 0xc153ad01  // fmla za.s[x9, 1], { z8.s-z11.s }, z3.s[3]\n"
        ".inst 0xc153ae82  // fmla za.s[x9, 2], { z20.s-z23.s }, z3.s[3]\n"
        ".inst 0xc153ae03  // fmla za.s[x9, 3], { z16.s-z19.s }, z3.s[3]\n"
        "25:"  // Width 4: Multiply loop: multiply skip
        "tbz %x[flags], #1, 26f\n"
        "add x21, %x[args_ptr], %[offset_min]\n"
        "add x20, %x[args_ptr], %[offset_max]\n"
        ".inst 0xc0062c04  // mova { z4.d-z7.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c20  // mova { z0.d-z3.d }, za.d[x9, #1]\n"
        "ld1rw { z21.s }, p1/Z, [x21]\n"
        ".inst 0xc0062c4c  // mova { z12.d-z15.d }, za.d[x9, #2]\n"
        "ld1rw { z20.s }, p1/Z, [x20]\n"
        ".inst 0xc0062c70  // mova { z16.d-z19.d }, za.d[x9, #3]\n"
        ".inst 0xc1b4caa4  // fclamp { z4.s-z7.s }, z21.s, z20.s\n"
        ".inst 0xc1b4caa0  // fclamp { z0.s-z3.s }, z21.s, z20.s\n"
        ".inst 0xc1b4caac  // fclamp { z12.s-z15.s }, z21.s, z20.s\n"
        ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
        ".inst 0xa060c724  // st1w { z4.s-z7.s }, pn9.b, [x25]\n"
        ".inst 0xa061c720  // st1w { z0.s-z3.s }, pn9.b, [x25, #0x4, MUL VL]\n"
        ".inst 0xa062c72c  // st1w { z12.s-z15.s }, pn9.b, [x25, #0x8, MUL VL]\n"
        ".inst 0xa063c330  // st1w { z16.s-z19.s }, p8, [x25, #0xc, MUL VL]\n"
        "addvl x25, x25, #16\n"
        "b 27f\n"
        "26:"  // Width 4: No activation
        ".inst 0xc0062c0c  // mova { z12.d-z15.d }, za.d[x9, #0]\n"
        ".inst 0xc0062c20  // mova { z0.d-z3.d }, za.d[x9, #1]\n"
        ".inst 0xc0062c50  // mova { z16.d-z19.d }, za.d[x9, #2]\n"
        ".inst 0xc0062c64  // mova { z4.d-z7.d }, za.d[x9, #3]\n"
        ".inst 0xa060c72c  // st1w { z12.s-z15.s }, pn9.b, [x25]\n"
        ".inst 0xa061c720  // st1w { z0.s-z3.s }, pn9.b, [x25, #0x4, MUL VL]\n"
        ".inst 0xa062c730  // st1w { z16.s-z19.s }, pn9.b, [x25, #0x8, MUL VL]\n"
        ".inst 0xa063c324  // st1w { z4.s-z7.s }, p8, [x25, #0xc, MUL VL]\n"
        "addvl x25, x25, #16\n"
        "27:"  // Width 4: Output done
        "subs x24, x24, #0x4\n"
        "sub %x[N], %x[N], x26, LSL #2\n"
        "bgt 4b\n"
        "28:"  // Exit
        ".inst 0xd503467f  // SMSTOP\n"
        : [N] "+&r"(N)
        : [A_ptr] "r"(A_ptr), [B_ptr] "r"(B_ptr), [K] "r"(K), [args_ptr] "r"(&ka), [flags] "r"(flags),
          [offset_max] "I"(offsetof(KernelArgs, maxval)), [offset_min] "I"(offsetof(KernelArgs, minval)),
          [output_ptr] "r"(output_ptr)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x9", "z0", "z1", "z10", "z11", "z12",
          "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
          "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
