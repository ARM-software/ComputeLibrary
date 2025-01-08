//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(int8_t);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    return n_idx * (sizeof(int32_t) + kai_roundup(k, kai_kr) * sizeof(int8_t) + sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);

    return m_idx * dst_stride + n_idx * sizeof(int8_t);
}

size_t kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

void kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct kai_matmul_requantize32_params* params) {
    KAI_ASSUME(dst_stride_col == sizeof(int8_t));

    typedef struct {
        const void* A;
        const void* B;

        void* C;
        uint64_t ldcb;
        uint64_t M, N, K;
        int32_t min;
        int32_t max;
        int32_t result_zero_point;

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
    args.min = params->min_value;
    args.max = params->max_value;
    args.result_zero_point = params->output_zero_point;

    args.accumulator_buffer = NULL;
    args.flags = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "ldr w14, [%x[args], %[offsetof_M]]\n"
        "mov x13, #0x0\n"
        "mov x11, #0x0\n"
        "ptrue p1.b\n"
        ".inst 0x25207811  // ptrue pn9.b\n"
        "ldr w10, [%x[args], %[offsetof_N]]\n"
        "ldr x9, [%x[args], %[offsetof_A]]\n"
        "1:"  // M loop
        "ldr x28, [%x[args], %[offsetof_B]]\n"
        "2:"  // N loop
        ".inst 0x25aa4570  // whilelt pn8.s, x11, x10, VLx2\n"
        ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
        "mov x27, x9\n"
        ".inst 0xa040438e  // ld1w { z14.s-z15.s }, p8/Z, [x28]\n"  // Load bias
        "addvl x28, x28, #2\n"
        ".inst 0xc09025c0  // addha za0.s, p1/M, p1/M, z14.s\n"
        ".inst 0xc09025e1  // addha za1.s, p1/M, p1/M, z15.s\n"
        ".inst 0xc09025c2  // addha za2.s, p1/M, p1/M, z14.s\n"
        ".inst 0xc09025e3  // addha za3.s, p1/M, p1/M, z15.s\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "add x20, x20, #0x3\n"
        "lsr x20, x20, #0x2\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 6f\n"
        "subs x21, x21, #0x1\n"
        ".inst 0xa0400762  // ld1b { z2.b-z3.b }, pn9.b/Z, [x27]\n"
        ".inst 0xa1400780  // ld1b { z0.b, z8.b }, pn9.b/Z, [x28]\n"
        ".inst 0xa0410772  // ld1b { z18.b-z19.b }, pn9.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0xa0410794  // ld1b { z20.b-z21.b }, pn9.b/Z, [x28, #0x2, MUL VL]\n"
        ".inst 0xa042077a  // ld1b { z26.b-z27.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa0420796  // ld1b { z22.b-z23.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
        ".inst 0xa0430778  // ld1b { z24.b-z25.b }, pn9.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        ".inst 0xa0430784  // ld1b { z4.b-z5.b }, pn9.b/Z, [x28, #0x6, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "ble 5f\n"
        "4:"  // K loop
        ".inst 0xa0802440  // smopa za0.s, p1/M, p1/M, z2.b, z0.b\n"
        "subs x21, x21, #0x1\n"
        ".inst 0xa0882441  // smopa za1.s, p1/M, p1/M, z2.b, z8.b\n"
        ".inst 0xa0802462  // smopa za2.s, p1/M, p1/M, z3.b, z0.b\n"
        ".inst 0xa0882463  // smopa za3.s, p1/M, p1/M, z3.b, z8.b\n"
        ".inst 0xa0400762  // ld1b { z2.b-z3.b }, pn9.b/Z, [x27]\n"
        ".inst 0xa0942640  // smopa za0.s, p1/M, p1/M, z18.b, z20.b\n"
        ".inst 0xa1400780  // ld1b { z0.b, z8.b }, pn9.b/Z, [x28]\n"
        ".inst 0xa0952641  // smopa za1.s, p1/M, p1/M, z18.b, z21.b\n"
        ".inst 0xa0942662  // smopa za2.s, p1/M, p1/M, z19.b, z20.b\n"
        ".inst 0xa0952663  // smopa za3.s, p1/M, p1/M, z19.b, z21.b\n"
        ".inst 0xa0410772  // ld1b { z18.b-z19.b }, pn9.b/Z, [x27, #0x2, MUL VL]\n"
        ".inst 0xa0962740  // smopa za0.s, p1/M, p1/M, z26.b, z22.b\n"
        ".inst 0xa0410794  // ld1b { z20.b-z21.b }, pn9.b/Z, [x28, #0x2, MUL VL]\n"
        ".inst 0xa0972741  // smopa za1.s, p1/M, p1/M, z26.b, z23.b\n"
        ".inst 0xa0962762  // smopa za2.s, p1/M, p1/M, z27.b, z22.b\n"
        ".inst 0xa0972763  // smopa za3.s, p1/M, p1/M, z27.b, z23.b\n"
        ".inst 0xa042077a  // ld1b { z26.b-z27.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
        ".inst 0xa0420796  // ld1b { z22.b-z23.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
        ".inst 0xa0842700  // smopa za0.s, p1/M, p1/M, z24.b, z4.b\n"
        ".inst 0xa0852701  // smopa za1.s, p1/M, p1/M, z24.b, z5.b\n"
        ".inst 0xa0842722  // smopa za2.s, p1/M, p1/M, z25.b, z4.b\n"
        ".inst 0xa0852723  // smopa za3.s, p1/M, p1/M, z25.b, z5.b\n"
        ".inst 0xa0430778  // ld1b { z24.b-z25.b }, pn9.b/Z, [x27, #0x6, MUL VL]\n"
        "addvl x27, x27, #8\n"
        ".inst 0xa0430784  // ld1b { z4.b-z5.b }, pn9.b/Z, [x28, #0x6, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "bgt 4b\n"
        "5:"  // K loop tail
        ".inst 0xa0802440  // smopa za0.s, p1/M, p1/M, z2.b, z0.b\n"
        ".inst 0xa0882441  // smopa za1.s, p1/M, p1/M, z2.b, z8.b\n"
        ".inst 0xa0802462  // smopa za2.s, p1/M, p1/M, z3.b, z0.b\n"
        ".inst 0xa0882463  // smopa za3.s, p1/M, p1/M, z3.b, z8.b\n"
        ".inst 0xa0942640  // smopa za0.s, p1/M, p1/M, z18.b, z20.b\n"
        ".inst 0xa0952641  // smopa za1.s, p1/M, p1/M, z18.b, z21.b\n"
        ".inst 0xa0942662  // smopa za2.s, p1/M, p1/M, z19.b, z20.b\n"
        ".inst 0xa0952663  // smopa za3.s, p1/M, p1/M, z19.b, z21.b\n"
        ".inst 0xa0962740  // smopa za0.s, p1/M, p1/M, z26.b, z22.b\n"
        ".inst 0xa0972741  // smopa za1.s, p1/M, p1/M, z26.b, z23.b\n"
        ".inst 0xa0962762  // smopa za2.s, p1/M, p1/M, z27.b, z22.b\n"
        ".inst 0xa0972763  // smopa za3.s, p1/M, p1/M, z27.b, z23.b\n"
        ".inst 0xa0842700  // smopa za0.s, p1/M, p1/M, z24.b, z4.b\n"
        ".inst 0xa0852701  // smopa za1.s, p1/M, p1/M, z24.b, z5.b\n"
        ".inst 0xa0842722  // smopa za2.s, p1/M, p1/M, z25.b, z4.b\n"
        ".inst 0xa0852723  // smopa za3.s, p1/M, p1/M, z25.b, z5.b\n"
        "6:"  // K oddments
        "cbz x20, 8f\n"
        "7:"  // K oddments: Loop
        ".inst 0xa0400770  // ld1b { z16.b-z17.b }, pn9.b/Z, [x27]\n"
        "subs x20, x20, #0x1\n"
        "addvl x27, x27, #2\n"
        ".inst 0xa0400788  // ld1b { z8.b-z9.b }, pn9.b/Z, [x28]\n"
        "addvl x28, x28, #2\n"
        ".inst 0xa0882600  // smopa za0.s, p1/M, p1/M, z16.b, z8.b\n"
        ".inst 0xa0892601  // smopa za1.s, p1/M, p1/M, z16.b, z9.b\n"
        ".inst 0xa0882622  // smopa za2.s, p1/M, p1/M, z17.b, z8.b\n"
        ".inst 0xa0892623  // smopa za3.s, p1/M, p1/M, z17.b, z9.b\n"
        "bgt 7b\n"
        "8:"  // K oddments: End
        "ldr x26, [%x[args], %[offsetof_C]]\n"
        "sub x25, x14, x13\n"
        "cntw x24\n"
        "ld1rw { z27.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
        "whilelt p0.h, x11, x10\n"
        "cmp x25, x24\n"
        "ld1rw { z1.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "csel x22, x25, x24, LT\n"
        "ld1rw { z0.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_result_zero_point]]\n"
        "mov x12, #0x0\n"
        "add x26, x26, x11\n"  // C += n
        "lsr x21, x22, #0x2\n"
        "ld1w { z22.s }, p1/Z, [x28]\n"
        "madd x26, x13, x23, x26\n"  // C += m * ldc
        "ld1w { z26.s }, p1/Z, [x28, #1, MUL VL]\n"
        "and x20, x22, #0x3\n"
        "addvl x28, x28, #2\n"
        "cbz x21, 11f\n"
        "10:"  // Store to output array: Accumulator row 0 loop
        ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
        ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
        ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
        ".inst 0xc132e39c  // scvtf { z28.s-z31.s }, { z28.s-z31.s }\n"
        "fmul z16.s, z16.s, z22.s\n"
        "fmul z17.s, z17.s, z22.s\n"
        "add x12, x12, #0x4\n"
        "fmul z18.s, z18.s, z22.s\n"
        "fmul z19.s, z19.s, z22.s\n"
        "cmp x12, x21, LSL #2\n"
        "fmul z28.s, z28.s, z26.s\n"
        "fmul z29.s, z29.s, z26.s\n"
        "fmul z30.s, z30.s, z26.s\n"
        "fmul z31.s, z31.s, z26.s\n"
        ".inst 0xc1b8e210  // frintn { z16.s-z19.s }, { z16.s-z19.s }\n"
        ".inst 0xc131e210  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }\n"
        ".inst 0xc1b8e39c  // frintn { z28.s-z31.s }, { z28.s-z31.s }\n"
        ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
        ".inst 0xc131e39c  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }\n"
        ".inst 0xc1a0ab1c  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s\n"
        ".inst 0xc1a1cf70  // sclamp { z16.s-z19.s }, z27.s, z1.s\n"
        ".inst 0xc1a1cf7c  // sclamp { z28.s-z31.s }, z27.s, z1.s\n"
        "uzp1 z5.h, z16.h, z28.h\n"
        "uzp1 z20.h, z17.h, z29.h\n"
        "uzp1 z17.h, z18.h, z30.h\n"
        "uzp1 z16.h, z19.h, z31.h\n"
        "st1b { z5.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z20.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z17.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "blt 10b\n"
        "11:"  // Store to output array: Accumulator row 0 oddments
        "cbz x20, 12f\n"
        ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
        ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
        ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
        ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
        "fmul z4.s, z4.s, z22.s\n"
        "fmul z5.s, z5.s, z22.s\n"
        "subs x20, x20, #0x1\n"
        "fmul z6.s, z6.s, z22.s\n"
        "fmul z7.s, z7.s, z22.s\n"
        "fmul z12.s, z12.s, z26.s\n"
        "fmul z13.s, z13.s, z26.s\n"
        "fmul z14.s, z14.s, z26.s\n"
        "fmul z15.s, z15.s, z26.s\n"
        ".inst 0xc1b8e084  // frintn { z4.s-z7.s }, { z4.s-z7.s }\n"
        ".inst 0xc131e084  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }\n"
        ".inst 0xc1b8e18c  // frintn { z12.s-z15.s }, { z12.s-z15.s }\n"
        ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
        ".inst 0xc131e18c  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }\n"
        ".inst 0xc1a0ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s\n"
        ".inst 0xc1a1cf64  // sclamp { z4.s-z7.s }, z27.s, z1.s\n"
        ".inst 0xc1a1cf6c  // sclamp { z12.s-z15.s }, z27.s, z1.s\n"
        "uzp1 z16.h, z4.h, z12.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "beq 12f\n"
        "subs x20, x20, #0x1\n"
        "uzp1 z16.h, z5.h, z13.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "beq 12f\n"
        "uzp1 z16.h, z6.h, z14.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "12:"  // Store to output array: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 16f\n"
        "cmp x25, x24\n"
        "mov x12, #0x0\n"
        "csel x20, x25, x24, LT\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 14f\n"
        "13:"  // Store to output array: Accumulator row 1 loop
        ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
        ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
        ".inst 0xc132e108  // scvtf { z8.s-z11.s }, { z8.s-z11.s }\n"
        ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
        "fmul z8.s, z8.s, z22.s\n"
        "fmul z9.s, z9.s, z22.s\n"
        "add x12, x12, #0x4\n"
        "fmul z10.s, z10.s, z22.s\n"
        "fmul z11.s, z11.s, z22.s\n"
        "cmp x12, x21, LSL #2\n"
        "fmul z16.s, z16.s, z26.s\n"
        "fmul z17.s, z17.s, z26.s\n"
        "fmul z18.s, z18.s, z26.s\n"
        "fmul z19.s, z19.s, z26.s\n"
        ".inst 0xc1b8e108  // frintn { z8.s-z11.s }, { z8.s-z11.s }\n"
        ".inst 0xc131e108  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }\n"
        ".inst 0xc1b8e210  // frintn { z16.s-z19.s }, { z16.s-z19.s }\n"
        ".inst 0xc1a0ab08  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s\n"
        ".inst 0xc131e210  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }\n"
        ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
        ".inst 0xc1a1cf68  // sclamp { z8.s-z11.s }, z27.s, z1.s\n"
        ".inst 0xc1a1cf70  // sclamp { z16.s-z19.s }, z27.s, z1.s\n"
        "uzp1 z21.h, z8.h, z16.h\n"
        "uzp1 z20.h, z9.h, z17.h\n"
        "uzp1 z17.h, z10.h, z18.h\n"
        "uzp1 z16.h, z11.h, z19.h\n"
        "st1b { z21.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z20.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z17.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "blt 13b\n"
        "14:"  // Store to output array: Accumulator row 1 oddments
        "cbz x20, 15f\n"
        ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
        ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
        ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
        ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
        "fmul z12.s, z12.s, z22.s\n"
        "fmul z13.s, z13.s, z22.s\n"
        "subs x20, x20, #0x1\n"
        "fmul z14.s, z14.s, z22.s\n"
        "fmul z15.s, z15.s, z22.s\n"
        "fmul z4.s, z4.s, z26.s\n"
        "fmul z5.s, z5.s, z26.s\n"
        "fmul z6.s, z6.s, z26.s\n"
        "fmul z7.s, z7.s, z26.s\n"
        ".inst 0xc1b8e18c  // frintn { z12.s-z15.s }, { z12.s-z15.s }\n"
        ".inst 0xc131e18c  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }\n"
        ".inst 0xc1b8e084  // frintn { z4.s-z7.s }, { z4.s-z7.s }\n"
        ".inst 0xc1a0ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s\n"
        ".inst 0xc131e084  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }\n"
        ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
        ".inst 0xc1a1cf6c  // sclamp { z12.s-z15.s }, z27.s, z1.s\n"
        ".inst 0xc1a1cf64  // sclamp { z4.s-z7.s }, z27.s, z1.s\n"
        "uzp1 z16.h, z12.h, z4.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "beq 15f\n"
        "subs x20, x20, #0x1\n"
        "uzp1 z16.h, z13.h, z5.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "add x26, x26, x23\n"
        "beq 15f\n"
        "uzp1 z16.h, z14.h, z6.h\n"
        "st1b { z16.h }, p0, [x26]\n"
        "15:"  // Store to output array: Accumulator row 1 oddments: End
        "16:"  // Store to output array: End
        "incw x11, ALL, MUL #2\n"
        "cmp x11, x10\n"
        "blt 2b\n"
        "incw x13, ALL, MUL #2\n"
        "mov x11, #0x0\n"
        "cmp x13, x14\n"
        "mov x9, x27\n"
        "blt 1b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)),
          [offsetof_KernelArgs_result_zero_point] "I"(offsetof(KernelArgs, result_zero_point)),
          [offsetof_M] "I"(offsetof(KernelArgs, M)), [offsetof_N] "I"(offsetof(KernelArgs, N)),
          [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

#endif  // Architectural features check.
