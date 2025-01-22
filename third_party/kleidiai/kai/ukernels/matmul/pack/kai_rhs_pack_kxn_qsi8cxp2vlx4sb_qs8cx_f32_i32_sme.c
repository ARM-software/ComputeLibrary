//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_num_bytes_input = 1;
static const size_t kai_num_bytes_output = 1;
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_num_bytes_scale = 4;

size_t kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % (kai_nr * kai_get_sme_vector_length_u8() / kai_kr) == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_scale_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_scale;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t k) {
    return kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() *
        (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output + kai_num_bytes_scale);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() == 0);

    return (n_idx / kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme()) *
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n, size_t k) {
    return kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
        kai_roundup(n, kai_nr * kai_get_sme_vector_length_u8() / kai_kr), k);
}

void kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_qsi8cx_params* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_nr * kai_get_sme_vector_length_u8() / kai_kr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params != NULL);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    uint8_t pad_row[nr];

    if (height % 4) {
        memset(pad_row, 0, nr * sizeof(uint8_t));
    }

    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(height);
    const int32_t lhs_zero_point = params->lhs_zero_point;
    const float scale_multiplier = params->scale_multiplier;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cmp %x[height], #0x8\n"
        "mov x11, %x[out]\n"
        "ptrue p2.b\n"
        "mov x10, %x[height]\n"
        "incb %x[out], ALL, MUL #2\n"
        "blt 4f\n"
        "1:"  // Main row loop: Head
        "mov x9, %x[in]\n"
        "mov x28, %x[out]\n"
        "add x27, x9, %x[in_stride]\n"
        "sub %x[height], %x[height], #0x8\n"
        "add x26, x27, %x[in_stride]\n"
        "mov x24, %x[width]\n"
        "add x25, x26, %x[in_stride]\n"
        "add x23, x25, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "add %x[in], x20, %x[in_stride]\n"
        "2:"  // Main row loop: Column loop
        "whilelt p0.b, XZR, x24\n"
        "decw x24, ALL, MUL #2\n"
        "ld1b { z18.b }, p0/Z, [x9]\n"
        "cmp x24, #0x0\n"
        "incd x9, ALL, MUL #4\n"
        "ld1b { z22.b }, p0/Z, [x27]\n"
        "incd x27, ALL, MUL #4\n"
        "ld1b { z17.b }, p0/Z, [x26]\n"
        "incd x26, ALL, MUL #4\n"
        "ld1b { z16.b }, p0/Z, [x25]\n"
        "incd x25, ALL, MUL #4\n"
        "ld1b { z20.b }, p0/Z, [x23]\n"
        "incd x23, ALL, MUL #4\n"
        "ld1b { z19.b }, p0/Z, [x22]\n"
        "zip1 z21.b, z18.b, z17.b\n"
        "incd x22, ALL, MUL #4\n"
        "ld1b { z18.b }, p0/Z, [x21]\n"
        "zip1 z17.b, z22.b, z16.b\n"
        "incd x21, ALL, MUL #4\n"
        "ld1b { z16.b }, p0/Z, [x20]\n"
        "incd x20, ALL, MUL #4\n"
        "zip1 z20.b, z20.b, z18.b\n"
        "zip1 z16.b, z19.b, z16.b\n"
        "zip1 z19.b, z21.b, z17.b\n"
        "zip2 z18.b, z21.b, z17.b\n"
        "zip1 z17.b, z20.b, z16.b\n"
        "zip2 z16.b, z20.b, z16.b\n"
        "st1b { z19.b }, p2, [x28]\n"
        "st1b { z18.b }, p2, [x28, #1, MUL VL]\n"
        "st1b { z17.b }, p2, [x28, #2, MUL VL]\n"
        "st1b { z16.b }, p2, [x28, #3, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 2b\n"
        "cmp %x[height], #0x8\n"
        "addvl %x[out], %x[out], #4\n"
        "bge 1b\n"
        "cbz %x[height], 8f\n"
        "4:"  // Main loop skip
        "5:"  // Tail row loop: Head
        "mov x9, %x[in]\n"
        "cntw x24, ALL, MUL #2\n"
        "add x27, x9, %x[in_stride]\n"
        "cmp %x[height], #0x3\n"
        "add x26, x27, %x[in_stride]\n"
        "csel x23, x24, XZR, GT\n"
        "add x25, x26, %x[in_stride]\n"
        "csel x26, x26, %x[pad_row], GE\n"
        "add %x[in], x25, %x[in_stride]\n"
        "csel x25, x25, %x[pad_row], GT\n"
        "csel x22, x24, XZR, GE\n"
        "cmp %x[height], #0x1\n"
        "mov x28, %x[out]\n"
        "csel x27, x27, %x[pad_row], GT\n"
        "csel x21, x24, XZR, GT\n"
        "sub %x[height], %x[height], #0x4\n"
        "mov x20, %x[width]\n"
        "6:"  // Tail row loop: Column loop
        "whilelt p0.b, XZR, x20\n"
        "decw x20, ALL, MUL #2\n"
        "ld1b { z18.b }, p0/Z, [x9]\n"
        "cmp x20, #0x0\n"
        "add x9, x9, x24\n"
        "ld1b { z19.b }, p0/Z, [x27]\n"
        "add x27, x27, x21\n"
        "ld1b { z17.b }, p0/Z, [x26]\n"
        "add x26, x26, x22\n"
        "ld1b { z16.b }, p0/Z, [x25]\n"
        "add x25, x25, x23\n"
        "zip1 z18.b, z18.b, z17.b\n"
        "zip1 z16.b, z19.b, z16.b\n"
        "zip1 z17.b, z18.b, z16.b\n"
        "zip2 z16.b, z18.b, z16.b\n"
        "st1b { z17.b }, p2, [x28]\n"
        "st1b { z16.b }, p2, [x28, #1, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 6b\n"
        "cmp %x[height], #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 5b\n"
        "8:"  // Done
        "mov x22, %x[out]\n"
        "mov x21, %x[width]\n"
        "dup z18.s, %w[scale_multiplier]\n"
        "cbz %x[scale], 10f\n"
        "9:"  // Scale: Full loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[scale]]\n"
        "cmp x21, #0x0\n"
        "ld1w { z16.s }, p0/Z, [%x[scale], #1, MUL VL]\n"
        "incb %x[scale], ALL, MUL #2\n"
        "fmul z17.s, z17.s, z18.s\n"
        "fmul z16.s, z16.s, z18.s\n"
        "st1w { z17.s }, p2, [x22]\n"
        "st1w { z16.s }, p2, [x22, #1, MUL VL]\n"
        "add x22, x22, %x[out_stride]\n"
        "bgt 9b\n"
        "10:"  // Scale: Done
        "cbz %x[width], 13f\n"
        "cbz x10, 13f\n"
        "dup z21.s, %w[lhs_zero_point]\n"
        "add x25, x10, #0x3\n"
        "cntw x24, ALL, MUL #2\n"
        "mov z20.b, #0x1\n"
        "lsr x25, x25, #0x2\n"
        "mov x23, %x[width]\n"
        "addvl x22, x11, #2\n"
        "neg z21.s, p2/M, z21.s\n"
        "11:"  // Bias: N loop
        "mov x21, x22\n"
        "mov x20, x25\n"
        "mov z19.s, #0x0\n"
        "mov z18.s, #0x0\n"
        "12:"  // Bias: K loop
        "ld1b { z17.b }, p2/Z, [x21]\n"
        "subs x20, x20, #0x1\n"
        "ld1b { z16.b }, p2/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        "sdot z19.s, z17.b, z20.b\n"
        "sdot z18.s, z16.b, z20.b\n"
        "bgt 12b\n"
        "mov x20, x23\n"
        "add x22, x22, %x[out_stride]\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[bias]]\n"
        "subs x23, x23, x24\n"
        "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
        "addvl %x[bias], %x[bias], #2\n"
        "mla z17.s, p2/M, z19.s, z21.s\n"
        "mla z16.s, p2/M, z18.s, z21.s\n"
        "st1w { z17.s }, p2, [x11]\n"
        "st1w { z16.s }, p2, [x11, #1, MUL VL]\n"
        "add x11, x11, %x[out_stride]\n"
        "bgt 11b\n"
        "13:"  // Bias: Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out), [scale] "+&r"(scale)
        : [in_stride] "r"(in_stride), [lhs_zero_point] "r"(lhs_zero_point), [out_stride] "r"(out_stride),
          [pad_row] "r"(pad_row), [scale_multiplier] "r"(scale_multiplier), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15", "x9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2",
          "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18",
          "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

#endif  // Architectural features check.
