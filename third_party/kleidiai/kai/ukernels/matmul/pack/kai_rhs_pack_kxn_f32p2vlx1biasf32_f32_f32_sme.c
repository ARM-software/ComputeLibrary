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

static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_data_size_in_bytes = sizeof(uint32_t);
static const size_t kai_bias_size_in_bytes = sizeof(uint32_t);

size_t kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % (kai_nr * kai_get_sme_vector_length_u32()) == 0);

    return n_idx * kai_data_size_in_bytes;
}

size_t kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx) {
    return n_idx * kai_bias_size_in_bytes;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(size_t k) {
    return kai_nr * kai_get_sme_vector_length_u32() * (kai_bias_size_in_bytes + k * kai_data_size_in_bytes);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % (kai_nr * kai_get_sme_vector_length_u32()) == 0);

    return n_idx * (kai_bias_size_in_bytes + k * kai_data_size_in_bytes);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(size_t n, size_t k) {
    return kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        kai_roundup(n, kai_nr * kai_get_sme_vector_length_u32()), k);
}

void kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_nr * kai_get_sme_vector_length_u32());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(height);

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x22, %x[out]\n"
        "mov x21, %x[width]\n"
        "ptrue p2.b\n"
        "1:"  // Bias: Full loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x21, #0x0\n"
        "ld1w { z17.s }, p1/Z, [%x[bias]]\n"
        "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
        "incb %x[bias], ALL, MUL #2\n"
        "st1w { z17.s }, p2, [x22]\n"
        "st1w { z16.s }, p2, [x22, #1, MUL VL]\n"
        "add x22, x22, %x[out_stride]\n"
        "bgt 1b\n"
        "cmp %x[height], #0x4\n"
        "incb %x[out], ALL, MUL #2\n"
        "blt 5f\n"
        "2:"  // Main row loop: Head
        "mov x26, %x[in]\n"
        "mov x25, %x[out]\n"
        "add x24, x26, %x[in_stride]\n"
        "sub %x[height], %x[height], #0x4\n"
        "add x23, x24, %x[in_stride]\n"
        "mov x22, %x[width]\n"
        "add x21, x23, %x[in_stride]\n"
        "add %x[in], x21, %x[in_stride]\n"
        "3:"  // Main row loop: Column loop
        "mov x20, x22\n"
        "decw x22, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x22, #0x0\n"
        "ld1w { z23.s }, p1/Z, [x26]\n"
        "ld1w { z22.s }, p0/Z, [x26, #1, MUL VL]\n"
        "addvl x26, x26, #2\n"
        "ld1w { z21.s }, p1/Z, [x24]\n"
        "ld1w { z20.s }, p0/Z, [x24, #1, MUL VL]\n"
        "addvl x24, x24, #2\n"
        "ld1w { z19.s }, p1/Z, [x23]\n"
        "ld1w { z18.s }, p0/Z, [x23, #1, MUL VL]\n"
        "addvl x23, x23, #2\n"
        "ld1w { z17.s }, p1/Z, [x21]\n"
        "ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        "st1w { z23.s }, p2, [x25]\n"
        "st1w { z22.s }, p2, [x25, #1, MUL VL]\n"
        "st1w { z21.s }, p2, [x25, #2, MUL VL]\n"
        "st1w { z20.s }, p2, [x25, #3, MUL VL]\n"
        "st1w { z19.s }, p2, [x25, #4, MUL VL]\n"
        "st1w { z18.s }, p2, [x25, #5, MUL VL]\n"
        "st1w { z17.s }, p2, [x25, #6, MUL VL]\n"
        "st1w { z16.s }, p2, [x25, #7, MUL VL]\n"
        "add x25, x25, %x[out_stride]\n"
        "bgt 3b\n"
        "cmp %x[height], #0x4\n"
        "addvl %x[out], %x[out], #8\n"
        "bge 2b\n"
        "cbz %x[height], 9f\n"
        "5:"  // Main loop skip
        "6:"  // Tail row loop: Head
        "mov x26, %x[in]\n"
        "mov x25, %x[out]\n"
        "add %x[in], x26, %x[in_stride]\n"
        "sub %x[height], %x[height], #0x1\n"
        "mov x21, %x[width]\n"
        "7:"  // Tail row loop: Column loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x21, #0x0\n"
        "ld1w { z17.s }, p1/Z, [x26]\n"
        "ld1w { z16.s }, p0/Z, [x26, #1, MUL VL]\n"
        "addvl x26, x26, #2\n"
        "st1w { z17.s }, p2, [x25]\n"
        "st1w { z16.s }, p2, [x25, #1, MUL VL]\n"
        "add x25, x25, %x[out_stride]\n"
        "bgt 7b\n"
        "cmp %x[height], #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 6b\n"
        "9:"  // Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out)
        : [in_stride] "r"(in_stride), [out_stride] "r"(out_stride), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
          "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
          "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

#endif  // Architectural features check.
