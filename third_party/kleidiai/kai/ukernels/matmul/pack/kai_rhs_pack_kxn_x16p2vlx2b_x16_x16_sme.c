//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 2;
static const size_t kai_kr = 2;
static const size_t kai_num_bytes_input = 2;
static const size_t kai_num_bytes_output = 2;
static const size_t kai_num_bytes_bias = 2;

size_t kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(void) {
    return kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t k) {
    return kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme() *
        (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme();
    return block_idx * kai_get_rhs_packed_stride_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n, size_t k) {
    const size_t n_nr_blocks = kai_roundup(n, kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme());
    return kai_get_rhs_packed_offset_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(n_nr_blocks, k);
}

void kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme());
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
    const uint16_t* pad_row = rhs;

    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(height);

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x21, %x[out]\n"
        "mov x20, %x[width]\n"
        "ptrue p1.b\n"
        "1:"  // Bias: Full loop
        "whilelt p0.h, XZR, x20\n"
        "dech x20\n"
        "cmp x20, #0x0\n"
        "ld1h { z16.h }, p0/Z, [%x[bias]]\n"
        "incb %x[bias]\n"
        "st1h { z16.h }, p1, [x21]\n"
        "add x21, x21, %x[out_stride]\n"
        "bgt 1b\n"
        "cmp %x[height], #0x8\n"
        "incb %x[out]\n"
        "blt 5f\n"
        "2:"  // Main row loop: Head
        "mov x9, %x[in]\n"
        "mov x28, %x[out]\n"
        "add x27, x9, %x[in_stride]\n"
        "sub %x[height], %x[height], #0x8\n"
        "add x26, x27, %x[in_stride]\n"
        "mov x25, %x[width]\n"
        "add x24, x26, %x[in_stride]\n"
        "add x23, x24, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "add %x[in], x20, %x[in_stride]\n"
        "3:"  // Main row loop: Column loop
        "whilelt p0.h, XZR, x25\n"
        "decw x25, ALL, MUL #2\n"
        "ld1h { z20.h }, p0/Z, [x9]\n"
        "cmp x25, #0x0\n"
        "addvl x9, x9, #1\n"
        "ld1h { z17.h }, p0/Z, [x27]\n"
        "addvl x27, x27, #1\n"
        "ld1h { z19.h }, p0/Z, [x26]\n"
        "addvl x26, x26, #1\n"
        "ld1h { z16.h }, p0/Z, [x24]\n"
        "addvl x24, x24, #1\n"
        "ld1h { z18.h }, p0/Z, [x23]\n"
        "addvl x23, x23, #1\n"
        "zip1 z24.h, z20.h, z17.h\n"
        "zip2 z23.h, z20.h, z17.h\n"
        "ld1h { z17.h }, p0/Z, [x22]\n"
        "addvl x22, x22, #1\n"
        "ld1h { z22.h }, p0/Z, [x21]\n"
        "addvl x21, x21, #1\n"
        "zip1 z21.h, z19.h, z16.h\n"
        "zip2 z20.h, z19.h, z16.h\n"
        "ld1h { z16.h }, p0/Z, [x20]\n"
        "addvl x20, x20, #1\n"
        "zip1 z19.h, z18.h, z17.h\n"
        "zip2 z18.h, z18.h, z17.h\n"
        "st1h { z24.h }, p1, [x28]\n"
        "st1h { z23.h }, p1, [x28, #1, MUL VL]\n"
        "zip1 z17.h, z22.h, z16.h\n"
        "zip2 z16.h, z22.h, z16.h\n"
        "st1h { z21.h }, p1, [x28, #2, MUL VL]\n"
        "st1h { z20.h }, p1, [x28, #3, MUL VL]\n"
        "st1h { z19.h }, p1, [x28, #4, MUL VL]\n"
        "st1h { z18.h }, p1, [x28, #5, MUL VL]\n"
        "st1h { z17.h }, p1, [x28, #6, MUL VL]\n"
        "st1h { z16.h }, p1, [x28, #7, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 3b\n"
        "cmp %x[height], #0x8\n"
        "addvl %x[out], %x[out], #8\n"
        "bge 2b\n"
        "cbz %x[height], 9f\n"
        "5:"  // Main loop skip
        "6:"  // Tail row loop: Head
        "mov x9, %x[in]\n"
        "cmp %x[height], #0x1\n"
        "add x27, x9, %x[in_stride]\n"
        "mov x28, %x[out]\n"
        "add %x[in], x27, %x[in_stride]\n"
        "csel x27, x27, %x[pad_row], GT\n"
        "sub %x[height], %x[height], #0x2\n"
        "mov x20, %x[width]\n"
        "7:"  // Tail row loop: Column loop
        "whilelt p0.h, XZR, x20\n"
        "decw x20, ALL, MUL #2\n"
        "ld1h { z18.h }, p0/Z, [x9]\n"
        "cmp x20, #0x0\n"
        "addvl x9, x9, #1\n"
        "ld1h { z16.h }, p0/Z, [x27]\n"
        "addvl x27, x27, #1\n"
        "zip1 z17.h, z18.h, z16.h\n"
        "zip2 z16.h, z18.h, z16.h\n"
        "st1h { z17.h }, p1, [x28]\n"
        "st1h { z16.h }, p1, [x28, #1, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 7b\n"
        "cmp %x[height], #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 6b\n"
        "9:"  // Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out)
        : [in_stride] "r"(in_stride), [out_stride] "r"(out_stride), [pad_row] "r"(pad_row), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1", "z10", "z11",
          "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
          "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
