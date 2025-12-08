//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 2;
static const size_t kai_kr = 2;
static const size_t kai_num_bytes_input = 4;
static const size_t kai_num_bytes_output = 2;
static const size_t kai_num_bytes_bias = 4;

size_t kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(void) {
    return kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t k) {
    return kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme() *
        (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme();
    return block_idx * kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(size_t n, size_t k) {
    const size_t n_rounded_up = kai_roundup(n, kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme());
    return kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(n_rounded_up, k);
}

void kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_get_n_step_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme());
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
    const float* pad_row = rhs;

    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(height);

    kai_commit_za();

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
        "cmp %x[height], #0x8\n"
        "incb %x[out], ALL, MUL #2\n"
        "blt 5f\n"
        "2:"  // Main row loop: Head
        "mov x10, %x[in]\n"
        "mov x9, %x[out]\n"
        "add x28, x10, %x[in_stride]\n"
        "sub %x[height], %x[height], #0x8\n"
        "add x27, x28, %x[in_stride]\n"
        "mov x26, %x[width]\n"
        "add x25, x27, %x[in_stride]\n"
        "add x24, x25, %x[in_stride]\n"
        "add x23, x24, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add %x[in], x21, %x[in_stride]\n"
        "3:"  // Main row loop: Column loop
        "mov x20, x26\n"
        "decw x26, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z19.s }, p1/Z, [x10]\n"
        "cmp x26, #0x0\n"
        "ld1w { z18.s }, p0/Z, [x10, #1, MUL VL]\n"
        "addvl x10, x10, #2\n"
        "ld1w { z17.s }, p1/Z, [x27]\n"
        "ld1w { z16.s }, p0/Z, [x27, #1, MUL VL]\n"
        ".inst 0x658aaa7b  // bfcvt z27.h, p2/M, z19.s\n"
        "addvl x27, x27, #2\n"
        "ld1w { z19.s }, p1/Z, [x24]\n"
        ".inst 0x658aaa5a  // bfcvt z26.h, p2/M, z18.s\n"
        "ld1w { z18.s }, p0/Z, [x24, #1, MUL VL]\n"
        ".inst 0x658aaa39  // bfcvt z25.h, p2/M, z17.s\n"
        "addvl x24, x24, #2\n"
        "ld1w { z17.s }, p1/Z, [x22]\n"
        ".inst 0x658aaa18  // bfcvt z24.h, p2/M, z16.s\n"
        "ld1w { z16.s }, p0/Z, [x22, #1, MUL VL]\n"
        ".inst 0x658aaa77  // bfcvt z23.h, p2/M, z19.s\n"
        "addvl x22, x22, #2\n"
        ".inst 0x658aaa56  // bfcvt z22.h, p2/M, z18.s\n"
        "ld1w { z19.s }, p1/Z, [x28]\n"
        ".inst 0x658aaa35  // bfcvt z21.h, p2/M, z17.s\n"
        "ld1w { z18.s }, p0/Z, [x28, #1, MUL VL]\n"
        "addvl x28, x28, #2\n"
        ".inst 0x658aaa14  // bfcvt z20.h, p2/M, z16.s\n"
        "ld1w { z17.s }, p1/Z, [x25]\n"
        "ld1w { z16.s }, p0/Z, [x25, #1, MUL VL]\n"
        "addvl x25, x25, #2\n"
        ".inst 0x648aaa7b  // bfcvtnt z27.h, p2/M, z19.s\n"
        "ld1w { z19.s }, p1/Z, [x23]\n"
        ".inst 0x648aaa5a  // bfcvtnt z26.h, p2/M, z18.s\n"
        "ld1w { z18.s }, p0/Z, [x23, #1, MUL VL]\n"
        "addvl x23, x23, #2\n"
        ".inst 0x648aaa39  // bfcvtnt z25.h, p2/M, z17.s\n"
        "ld1w { z17.s }, p1/Z, [x21]\n"
        ".inst 0x648aaa18  // bfcvtnt z24.h, p2/M, z16.s\n"
        "ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        ".inst 0x648aaa77  // bfcvtnt z23.h, p2/M, z19.s\n"
        "st1h { z27.h }, p2, [x9]\n"
        ".inst 0x648aaa56  // bfcvtnt z22.h, p2/M, z18.s\n"
        "st1h { z26.h }, p2, [x9, #1, MUL VL]\n"
        ".inst 0x648aaa35  // bfcvtnt z21.h, p2/M, z17.s\n"
        "st1h { z25.h }, p2, [x9, #2, MUL VL]\n"
        ".inst 0x648aaa14  // bfcvtnt z20.h, p2/M, z16.s\n"
        "st1h { z24.h }, p2, [x9, #3, MUL VL]\n"
        "st1h { z23.h }, p2, [x9, #4, MUL VL]\n"
        "st1h { z22.h }, p2, [x9, #5, MUL VL]\n"
        "st1h { z21.h }, p2, [x9, #6, MUL VL]\n"
        "st1h { z20.h }, p2, [x9, #7, MUL VL]\n"
        "add x9, x9, %x[out_stride]\n"
        "bgt 3b\n"
        "cmp %x[height], #0x8\n"
        "addvl %x[out], %x[out], #8\n"
        "bge 2b\n"
        "cbz %x[height], 9f\n"
        "5:"  // Main loop skip
        "6:"  // Tail row loop: Head
        "mov x10, %x[in]\n"
        "cmp %x[height], #0x1\n"
        "add x28, x10, %x[in_stride]\n"
        "mov x9, %x[out]\n"
        "add %x[in], x28, %x[in_stride]\n"
        "csel x28, x28, %x[pad_row], GT\n"
        "sub %x[height], %x[height], #0x2\n"
        "mov x21, %x[width]\n"
        "7:"  // Tail row loop: Column loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [x10]\n"
        "cmp x21, #0x0\n"
        "ld1w { z16.s }, p0/Z, [x10, #1, MUL VL]\n"
        "addvl x10, x10, #2\n"
        "ld1w { z19.s }, p1/Z, [x28]\n"
        ".inst 0x658aaa32  // bfcvt z18.h, p2/M, z17.s\n"
        "ld1w { z17.s }, p0/Z, [x28, #1, MUL VL]\n"
        "addvl x28, x28, #2\n"
        ".inst 0x658aaa10  // bfcvt z16.h, p2/M, z16.s\n"
        ".inst 0x648aaa72  // bfcvtnt z18.h, p2/M, z19.s\n"
        ".inst 0x648aaa30  // bfcvtnt z16.h, p2/M, z17.s\n"
        "st1h { z18.h }, p2, [x9]\n"
        "st1h { z16.h }, p2, [x9, #1, MUL VL]\n"
        "add x9, x9, %x[out_stride]\n"
        "bgt 7b\n"
        "cmp %x[height], #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 6b\n"
        "9:"  // Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out)
        : [in_stride] "r"(in_stride), [out_stride] "r"(out_stride), [pad_row] "r"(pad_row), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1", "z10",
          "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25",
          "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
