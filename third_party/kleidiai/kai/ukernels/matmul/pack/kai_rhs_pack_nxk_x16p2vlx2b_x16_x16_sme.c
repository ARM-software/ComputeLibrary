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
static const size_t kai_kr = 2;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_data = 2;
static const size_t kai_num_bytes_bias = 2;

static size_t kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(void) {
    const size_t block_height = kai_nr * kai_get_sme_vector_length_u16() / kai_kr;
    return block_height;
}

size_t kai_get_n_step_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(void) {
    return kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme();
}

size_t kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t rhs_stride) {
    KAI_ASSUME(n_idx % kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme() == 0);

    return n_idx * rhs_stride;
}

size_t kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme() == 0);

    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t k) {
    return kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_data;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme() == 0);

    return n_idx * kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n, size_t k) {
    return kai_roundup(n, kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme()) *
        kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(k);
}

void kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    const size_t block_height = kai_get_block_height_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme();
    const size_t width = k;
    const size_t row_offset = 0;

    const uint8_t* in[block_height];

    for (size_t block_y = 0; block_y < n; block_y += block_height) {
        const size_t height = KAI_MIN(n - block_y, block_height);
        uint8_t* out =
            (uint8_t*)rhs_packed + block_y * (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_data);

        for (size_t y = 0; y < height; y++) {
            in[y] = (const uint8_t*)rhs + (block_y + y) * rhs_stride;
        }

        __asm__ __volatile__(
            ".inst 0xd503477f  // SMSTART ZA\n"
            "ptrue p1.b\n"
            "cbz %x[bias], 1f\n"
            "whilelt p0.h, XZR, %x[height]\n"
            "ld1h { z16.h }, p0/Z, [%x[bias]]\n"
            "st1h { z16.h }, p1, [%x[out]]\n"
            "addvl %x[out], %x[out], #1\n"
            "1:"  // Bias: Done
            "cnth x21\n"
            "mov x22, %x[width]\n"
            "inch x22\n"
            "mov x20, %x[width]\n"
            "sub x7, x21, #0x1\n"
            "sub x22, x22, #0x1\n"
            "ands x7, x20, x7\n"
            "cntw x8\n"
            "udiv x22, x22, x21\n"  // n_passes = ceildiv(width, VL<T>)
            "csel x7, x7, x21, NE\n"
            "sub x13, x22, #0x1\n"
            "add x7, x7, #0x1\n"
            "sub x17, x8, #0x2\n"
            "lsl x21, %x[height], #0x1\n"  // height * 2
            "lsl x20, x8, #0x1\n"
            "mov x16, #0x0\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            "cntw x9, ALL, MUL #2\n"
            "cntw x28, ALL, MUL #3\n"
            "ldr x27, [x11, #0x0]\n"
            "lsr x13, x13, #0x1\n"  // n_loops = (n_passes - 1) / 2
            "and x26, x22, #0x1\n"  // odd_tail = bool(n_passes & 0x1)
            "ldr x25, [x10, #0x0]\n"
            "lsr x7, x7, #0x1\n"
            "ptrue p12.s\n"
            "ldr x24, [x11, #0x8]\n"
            "whilelt p11.h, XZR, x21\n"
            "whilelt p10.h, x20, x21\n"
            "ldr x21, [x10, #0x8]\n"
            "mov x23, %x[row_offset]\n"
            "mov x22, %x[out]\n"
            "whilelt p9.h, x16, %x[width]\n"
            "whilelt p8.h, x16, %x[width]\n"
            "add x11, x11, #0x10\n"
            "add x10, x10, #0x10\n"
            "mov x12, #0x0\n"
            "cbz x17, 3f\n"
            "2:"  // K loop: Charge: Loop
            ".inst 0x25286163  // psel p3.h, p8.h/Z, p11.h[w12]\n"
            ".inst 0x25286142  // psel p2.h, p8.h/Z, p10.h[w12]\n"
            ".inst 0x25686161  // psel p1.h, p8.h/Z, p11.h[w12, #2]\n"
            ".inst 0x25686140  // psel p0.h, p8.h/Z, p10.h[w12, #2]\n"
            ".inst 0xe0570f60  // ld1h { za0h.h[x12] }, p3/Z, [x27, x23, LSL #1]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0570b28  // ld1h { za1h.h[x12] }, p2/Z, [x25, x23, LSL #1]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0570702  // ld1h { za0h.h[x12, #2] }, p1/Z, [x24, x23, LSL #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe05702aa  // ld1h { za1h.h[x12, #2] }, p0/Z, [x21, x23, LSL #1]\n"
            "add x12, x12, #0x4\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "cmp x12, x17, LSL #1\n"
            "blt 2b\n"
            "3:"  // K loop: Charge: End
            ".inst 0x25286163  // psel p3.h, p8.h/Z, p11.h[w12]\n"
            ".inst 0x25286142  // psel p2.h, p8.h/Z, p10.h[w12]\n"
            ".inst 0x25686161  // psel p1.h, p8.h/Z, p11.h[w12, #2]\n"
            ".inst 0x25686140  // psel p0.h, p8.h/Z, p10.h[w12, #2]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0570f60  // ld1h { za0h.h[x12] }, p3/Z, [x27, x23, LSL #1]\n"
            "ldr x27, [x11, #0x0]\n"
            "inch x16\n"
            ".inst 0xe0570b28  // ld1h { za1h.h[x12] }, p2/Z, [x25, x23, LSL #1]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0570702  // ld1h { za0h.h[x12, #2] }, p1/Z, [x24, x23, LSL #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe05702aa  // ld1h { za1h.h[x12, #2] }, p0/Z, [x21, x23, LSL #1]\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "inch x23\n"
            "cbz x13, 9f\n"
            "mov x20, x13\n"
            "4:"  // K loop: Main loop
            "whilelt p8.h, x16, %x[width]\n"
            "mov x15, #0x0\n"
            "mov x14, #0x0\n"
            "cbz x17, 6f\n"
            "5:"  // K loop: Main loop: First: Loop
            ".inst 0x253b6160  // psel p0.h, p8.h/Z, p11.h[w15, #1]\n"
            ".inst 0x253b6142  // psel p2.h, p8.h/Z, p10.h[w15, #1]\n"
            ".inst 0x257b6161  // psel p1.h, p8.h/Z, p11.h[w15, #3]\n"
            ".inst 0x257b6143  // psel p3.h, p8.h/Z, p10.h[w15, #3]\n"
            ".inst 0xe0576361  // ld1h { za0h.h[x15, #1] }, p0/Z, [x27, x23, LSL #1]\n"
            ".inst 0x252a7120  // psel p0.h, p12.h/Z, p9.h[w14]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0576b29  // ld1h { za1h.h[x15, #1] }, p2/Z, [x25, x23, LSL #1]\n"
            ".inst 0x252a7122  // psel p2.h, p12.h/Z, p9.h[w14]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0576703  // ld1h { za0h.h[x15, #3] }, p1/Z, [x24, x23, LSL #1]\n"
            ".inst 0x253a7121  // psel p1.h, p12.h/Z, p9.h[w14, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0576eab  // ld1h { za1h.h[x15, #3] }, p3/Z, [x21, x23, LSL #1]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bfc2c0  // st1w { za0v.s[x14] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x253a7120  // psel p0.h, p12.h/Z, p9.h[w14, #1]\n"
            ".inst 0xe0a8cac4  // st1w { za1v.s[x14] }, p2/Z, [x22, x8, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "add x15, x15, #0x4\n"
            ".inst 0xe0a9c6c1  // st1w { za0v.s[x14, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bcc2c5  // st1w { za1v.s[x14, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "add x14, x14, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x14, x17\n"
            "blt 5b\n"
            "6:"  // K loop: Main loop: First: Tail
            ".inst 0x253b6160  // psel p0.h, p8.h/Z, p11.h[w15, #1]\n"
            ".inst 0x253b6142  // psel p2.h, p8.h/Z, p10.h[w15, #1]\n"
            ".inst 0x257b6161  // psel p1.h, p8.h/Z, p11.h[w15, #3]\n"
            ".inst 0x257b6143  // psel p3.h, p8.h/Z, p10.h[w15, #3]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0576361  // ld1h { za0h.h[x15, #1] }, p0/Z, [x27, x23, LSL #1]\n"
            ".inst 0x252a7120  // psel p0.h, p12.h/Z, p9.h[w14]\n"
            "ldr x27, [x11, #0x0]\n"
            "mov x13, #0x0\n"
            ".inst 0xe0576b29  // ld1h { za1h.h[x15, #1] }, p2/Z, [x25, x23, LSL #1]\n"
            ".inst 0x252a7122  // psel p2.h, p12.h/Z, p9.h[w14]\n"
            "ldr x25, [x10, #0x0]\n"
            "mov x12, #0x0\n"
            ".inst 0xe0576703  // ld1h { za0h.h[x15, #3] }, p1/Z, [x24, x23, LSL #1]\n"
            ".inst 0x253a7121  // psel p1.h, p12.h/Z, p9.h[w14, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0576eab  // ld1h { za1h.h[x15, #3] }, p3/Z, [x21, x23, LSL #1]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bfc2c0  // st1w { za0v.s[x14] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x253a7120  // psel p0.h, p12.h/Z, p9.h[w14, #1]\n"
            ".inst 0xe0a8cac4  // st1w { za1v.s[x14] }, p2/Z, [x22, x8, LSL #2]\n"
            "whilelt p9.h, x16, %x[width]\n"
            "inch x16\n"
            ".inst 0xe0a9c6c1  // st1w { za0v.s[x14, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "inch x23\n"
            ".inst 0xe0bcc2c5  // st1w { za1v.s[x14, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "whilelt p8.h, x16, %x[width]\n"
            "cbz x17, 8f\n"
            "7:"  // K loop: Main loop: Second: Loop
            ".inst 0x25296160  // psel p0.h, p8.h/Z, p11.h[w13]\n"
            ".inst 0x25296142  // psel p2.h, p8.h/Z, p10.h[w13]\n"
            ".inst 0x25696161  // psel p1.h, p8.h/Z, p11.h[w13, #2]\n"
            ".inst 0x25696143  // psel p3.h, p8.h/Z, p10.h[w13, #2]\n"
            ".inst 0xe0572360  // ld1h { za0h.h[x13] }, p0/Z, [x27, x23, LSL #1]\n"
            ".inst 0x25287120  // psel p0.h, p12.h/Z, p9.h[w12]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0572b28  // ld1h { za1h.h[x13] }, p2/Z, [x25, x23, LSL #1]\n"
            ".inst 0x25287122  // psel p2.h, p12.h/Z, p9.h[w12]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0572702  // ld1h { za0h.h[x13, #2] }, p1/Z, [x24, x23, LSL #1]\n"
            ".inst 0x25387121  // psel p1.h, p12.h/Z, p9.h[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0572eaa  // ld1h { za1h.h[x13, #2] }, p3/Z, [x21, x23, LSL #1]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bf82c8  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25387120  // psel p0.h, p12.h/Z, p9.h[w12, #1]\n"
            ".inst 0xe0a88acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "add x13, x13, #0x4\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bc82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "add x12, x12, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x12, x17\n"
            "blt 7b\n"
            "8:"  // K loop: Main loop: Second: Tail
            ".inst 0x25296160  // psel p0.h, p8.h/Z, p11.h[w13]\n"
            ".inst 0x25296142  // psel p2.h, p8.h/Z, p10.h[w13]\n"
            ".inst 0x25696161  // psel p1.h, p8.h/Z, p11.h[w13, #2]\n"
            ".inst 0x25696143  // psel p3.h, p8.h/Z, p10.h[w13, #2]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0572360  // ld1h { za0h.h[x13] }, p0/Z, [x27, x23, LSL #1]\n"
            ".inst 0x25287120  // psel p0.h, p12.h/Z, p9.h[w12]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0572b28  // ld1h { za1h.h[x13] }, p2/Z, [x25, x23, LSL #1]\n"
            ".inst 0x25287122  // psel p2.h, p12.h/Z, p9.h[w12]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0572702  // ld1h { za0h.h[x13, #2] }, p1/Z, [x24, x23, LSL #1]\n"
            ".inst 0x25387121  // psel p1.h, p12.h/Z, p9.h[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0572eaa  // ld1h { za1h.h[x13, #2] }, p3/Z, [x21, x23, LSL #1]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bf82c8  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25387120  // psel p0.h, p12.h/Z, p9.h[w12, #1]\n"
            ".inst 0xe0a88acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            "whilelt p9.h, x16, %x[width]\n"
            "subs x20, x20, #0x1\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "inch x16\n"
            ".inst 0xe0bc82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "inch x23\n"
            "bgt 4b\n"
            "9:"  // K loop: Tails
            "cbnz x26, 12f\n"
            "mov x11, %x[in]\n"
            "whilelt p8.h, x16, %x[width]\n"
            "mov x13, #0x0\n"
            "mov x12, #0x0\n"
            "10:"  // K loop: Tails: Even: First
            ".inst 0x25307123  // psel p3.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307122  // psel p2.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25396161  // psel p1.h, p8.h/Z, p11.h[w13, #1]\n"
            ".inst 0x25396140  // psel p0.h, p8.h/Z, p10.h[w13, #1]\n"
            ".inst 0xe0bf8ec0  // st1w { za0v.s[x12] }, p3/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a88ac4  // st1w { za1v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "ldr x21, [x11, #0x0]\n"
            "cmp x12, x8\n"
            "ldr x20, [x11, x8, LSL #0x3]\n"
            "add x11, x11, #0x8\n"
            ".inst 0xe05726a1  // ld1h { za0h.h[x13, #1] }, p1/Z, [x21, x23, LSL #1]\n"
            ".inst 0xe0572289  // ld1h { za1h.h[x13, #1] }, p0/Z, [x20, x23, LSL #1]\n"
            "add x13, x13, #0x2\n"
            "blt 10b\n"
            "whilelt p9.h, x16, %x[width]\n"
            "whilelt p8.h, x16, %x[width]\n"
            "mov x20, #0x0\n"
            "mov x12, #0x0\n"
            "11:"  // K loop: Tails: Even: Second
            ".inst 0x25307121  // psel p1.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            "add x20, x20, #0x2\n"
            ".inst 0xe0bf86c8  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a882cc  // st1w { za3v.s[x12] }, p0/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x7\n"
            "blt 11b\n"
            "whilelt p8.h, x16, %x[width]\n"
            "b 14f\n"
            "12:"  // K loop: Tails: Odd
            "mov x12, #0x0\n"
            "13:"  // K loop: Tails: Odd: Loop
            ".inst 0x25307121  // psel p1.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0xe0bf86c0  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a882c4  // st1w { za1v.s[x12] }, p0/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x7\n"
            "blt 13b\n"
            "14:"  // K loop: End
            "mov %x[out], x22\n"
            ".inst 0xd503467f  // SMSTOP\n"
            : [out] "+&r"(out)
            : [bias] "r"(bias), [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset), [width] "r"(width)
            : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
              "p8", "p9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24",
              "x25", "x26", "x27", "x28", "x7", "x8", "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
              "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3",
              "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");

        bias = (const uint8_t*)bias + height * kai_num_bytes_bias;
    }
}

#endif  // Architectural features check.
