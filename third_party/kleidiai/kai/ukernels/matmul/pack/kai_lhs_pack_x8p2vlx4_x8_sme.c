//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_lhs_pack_x8p2vlx4_x8_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

static inline size_t kai_get_m_step(void) {
    return (kai_mr * kai_get_sme_vector_length_u8()) / kai_kr;
}

size_t kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme(size_t mr) {
    KAI_ASSUME(mr == kai_mr * kai_get_sme_vector_length_u8() / kai_kr);
    KAI_UNUSED(mr);

    return (kai_mr * kai_get_sme_vector_length_u8()) / kai_kr;
}

size_t kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % (kai_get_m_step()) == 0);

    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t scaled_mr = kai_get_m_step();
    KAI_ASSUME(m_idx % scaled_mr == 0);
    KAI_ASSUME(mr == scaled_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return m_idx * kai_roundup(k, kai_kr) * sizeof(int8_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_get_m_step());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return (kai_roundup(m, kai_get_m_step()) * kai_roundup(k, kai_kr) * sizeof(int8_t));
}

void kai_run_lhs_pack_x8p2vlx4_x8_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_get_m_step());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    KAI_ASSUME(m_idx_start == 0);

    const size_t block_height = kai_get_m_step();
    const size_t width = k;
    const size_t row_offset = 0;

    const void* in[block_height];
    const uint8_t* lhs_ptr = lhs;
    uint8_t* lhs_packed_ptr = lhs_packed;

    for (size_t block_y = 0; block_y < m; block_y += block_height) {
        const size_t height = KAI_MIN(m - block_y, block_height);
        void* out = lhs_packed_ptr + block_y * kai_roundup(k, kai_kr) * sizeof(int8_t);

        for (size_t y = 0; y < height; y++) {
            in[y] = lhs_ptr + (block_y + y) * lhs_stride;
        }

        __asm__ __volatile__(
            ".inst 0xd503477f  // SMSTART ZA\n"
            "mov x23, %x[width]\n"
            "mov x21, %x[width]\n"
            "cntb x20\n"
            "incb x23\n"
            "sub x7, x20, #0x1\n"
            "cntw x8\n"
            "sub x23, x23, #0x1\n"
            "ands x7, x21, x7\n"
            "udiv x23, x23, x20\n"  // n_passes = ceildiv(width, VL<T>)
            "csel x7, x7, x20, NE\n"
            "lsl x22, %x[height], #0x1\n"  // height * 2
            "lsl x21, x8, #0x1\n"
            "sub x20, x23, #0x1\n"
            "add x7, x7, #0x3\n"
            "sub x17, x8, #0x2\n"
            "whilelt p9.b, XZR, x22\n"
            "whilelt p8.b, x21, x22\n"
            "mov x16, #0x0\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            "cntw x9, ALL, MUL #2\n"
            "cntw x28, ALL, MUL #3\n"
            "ldr x27, [x11, #0x0]\n"
            "lsr x20, x20, #0x1\n"  // n_loops = (n_passes - 1) / 2
            "and x26, x23, #0x1\n"  // odd_tail = bool(n_passes & 0x1)
            "ldr x25, [x10, #0x0]\n"
            "lsr x7, x7, #0x2\n"
            "ptrue p11.s\n"
            "ldr x24, [x11, #0x8]\n"
            "zip1 p10.b, p9.b, p8.b\n"
            "mov x23, %x[row_offset]\n"
            "ldr x21, [x10, #0x8]\n"
            "mov x22, %x[out]\n"
            "whilelt p9.b, x16, %x[width]\n"
            "whilelt p8.b, x16, %x[width]\n"
            "add x11, x11, #0x10\n"
            "add x10, x10, #0x10\n"
            "mov x12, #0x0\n"
            "cbz x17, 2f\n"
            "1:"  // K loop: Charge: Loop
            ".inst 0x25246143  // psel p3.b, p8.b/Z, p10.b[w12]\n"
            ".inst 0x252c6142  // psel p2.b, p8.b/Z, p10.b[w12, #1]\n"
            ".inst 0x25646141  // psel p1.b, p8.b/Z, p10.b[w12, #4]\n"
            ".inst 0x256c6140  // psel p0.b, p8.b/Z, p10.b[w12, #5]\n"
            ".inst 0xe0170f60  // ld1b { za0h.b[x12] }, p3/Z, [x27, x23]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0170b21  // ld1b { za0h.b[x12, #1] }, p2/Z, [x25, x23]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0170704  // ld1b { za0h.b[x12, #4] }, p1/Z, [x24, x23]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01702a5  // ld1b { za0h.b[x12, #5] }, p0/Z, [x21, x23]\n"
            "add x12, x12, #0x8\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "cmp x12, x17, LSL #2\n"
            "blt 1b\n"
            "2:"  // K loop: Charge: End
            ".inst 0x25246143  // psel p3.b, p8.b/Z, p10.b[w12]\n"
            ".inst 0x252c6142  // psel p2.b, p8.b/Z, p10.b[w12, #1]\n"
            ".inst 0x25646141  // psel p1.b, p8.b/Z, p10.b[w12, #4]\n"
            ".inst 0x256c6140  // psel p0.b, p8.b/Z, p10.b[w12, #5]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0170f60  // ld1b { za0h.b[x12] }, p3/Z, [x27, x23]\n"
            "ldr x27, [x11, #0x0]\n"
            "incb x16\n"
            ".inst 0xe0170b21  // ld1b { za0h.b[x12, #1] }, p2/Z, [x25, x23]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0170704  // ld1b { za0h.b[x12, #4] }, p1/Z, [x24, x23]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01702a5  // ld1b { za0h.b[x12, #5] }, p0/Z, [x21, x23]\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "incb x23\n"
            "cbz x20, 8f\n"
            "mov x20, x20\n"
            "3:"  // K loop: Main loop
            "whilelt p8.b, x16, %x[width]\n"
            "mov x15, #0x0\n"
            "mov x14, #0x0\n"
            "cbz x17, 5f\n"
            "4:"  // K loop: Main loop: First: Loop
            ".inst 0x25376143  // psel p3.b, p8.b/Z, p10.b[w15, #2]\n"
            ".inst 0x253f6142  // psel p2.b, p8.b/Z, p10.b[w15, #3]\n"
            ".inst 0x25776141  // psel p1.b, p8.b/Z, p10.b[w15, #6]\n"
            ".inst 0x257f6140  // psel p0.b, p8.b/Z, p10.b[w15, #7]\n"
            ".inst 0xe0176f62  // ld1b { za0h.b[x15, #2] }, p3/Z, [x27, x23]\n"
            ".inst 0x25266d23  // psel p3.b, p11.b/Z, p9.b[w14]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0176b23  // ld1b { za0h.b[x15, #3] }, p2/Z, [x25, x23]\n"
            ".inst 0x25266d22  // psel p2.b, p11.b/Z, p9.b[w14]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0176706  // ld1b { za0h.b[x15, #6] }, p1/Z, [x24, x23]\n"
            ".inst 0x252e6d21  // psel p1.b, p11.b/Z, p9.b[w14, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01762a7  // ld1b { za0h.b[x15, #7] }, p0/Z, [x21, x23]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0x252e6d20  // psel p0.b, p11.b/Z, p9.b[w14, #1]\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0bfcec0  // st1w { za0v.s[x14] }, p3/Z, [x22, XZR, LSL #2]\n"
            "add x15, x15, #0x8\n"
            ".inst 0xe0a8cac4  // st1w { za1v.s[x14] }, p2/Z, [x22, x8, LSL #2]\n"
            ".inst 0xe0a9c6c1  // st1w { za0v.s[x14, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bcc2c5  // st1w { za1v.s[x14, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "add x14, x14, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x14, x17\n"
            "blt 4b\n"
            "5:"  // K loop: Main loop: First: Tail
            ".inst 0x25376143  // psel p3.b, p8.b/Z, p10.b[w15, #2]\n"
            ".inst 0x253f6142  // psel p2.b, p8.b/Z, p10.b[w15, #3]\n"
            ".inst 0x25776141  // psel p1.b, p8.b/Z, p10.b[w15, #6]\n"
            ".inst 0x257f6140  // psel p0.b, p8.b/Z, p10.b[w15, #7]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0176f62  // ld1b { za0h.b[x15, #2] }, p3/Z, [x27, x23]\n"
            ".inst 0x25266d23  // psel p3.b, p11.b/Z, p9.b[w14]\n"
            "ldr x27, [x11, #0x0]\n"
            "mov x13, #0x0\n"
            ".inst 0xe0176b23  // ld1b { za0h.b[x15, #3] }, p2/Z, [x25, x23]\n"
            ".inst 0x25266d22  // psel p2.b, p11.b/Z, p9.b[w14]\n"
            "ldr x25, [x10, #0x0]\n"
            "mov x12, #0x0\n"
            ".inst 0xe0176706  // ld1b { za0h.b[x15, #6] }, p1/Z, [x24, x23]\n"
            ".inst 0x252e6d21  // psel p1.b, p11.b/Z, p9.b[w14, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01762a7  // ld1b { za0h.b[x15, #7] }, p0/Z, [x21, x23]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0x252e6d20  // psel p0.b, p11.b/Z, p9.b[w14, #1]\n"
            "whilelt p9.b, x16, %x[width]\n"
            ".inst 0xe0bfcec0  // st1w { za0v.s[x14] }, p3/Z, [x22, XZR, LSL #2]\n"
            "incb x16\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0a8cac4  // st1w { za1v.s[x14] }, p2/Z, [x22, x8, LSL #2]\n"
            "incb x23\n"
            "whilelt p8.b, x16, %x[width]\n"
            ".inst 0xe0a9c6c1  // st1w { za0v.s[x14, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bcc2c5  // st1w { za1v.s[x14, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "cbz x17, 7f\n"
            "6:"  // K loop: Main loop: Second: Loop
            ".inst 0x25256143  // psel p3.b, p8.b/Z, p10.b[w13]\n"
            ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
            ".inst 0x25656141  // psel p1.b, p8.b/Z, p10.b[w13, #4]\n"
            ".inst 0x256d6140  // psel p0.b, p8.b/Z, p10.b[w13, #5]\n"
            ".inst 0xe0172f60  // ld1b { za0h.b[x13] }, p3/Z, [x27, x23]\n"
            ".inst 0x25246d23  // psel p3.b, p11.b/Z, p9.b[w12]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0172b21  // ld1b { za0h.b[x13, #1] }, p2/Z, [x25, x23]\n"
            ".inst 0x25246d22  // psel p2.b, p11.b/Z, p9.b[w12]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0172704  // ld1b { za0h.b[x13, #4] }, p1/Z, [x24, x23]\n"
            ".inst 0x252c6d21  // psel p1.b, p11.b/Z, p9.b[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01722a5  // ld1b { za0h.b[x13, #5] }, p0/Z, [x21, x23]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0x252c6d20  // psel p0.b, p11.b/Z, p9.b[w12, #1]\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0bf8ec8  // st1w { za2v.s[x12] }, p3/Z, [x22, XZR, LSL #2]\n"
            "add x13, x13, #0x8\n"
            ".inst 0xe0a88acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bc82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "add x12, x12, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x12, x17\n"
            "blt 6b\n"
            "7:"  // K loop: Main loop: Second: Tail
            ".inst 0x25256143  // psel p3.b, p8.b/Z, p10.b[w13]\n"
            ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
            ".inst 0x25656141  // psel p1.b, p8.b/Z, p10.b[w13, #4]\n"
            ".inst 0x256d6140  // psel p0.b, p8.b/Z, p10.b[w13, #5]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x8, LSL #3\n"
            ".inst 0xe0172f60  // ld1b { za0h.b[x13] }, p3/Z, [x27, x23]\n"
            ".inst 0x25246d23  // psel p3.b, p11.b/Z, p9.b[w12]\n"
            "ldr x27, [x11, #0x0]\n"
            ".inst 0xe0172b21  // ld1b { za0h.b[x13, #1] }, p2/Z, [x25, x23]\n"
            ".inst 0x25246d22  // psel p2.b, p11.b/Z, p9.b[w12]\n"
            "ldr x25, [x10, #0x0]\n"
            ".inst 0xe0172704  // ld1b { za0h.b[x13, #4] }, p1/Z, [x24, x23]\n"
            ".inst 0x252c6d21  // psel p1.b, p11.b/Z, p9.b[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe01722a5  // ld1b { za0h.b[x13, #5] }, p0/Z, [x21, x23]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0x252c6d20  // psel p0.b, p11.b/Z, p9.b[w12, #1]\n"
            "whilelt p9.b, x16, %x[width]\n"
            ".inst 0xe0bf8ec8  // st1w { za2v.s[x12] }, p3/Z, [x22, XZR, LSL #2]\n"
            "subs x20, x20, #0x1\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0a88acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            "incb x16\n"
            "incb x23\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bc82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x28, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "bgt 3b\n"
            "8:"  // K loop: Tails
            "cbnz x26, 11f\n"
            "mov x11, %x[in]\n"
            "whilelt p8.b, x16, %x[width]\n"
            "mov x13, #0x0\n"
            "mov x12, #0x0\n"
            "9:"  // K loop: Tails: Even: First
            ".inst 0x25306d23  // psel p3.s, p11.s/Z, p9.s[w12]\n"
            ".inst 0x25306d22  // psel p2.s, p11.s/Z, p9.s[w12]\n"
            ".inst 0x25356141  // psel p1.b, p8.b/Z, p10.b[w13, #2]\n"
            ".inst 0x253d6140  // psel p0.b, p8.b/Z, p10.b[w13, #3]\n"
            ".inst 0xe0bf8ec0  // st1w { za0v.s[x12] }, p3/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a88ac4  // st1w { za1v.s[x12] }, p2/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "ldr x21, [x11, #0x0]\n"
            "cmp x12, x8\n"
            "ldr x20, [x11, x8, LSL #0x3]\n"
            "add x11, x11, #0x8\n"
            ".inst 0xe01726a2  // ld1b { za0h.b[x13, #2] }, p1/Z, [x21, x23]\n"
            ".inst 0xe0172283  // ld1b { za0h.b[x13, #3] }, p0/Z, [x20, x23]\n"
            "add x13, x13, #0x4\n"
            "blt 9b\n"
            "whilelt p9.b, x16, %x[width]\n"
            "whilelt p8.b, x16, %x[width]\n"
            "mov x20, #0x0\n"
            "mov x12, #0x0\n"
            "10:"  // K loop: Tails: Even: Second
            ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
            ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
            "add x20, x20, #0x4\n"
            ".inst 0xe0bf86c8  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a882cc  // st1w { za3v.s[x12] }, p0/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x7\n"
            "blt 10b\n"
            "whilelt p8.b, x16, %x[width]\n"
            "b 13f\n"
            "11:"  // K loop: Tails: Odd
            "mov x12, #0x0\n"
            "12:"  // K loop: Tails: Odd: Loop
            ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
            ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
            ".inst 0xe0bf86c0  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0a882c4  // st1w { za1v.s[x12] }, p0/Z, [x22, x8, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x7\n"
            "blt 12b\n"
            "13:"  // K loop: End
            "mov %x[out], x22\n"
            ".inst 0xd503467f  // SMSTOP\n"
            : [out] "+&r"(out)
            : [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset), [width] "r"(width)
            : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13",
              "p14", "p15", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21",
              "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
              "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
    }
}

#endif  // Architectural features check.
