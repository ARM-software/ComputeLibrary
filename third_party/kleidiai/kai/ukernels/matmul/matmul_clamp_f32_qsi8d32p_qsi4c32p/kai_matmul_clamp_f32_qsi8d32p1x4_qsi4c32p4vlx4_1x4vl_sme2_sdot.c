//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !(defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME2))
#error This file must be compiled for AArch64, FEAT_SVE2 or FEAT_SME2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // Multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_bl = 32;

// Look-up table used for int4->int8 convert
static const int32_t lut[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

inline static size_t kai_get_num_bytes_per_block_lhs(size_t bl) {
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    size_t num_bytes_per_block_rhs = (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k, size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();

    size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m_idx, size_t k, size_t bl) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();

    KAI_ASSUME((m_idx % m_step) == 0);

    return (m_idx / mr) * kai_get_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t n_idx, size_t k, size_t bl) {
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();

    KAI_ASSUME((n_idx % nr) == 0);

    return (n_idx / n_step) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    KAI_ASSUME((m_idx % m_step) == 0);
    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    size_t bl,                        //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));
    KAI_ASSUME(m == 1);

    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(scalar_min);
    KAI_UNUSED(scalar_max);

    if (m == 0) {
        return;
    }

    const size_t lhs_packed_stride = kai_get_lhs_packed_stride(k, bl);
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride(k, bl);
    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();

    const uint16_t* lhs_scales = (const uint16_t*)((const int8_t*)lhs_packed + lhs_packed_stride -
                                                   (mr * num_blocks) * kai_num_bytes_multiplier_lhs);
    const uint16_t* rhs_scales = (const uint16_t*)((const uint8_t*)rhs_packed + rhs_packed_stride -
                                                   (nr * num_blocks) * kai_num_bytes_multiplier_rhs);

    kai_commit_za();

    __asm__ volatile(
        // Switch to streaming mode with ZA enabling
        " .inst 0xd503477f // smstart \n"

        " ptrue p2.b, all \n"
        " .inst 0x25607810 // ptrue pn8.h \n"

        " fmov  z28.s, #0.0 \n"

        // Initialize ZT0 (Lookup table)
        " mov x9, %[lut] \n"
        " .inst 0xe11f8120 // ldr zt0, [x9] \n"

        // Initialize the RHS packed and scale pointers
        " mov x0, %[rhs_packed] \n"
        " mov x1, %[rhs_scales] \n"

        // Initialize the DST pointer
        " mov x5, %[dst] \n"

        // Iterate over n (x0)
        // e.g. for(n_idx = 0; n_idx < n; n_idx+=n_step)
        " mov x4, #0\n"
        " mov x17, %[n] \n"
        " .inst 0x25b16491 // whilelt pn9.s, x4, x17, VLx4 \n"

        " b.none  5f // .LOOP_N_END%= \n"

        " 1: // .LOOP_N_START%=: \n"

        // Initialize the LHS packed and scale pointers
        " mov x2, %[lhs_packed] \n"
        " mov x3, %[lhs_scales] \n"

        // Initialize the 4xVL-32bit accumulators to zero
        " dup z24.s, #0 \n"
        " dup z25.s, #0 \n"
        " dup z26.s, #0 \n"
        " dup z27.s, #0 \n"

        // Initialize the vector selector for ZA array
        " mov w8, #0 \n"

        // Iterate over all K values
        // e.g. for(k_idx = 0; k_idx < k; k_idx += bl)
        " mov x6, #0 \n"
        " whilelt p1.s, x6, %[k] \n"
        " b.none 4f // .LOOP_K_END%= \n"

        " 2: // .LOOP_K_START%=: \n"
        // Zeroing of inner accumulation array
        " .inst 0xc00800ff // zero {za} \n"

        // Iterate over all values in the block
        // k_blk_idx = bl
        // e.g. while(k_blk_idx > 0) {... k_blk_idx -= 16}

        "mov x13, %[bl] \n"

        "3: // .LOOP_BL_START%=: \n"
        // Load the LHS (int8) quantized values
        // Load contiguous 16 bytes and replicate.
        // For GeMV, we do not interleave the LHS M rows.
        " ld1rqb  { z0.b }, p2/z , [x2] \n"
        " add x2, x2, #16 \n"

        // -- First half
        // Load the RHS (int4) quantized values
        " .inst 0xa040a00c // ld1h { z12.h - z15.h }, pn8/z, [x0] \n"

        // Increment the RHS pointer
        " addvl x0, x0, #4 \n"

        // Convert Int4 -> Int8
        " .inst 0xc08a4184 // luti4 { z4.b, z5.b },   zt0, z12[0] \n"
        " .inst 0xc08a41a6 // luti4 { z6.b, z7.b },   zt0, z13[0] \n"
        " .inst 0xc08a41c8 // luti4 { z8.b,  z9.b },  zt0, z14[0] \n"
        " .inst 0xc08a41ea // luti4 { z10.b, z11.b }, zt0, z15[0] \n"

        // SDOT indexed
        " .inst 0xc15090a0 // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[0] \n"
        " .inst 0xc1509520 // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[1] \n"

        // -- Second half

        // Load the RHS (int4) quantized values
        " .inst 0xa040a00c // ld1h { z12.h - z15.h }, pn8/z, [x0]\n"

        // Increment the RHS scale pointer
        " addvl x0, x0, #4 \n"

        // Convert Int4 -> Int8
        " .inst 0xc08a4184 // luti4 { z4.b, z5.b },   zt0, z12[0] \n"
        " .inst 0xc08a41a6 // luti4 { z6.b, z7.b },   zt0, z13[0] \n"
        " .inst 0xc08a41c8 // luti4 { z8.b,  z9.b },  zt0, z14[0] \n"
        " .inst 0xc08a41ea // luti4 { z10.b, z11.b }, zt0, z15[0] \n"

        // SDOT indexed
        " .inst 0xc15098a0 // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[2] \n"
        " .inst 0xc1509d20 // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[3] \n"

        // Decrement the block loop index
        "subs x13, x13, #16 \n"

        "b.gt 3b // .LOOP_BL_START%= \n"

        // === End of the block loop ===

        // Load Z registers with intermediate values from ZA array
        " .inst 0xc0060c10 // mova {z16.s - z19.s}, za.s[w8, 0, vgx4] \n"

        // Convert from int32 to float32
        " .inst 0xc132e210 // scvtf{z16.s - z19.s}, {z16.s - z19.s} \n"

        // Load 1 fp16 LHS scale scalar value and replicate for VL-16-bit
        " ld1rh z1.h, p2/z, [x3] \n"

        // Increment the LHS scale pointer by 2 (1 x sizeof(fp16))
        " add x3, x3, #2 \n"

        // Load 2xVL-16bit (fp16) RHS scales.
        // If VL=512bit, we load 64 fp16 values, which is equal to the number of output columns (n_step) processed
        " .inst 0xa0402024 // ld1h { z4.h - z5.h }, pn8/z, [x1]\n"

        // Increment the RHS scale pointer
        " addvl x1, x1, #2 \n"

        // Combine all the LHS and RHS scales
        " .inst 0xc165d082 //zip { z2.h-z3.h }, z4.h, z5.h\n"
        " movprfx z4, z28 \n"

        // Multiply two half floating-point vectors and store the result
        // to a floating-point 32-bit vector
        " fmlalb z4.s, z1.h, z2.h\n"
        " movprfx z5, z28 \n"
        " fmlalb z5.s, z1.h, z3.h\n"
        " movprfx z6, z28 \n"
        " fmlalt z6.s, z1.h, z2.h\n"
        " movprfx z7, z28 \n"
        " fmlalt z7.s, z1.h, z3.h\n"

        // Multiply the intermediate results by LHS_SCALE x RHS_SCALE
        // and store in the main floating-point accumulator
        " fmla z24.s, p2/m, z16.s, z4.s \n"
        " fmla z25.s, p2/m, z17.s, z5.s \n"
        " fmla z26.s, p2/m, z18.s, z6.s \n"
        " fmla z27.s, p2/m, z19.s, z7.s \n"

        // Increment the number of K values processed and
        // go to the next block
        " add x6, x6, %[bl] \n"
        " whilelt p1.s, x6, %[k] \n"
        " b.first 2b // .LOOP_K_START%= \n"
        " 4: //.LOOP_K_END%=: \n"

        // Store the results into memory
        " .inst 0xa060c4b8 // st1w { z24.s-z27.s }, pn9, [x5] \n"
        " incb  x4, all \n"
        " addvl x5, x5, #4 \n"

        // The new rhs_packed pointer is the current rhs_scales pointer
        // The new rhs_scales pointer is the current rhs_packed plus the rhs_packed_stride
        " mov x7, x0 \n"

        // Initialize the rhs_packed pointer
        " mov x0, x1 \n"

        // Initialize the rhs_scales pointer
        " add x1, x7, %[rhs_packed_stride] \n"

        " .inst 0x25b16491 // whilelt pn9.s, x4, %[n], VLx4 \n"

        " b.first  1b // .LOOP_N_START%= \n"

        " 5: // .LOOP_N_END%=: \n"

        // Exit streaming mode
        " .inst 0xd503467f //smstop \n"
        :
        : [lut] "r"(lut), [dst] "r"(dst), [rhs_packed] "r"(rhs_packed), [rhs_scales] "r"(rhs_scales),
          [lhs_packed] "r"(lhs_packed), [lhs_scales] "r"(lhs_scales), [rhs_packed_stride] "r"(rhs_packed_stride),
          [n] "r"((int64_t)n), [k] "r"(k), [bl] "i"(kai_bl)
        : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "z0",
          "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",
          "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "x0", "x1",
          "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x13", "x17", "memory", "cc");
}

#endif  // Architectural features check.
