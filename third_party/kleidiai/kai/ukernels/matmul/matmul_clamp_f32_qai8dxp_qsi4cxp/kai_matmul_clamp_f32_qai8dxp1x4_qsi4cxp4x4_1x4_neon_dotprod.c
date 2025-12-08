//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 4;
static const size_t kai_num_bytes_zp_lhs = 4;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_num_bytes_recip_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 4;
static const size_t kai_num_bytes_rsum_rhs = 4;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_k_multiple_of = 32;
static const size_t kai_bl = 32;

inline static size_t kai_get_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_get_k_roundedup(k);
    size_t lhs_packed_stride = kai_mr * ((k_internal * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs);
    // Since the LHS matrix is asymmetric with per-row quantization, we must include
    // the number of bytes to hold the zero point value
    lhs_packed_stride += kai_mr * kai_num_bytes_zp_lhs;

    return lhs_packed_stride;
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_get_k_roundedup(k);
    size_t rhs_packed_stride = kai_nr * (k_internal / kai_num_bytes_recip_qvalue_rhs);

    rhs_packed_stride += kai_nr * kai_num_bytes_multiplier_rhs;
    // Since the LHS matrix is quantized asymmetric with per-row quantization, we also include
    // the number of bytes for the reduction sum
    rhs_packed_stride += kai_nr * kai_num_bytes_rsum_rhs;
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += kai_nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_get_k_roundedup(k);
    size_t num_blocks = k_internal / kai_bl;
    float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x26, #0x20\n"
        "mov x20, #0x8\n"
        "movi v27.16b, #0xf0\n"
        "mov x25, %x[m]\n"
        "madd x26, %x[num_blocks], x26, x20\n"
        "1:"  // Row loop
        "mov x24, %x[rhs_packed]\n"
        "mov x23, %x[n]\n"
        "add x22, %x[dst], %x[dst_stride_row]\n"
        "2:"  // Column loop
        "mov x21, %x[lhs_packed]\n"
        "movi v26.4s, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "3:"  // Sub block loop
        "ldr q25, [x24, #0x0]\n"
        "ldr q24, [x21, #0x0]\n"
        "subs x20, x20, #0x1\n"
        "ldr q23, [x24, #0x10]\n"
        "ldr q22, [x24, #0x20]\n"
        "ldr q21, [x24, #0x30]\n"
        "ldr q20, [x21, #0x10]\n"
        "add x24, x24, #0x40\n"
        "add x21, x21, #0x20\n"
        "shl v19.16b, v25.16b, #0x4\n"
        "and v25.16b, v25.16b, v27.16b\n"
        "shl v18.16b, v23.16b, #0x4\n"
        "shl v17.16b, v22.16b, #0x4\n"
        "shl v16.16b, v21.16b, #0x4\n"
        "and v23.16b, v23.16b, v27.16b\n"
        ".inst 0x4f98e27a  // sdot v26.4s, v19.16b, v24.4b[0]\n"
        "and v22.16b, v22.16b, v27.16b\n"
        "and v21.16b, v21.16b, v27.16b\n"
        ".inst 0x4fb8e25a  // sdot v26.4s, v18.16b, v24.4b[1]\n"
        ".inst 0x4f98ea3a  // sdot v26.4s, v17.16b, v24.4b[2]\n"
        ".inst 0x4fb8ea1a  // sdot v26.4s, v16.16b, v24.4b[3]\n"
        ".inst 0x4f94e33a  // sdot v26.4s, v25.16b, v20.4b[0]\n"
        ".inst 0x4fb4e2fa  // sdot v26.4s, v23.16b, v20.4b[1]\n"
        ".inst 0x4f94eada  // sdot v26.4s, v22.16b, v20.4b[2]\n"
        ".inst 0x4fb4eaba  // sdot v26.4s, v21.16b, v20.4b[3]\n"
        "bgt 3b\n"
        "ldr q22, [x24, #0x0]\n"
        "ld1r { v21.4s }, [x21]\n"
        "add x21, x21, #0x4\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ld1r { v20.4s }, [x21]\n"
        "ldr q16, [x24, #0x10]\n"
        "cmp x23, #0x4\n"
        "ldr q19, [x24, #0x20]\n"
        "ld1r { v18.4s }, [%x[clamp_vals]]\n"
        "add x24, x24, #0x30\n"
        "ld1r { v17.4s }, [x20]\n"
        "mla v26.4s, v22.4s, v21.s[0]\n"
        "fmul v16.4s, v16.4s, v20.4s\n"
        "scvtf v26.4s, v26.4s\n"
        "fmul v16.4s, v26.4s, v16.4s\n"
        "fadd v16.4s, v16.4s, v19.4s\n"
        "fmax v16.4s, v16.4s, v18.4s\n"
        "fmin v16.4s, v16.4s, v17.4s\n"
        "blt 4f\n"
        "str q16, [%x[dst], #0x0]\n"
        "b 7f\n"
        "4:"  // Partial output
        "mov x20, %x[dst]\n"
        "tbz x23, #1, 5f\n"
        "st1 { v16.d }[0], [x20], #0x8\n"
        "tbz x23, #0, 6f\n"
        "st1 { v16.s }[2], [x20]\n"
        "b 6f\n"
        "5:"  // Output block 0: partial_1_0
        "st1 { v16.s }[0], [x20]\n"
        "6:"  // Output block 0: Done
        "7:"  // Stores done
        "subs x23, x23, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "subs x25, x25, #0x1\n"
        "add %x[lhs_packed], %x[lhs_packed], x26\n"
        "mov %x[dst], x22\n"
        "bgt 1b\n"
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "x20",
          "x21", "x22", "x23", "x24", "x25", "x26");
}

#endif  // Architectural features check.
