//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__aarch64__)
#error "Dotprod and I8mm extensions required to compile this example"
#else
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

// Include micro-kernel variants
#include "kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)

// Micro-kernel interface
struct kai_matmul_ukernel_f32_qa8d32p_qs4c32p {
    kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;
    std::string name = {};
};

kai_matmul_ukernel_f32_qa8d32p_qs4c32p ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm"},
};

// Number of micro-kernel variants stored in the array
const size_t num_ukernel_variants = sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);

static void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (int i = 0; i < num_rows * num_cols; i++) {
        dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
}

static inline size_t num_blocks_per_row(size_t k, size_t bl) {
    return k / bl;
}

static inline size_t num_bytes_per_block_qs8c32(size_t bl) {
    return bl + sizeof(int16_t);
}

static inline size_t num_bytes_per_block_qs4c32(size_t bl) {
    return (bl / 2) + sizeof(int16_t);
}

static void quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block = num_bytes_per_block_qs4c32(bl);
    const size_t dst_stride = num_blocks_row * num_bytes_block;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        uint8_t* dst_ptr = (uint8_t*)rhs_qs4c32 + row_idx * dst_stride;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const float src0_0 = src_ptr[block_idx * bl + b];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -8.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale at the beginning of the block
            *((uint16_t*)dst_ptr) = kai_cast_f16_f32(scale);
            dst_ptr += sizeof(uint16_t);

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                for (size_t i = 0; i < block_size / 2; ++i) {
                    const size_t src_base_addr = block_idx * bl + i + subblock_idx * block_size;
                    float v0_f32 = src_ptr[src_base_addr];
                    float v1_f32 = src_ptr[src_base_addr + block_size / 2];

                    v0_f32 *= recip_scale;
                    v1_f32 *= recip_scale;

                    const uint8_t v0_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v0_f32 + 8.5f));
                    const uint8_t v1_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v1_f32 + 8.5f));

                    const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

                    dst_ptr[0] = rhs_v0;
                    dst_ptr += sizeof(uint8_t);
                }
            }
        }
    }
};

static void ref_quant_qs8d32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs8c32) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block = num_bytes_per_block_qs8c32(bl);
    const size_t dst_stride = num_blocks_row * num_bytes_block;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        int8_t* dst_ptr = (int8_t*)rhs_qs8c32 + row_idx * dst_stride;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const float src0_0 = src_ptr[block_idx * bl + b];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                }
            }

            const float scale = amax / ((1 << 7) - 1);
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale at the beginning of the block
            *((uint16_t*)dst_ptr) = kai_cast_f16_f32(scale);
            dst_ptr += sizeof(uint16_t);

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                for (size_t i = 0; i < block_size; ++i) {
                    const size_t src_base_addr = block_idx * bl + i + subblock_idx * block_size;
                    float v0_f32 = src_ptr[src_base_addr];

                    v0_f32 *= recip_scale;

                    dst_ptr[0] = roundf(v0_f32);
                    dst_ptr += sizeof(int8_t);
                }
            }
        }
    }
};

static void ref_matmul_f32_qs8d32_qs4c32(
    size_t m, size_t n, size_t k, size_t bl, const int8_t* lhs_qa8d32, const uint8_t* rhs_qs4c32, float* dst_f32,
    float scalar_min, float scalar_max) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block_qs4c32 = num_bytes_per_block_qs4c32(bl);
    const size_t num_bytes_block_qs8c32 = num_bytes_per_block_qs8c32(bl);

    const size_t lhs_stride = num_blocks_row * num_bytes_block_qs8c32;
    const size_t rhs_stride = num_blocks_row * num_bytes_block_qs4c32;

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8d32 + row_idx * lhs_stride;
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            // Main f32 accumulator
            float main_acc = 0.0f;

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
                const int8_t* lhs_ptr = lhs_ptr_start;
                const uint8_t* rhs_ptr = rhs_qs4c32 + col_idx * rhs_stride;

                lhs_ptr += block_idx * num_bytes_block_qs8c32;
                rhs_ptr += block_idx * num_bytes_block_qs4c32;

                for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                    int32_t temp_acc = 0;

                    // Get the LHS/RHS quantization scale stored at the
                    // beginning of each block
                    const float lhs_scale = kai_cast_f32_f16(*(const uint16_t*)lhs_ptr);
                    const float rhs_scale = kai_cast_f32_f16(*(const uint16_t*)rhs_ptr);

                    lhs_ptr += sizeof(uint16_t);
                    rhs_ptr += sizeof(uint16_t);

                    for (size_t i = 0; i < block_size / 2; ++i) {
                        // Get the LHS values
                        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];
                        const int32_t lhs_v1 = (int32_t)lhs_ptr[block_size / 2];

                        // Get the RHS values
                        const uint8_t rhs_byte = rhs_ptr[0];

                        // Unpack the RHS values
                        const int32_t rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                        const int32_t rhs_v1 = (((int32_t)(rhs_byte >> 4)) - 8);

                        temp_acc += lhs_v0 * rhs_v0;
                        temp_acc += lhs_v1 * rhs_v1;

                        lhs_ptr += 1;
                        rhs_ptr += 1;
                    }

                    main_acc += temp_acc * lhs_scale * rhs_scale;
                }
            }

            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};

static bool is_output_correct(size_t num_rows, size_t num_cols, float tolerance, const float* ref, const float* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;
            printf("ERROR![%ld][%ld]: ref=%.5f vs. act=%.5f\n", y, x, ref[i], act[i]);
            is_valid = false;
        }
    }
    return is_valid;
}

int main(int argc, char** argv) {
    const size_t bl = 32;  // Block length. It must be 32
    const size_t m = 71;
    const size_t n = 63;
    const size_t k = 128;
    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;

    KAI_ASSERT_MSG((k % bl) == 0, "K must be a multiple of block length");

    const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block_qs4c32 = (bl / 2) + sizeof(int16_t);
    const size_t num_bytes_per_block_qs8c32 = bl + sizeof(int16_t);

    const size_t lhs_native_size_f32 = m * k * sizeof(float);
    const size_t rhs_native_size_f32 = n * k * sizeof(float);
    const size_t rhs_native_size_qs4c32 = n * num_blocks * num_bytes_per_block_qs4c32;

    // Allocate the memory
    uint8_t* lhs_native_mtx_f32 = new uint8_t[lhs_native_size_f32];
    uint8_t* rhs_native_mtx_f32 = new uint8_t[rhs_native_size_f32];
    uint8_t* rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];

    fill_uniform_random(m, k, (float*)lhs_native_mtx_f32, seed_lhs);
    fill_uniform_random(n, k, (float*)rhs_native_mtx_f32, seed_rhs);

    quant_qs4c32_f32(n, k, bl, (const float*)rhs_native_mtx_f32, (uint8_t*)rhs_native_mtx_qs4c32);

    delete[] rhs_native_mtx_f32;

    //----------- REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------
    // Memory sizes for the reference implementation
    // After dynamically quantized the LHS matrix, each row is quantized per block of 32 values.
    // Each block has a scale factor in f16 format.
    // The scale factor is stored at the beginning of each block
    const size_t lhs_ref_size_qa8d32 = m * num_blocks * num_bytes_per_block_qs8c32;
    const size_t dst_ref_size_f32 = m * n * sizeof(float);

    uint8_t* lhs_ref_mtx_qa8d32 = new uint8_t[lhs_ref_size_qa8d32];
    uint8_t* dst_ref_mtx_f32 = new uint8_t[dst_ref_size_f32];

    ref_quant_qs8d32_f32(m, k, bl, (const float*)lhs_native_mtx_f32, (uint8_t*)lhs_ref_mtx_qa8d32);

    ref_matmul_f32_qs8d32_qs4c32(
        m, n, k, bl, (const int8_t*)lhs_ref_mtx_qa8d32, (const uint8_t*)rhs_native_mtx_qs4c32, (float*)dst_ref_mtx_f32,
        -FLT_MAX, FLT_MAX);

    // Remove the unnecessary buffer
    delete[] lhs_ref_mtx_qa8d32;

    //----------- END REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------
    for (size_t idx_variant = 0; idx_variant < num_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << ukernel_variants[idx_variant].name << std::endl;

        // Get the packing parameters
        const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

        // Get the size in bytes for the packed matrices
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
        const size_t dst_size = ukernel_variants[idx_variant].ukernel.get_dst_size(m, n);

        // Allocate the matrices
        uint8_t* lhs_packed_mtx_qs8d32 = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed_mtx_qs4c32 = new uint8_t[rhs_packed_size];
        uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

        // If the RHS matrix contains constant values, the packing can be performed
        // only once
        struct kai_rhs_pack_qs4cxs1s0_param params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;

        // RHS packing
        kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            1, n, k,                                  // Dimensions
            nr, kr, sr,                               // Packing arguments
            bl,                                       // Block length
            (const uint8_t*)(rhs_native_mtx_qs4c32),  // RHS
            NULL,                                     // Bias
            rhs_packed_mtx_qs4c32,                    // RHS packed
            0, &params);

        const auto time_s = std::chrono::high_resolution_clock::now();

        // LHS packing
        kai_run_lhs_quant_pack_qsi8d32p_f32(
            m, k, bl,                          // Dimensions
            mr, kr, sr, 0,                     // Packing arguments
            (const float*)lhs_native_mtx_f32,  // LHS
            k * sizeof(float),                 // LHS stride
            lhs_packed_mtx_qs8d32);            // LHS packed

        // Matmul
        {
            const size_t dst_stride = n * sizeof(float);
            const size_t lhs_offset = ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
            const size_t rhs_offset = ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
            const size_t dst_offset = ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

            const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qs8d32 + lhs_offset);
            const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4c32 + rhs_offset);
            float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 + dst_offset);

            ukernel_variants[idx_variant].ukernel.run_matmul(
                m, n, k, bl,       // Dimensions
                lhs_ptr,           // LHS packed
                rhs_ptr,           // RHS packed
                dst_ptr,           // DST
                dst_stride,        // DST stride (row)
                sizeof(float),     // DST stride (col)
                -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
            );
        }

        const auto time_e = std::chrono::high_resolution_clock::now();

        const auto elap = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);

        const bool is_valid =
            is_output_correct(m, n, 0.0001f, (const float*)dst_ref_mtx_f32, (const float*)dst_act_mtx_f32);

        if (is_valid) {
            printf("TEST[%ld] = PASSED\n", idx_variant);
            std::cout << "- Performance: " << elap.count() << " us" << std::endl;
        } else {
            printf("TEST[%ld] = FAILED\n", idx_variant);
        }
        delete[] lhs_packed_mtx_qs8d32;
        delete[] rhs_packed_mtx_qs4c32;
        delete[] dst_act_mtx_f32;
    }
    delete[] lhs_native_mtx_f32;
    delete[] rhs_native_mtx_qs4c32;
    delete[] dst_ref_mtx_f32;
}

//----------- END MICRO-KERNELS TESTS
//------------------------------------
//------------------------------------

#endif  // Architectural feature check
