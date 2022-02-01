/*
 * Copyright (c) 2021-2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "fp_post_ops_act_eltwise_op_act.h"
#include "gemm_helpers.h"
#include "repeat.h"

/** (EXPERIMENTAL_POST_OPS) gemm_mm_reshaped_only_rhs kernel */
#if defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE)
#if defined(P2_ELTWISE_OP) && defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)

#define CONCAT(a, b) a##b

#define ARM_DOT1(a, b, c) \
    ({                    \
        c = fma(a, b, c); \
    })
#define ARM_DOT2(a, b, c)       \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#define ARM_DOT3(a, b, c)           \
    ({                              \
        ARM_DOT2(a, b, c);          \
        c = fma((a.s2), (b.s2), c); \
    })
#define ARM_DOT4(a, b, c)           \
    ({                              \
        ARM_DOT3(a, b, c);          \
        c = fma((a.s3), (b.s3), c); \
    })
#define ARM_DOT8(a, b, c)            \
    ({                               \
        ARM_DOT4((a.lo), (b.lo), c); \
        ARM_DOT4((a.hi), (b.hi), c); \
    })
#define ARM_DOT16(a, b, c)           \
    ({                               \
        ARM_DOT8((a.lo), (b.lo), c); \
        ARM_DOT8((a.hi), (b.hi), c); \
    })

#if N0 == 2
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##8), (c.s8));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##9), (c.s9));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##A), (c.sA));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##B), (c.sB));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##C), (c.sC));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##D), (c.sD));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##E), (c.sE));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##F), (c.sF));     \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_T_POST_ACT_ELTWISE_OP_ACT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 * @note (Optional) -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 * @note (Required) -DP2_ELTWISE_OP: The (binary) elementwise post op to perform
 * @note (Required) -DP2_ELTWISE_ARG1_HEIGHT: The height (Y dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Required) -DP2_ELTWISE_ARG1_WIDTH: The width (X dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Optional) -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 *
 * All parameters are similarly defined in kernel gemm_mm_reshaped_only_rhs_t, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 */
__kernel void gemm_mm_reshaped_only_rhs_t_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                  IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                                                  IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                  IMAGE_DECLARATION(dst),
                                                                  // Post Op arguments
                                                                  IMAGE_DECLARATION(eltwise_operand),
                                                                  uint lhs_stride_z,
                                                                  uint rhs_stride_z,
#if defined(BETA)
                                                                  uint bias_stride_z,
#endif //defined(BETA)
                                                                  uint dst_stride_z,
                                                                  uint eltwise_operand_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                                  ,
                                                                  uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                  ,
                                                                  uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                  ,
                                                                  const int M,
                                                                  const int N,
                                                                  const int K)
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (K0)
#define RHS_STEP_X ((K0) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (K0)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS reshaped matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS reshaped matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(K0, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(K0, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(K0, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(K0, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(K0, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(K0, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(K0, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(K0, a7, b, c7);
#endif // M0 > 7

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += (N0 * RHS_STEP_X * RHS_STEP_LOOP) * sizeof(DATA_TYPE);
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, 1, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS reshaped matrix
        LOAD_BLOCK(N0, 1, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(1, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(1, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(1, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(1, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(1, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(1, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(1, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(1, a7, b, c7);
#endif // M0 > 7

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, 1, PARTIAL_STORE_N0, false, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_MM_RESHAPED_ONLY_RHS_T_POST_ACT_ELTWISE_OP_ACT)

#if defined(OPENCL_IMAGE_SUPPORT) && defined(GEMM_MM_RESHAPED_ONLY_RHS_T_TEXTURE_POST_ACT_ELTWISE_OP_ACT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops. The RHS matrix is stored in OpenCL image object.
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 * @note (Optional) -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 * @note (Required) -DP2_ELTWISE_OP: The (binary) elementwise post op to perform
 * @note (Required) -DP2_ELTWISE_ARG1_HEIGHT: The height (Y dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Required) -DP2_ELTWISE_ARG1_WIDTH: The width (X dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Optional) -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 *
 * All parameters are similarly defined in kernel gemm_mm_reshaped_only_rhs_t_texture, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 * @param[in] M                        Number of rows in LHS matrix not reshaped.
 * @param[in] N                        Number of columns in RHS matrix not reshaped.
 * @param[in] K                        Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 */
__kernel void gemm_mm_reshaped_only_rhs_t_texture_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                          __read_only image2d_t rhs_img,
#if defined(BETA)
                                                                          IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                          IMAGE_DECLARATION(dst),
                                                                          // Post Op arguments
                                                                          IMAGE_DECLARATION(eltwise_operand),
                                                                          uint lhs_stride_z,
                                                                          uint rhs_stride_z,
#if defined(BETA)
                                                                          uint bias_stride_z,
#endif //defined(BETA)
                                                                          uint dst_stride_z,
                                                                          uint eltwise_operand_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                                          ,
                                                                          uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                          ,
                                                                          uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                          ,
                                                                          const int M,
                                                                          const int N,
                                                                          const int K)
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(K0)

    const uint LEFTOVER_K = K % K0;

    // Block size
#define RHS_BLOCK_SIZE (PIXEL_UNIT * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X (PIXEL_UNIT * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X PIXEL_UNIT
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (get_global_id(2) % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = get_global_id(2);
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (get_global_id(0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (get_global_id(0) / (uint)H0) + z_rhs * RHS_HEIGHT;

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0);

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix stored in a cl_image
        REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);
        LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

        // Accumulate
        ARM_DOT_K0XN0(K0, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(K0, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(K0, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(K0, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(K0, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(K0, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(K0, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(K0, a7, b, c7);
#endif // M0 > 7

        lhs_offset += K0 * sizeof(DATA_TYPE);
        x_rhs += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    if(LEFTOVER_K != 0)
    {
        // Note: We cannot read out-of-bound elements from the RHS matrix because
        // the RHS width is always multiple of K0. This is not be true for the LHS matrix

        union UNION_VEC_TYPE
        {
            DATA_TYPE s[K0];
            VEC_DATA_TYPE(DATA_TYPE, K0)
            v;
        };

        union UNION_VEC_TYPE a0 = {.v = 0 };
#if M0 > 1
        union UNION_VEC_TYPE a1 = {.v = 0 };
#endif // M0 > 1
#if M0 > 2
        union UNION_VEC_TYPE a2 = {.v = 0 };
#endif // M0 > 2
#if M0 > 3
        union UNION_VEC_TYPE a3 = {.v = 0 };
#endif // M0 > 3
#if M0 > 4
        union UNION_VEC_TYPE a4 = {.v = 0 };
#endif // M0 > 4
#if M0 > 5
        union UNION_VEC_TYPE a5 = {.v = 0 };
#endif // M0 > 5
#if M0 > 6
        union UNION_VEC_TYPE a6 = {.v = 0 };
#endif // M0 > 6
#if M0 > 7
        union UNION_VEC_TYPE a7 = {.v = 0 };
#endif // M0 > 7

        REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);

        // Load from RHS matrix
        LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

        // Load from LHS matrix
        for(int k = 0; k < LEFTOVER_K; ++k)
        {
            a0.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zlhs0);
#if M0 > 1
            a1.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zlhs1);
#endif // M0 > 1
#if M0 > 2
            a2.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zlhs2);
#endif // M0 > 2
#if M0 > 3
            a3.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zlhs3);
#endif // M0 > 3
#if M0 > 4
            a4.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zlhs4);
#endif // M0 > 4
#if M0 > 5
            a5.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zlhs5);
#endif // M0 > 5
#if M0 > 6
            a6.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zlhs6);
#endif // M0 > 6
#if M0 > 7
            a7.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zlhs7);
#endif // M0 > 7

            lhs_offset += sizeof(DATA_TYPE);
        }

        // Accumulate
        ARM_DOT_K0XN0(K0, a0.v, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(K0, a1.v, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(K0, a2.v, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(K0, a3.v, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(K0, a4.v, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(K0, a5.v, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(K0, a6.v, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(K0, a7.v, b, c7);
#endif // M0 > 7
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, 1, PARTIAL_STORE_N0, false, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef PIXEL_UNIT
}
#endif // defined(OPENCL_IMAGE_SUPPORT) && defined(GEMM_MM_RESHAPED_ONLY_RHS_T_TEXTURE_POST_ACT_ELTWISE_OP_ACT)

#define VFMA(a, b, c)     \
    ({                    \
        c = fma(a, b, c); \
    })

#if M0 == 1
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_POST_ACT_ELTWISE_OP_ACT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 * @note (Optional) -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 * @note (Required) -DP2_ELTWISE_OP: The (binary) elementwise post op to perform
 * @note (Required) -DP2_ELTWISE_ARG1_HEIGHT: The height (Y dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Required) -DP2_ELTWISE_ARG1_WIDTH: The width (X dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Optional) -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 *
 * All parameters are similarly defined in kernel gemm_mm_reshaped_only_rhs_nt, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 * @param[in] M                        Number of rows in LHS matrix not reshaped.
 * @param[in] N                        Number of columns in RHS matrix not reshaped.
 * @param[in] K                        Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 */
__kernel void gemm_mm_reshaped_only_rhs_nt_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                   IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                                                   IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                   IMAGE_DECLARATION(dst),
                                                                   // Post Op arguments
                                                                   IMAGE_DECLARATION(eltwise_operand),
                                                                   uint lhs_stride_z,
                                                                   uint rhs_stride_z,
#if defined(BETA)
                                                                   uint bias_stride_z,
#endif //defined(BETA)
                                                                   uint dst_stride_z,
                                                                   uint eltwise_operand_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                                   ,
                                                                   uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                   ,
                                                                   uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                   ,
                                                                   const int M,
                                                                   const int N,
                                                                   const int K)
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (N0)
#define RHS_STEP_X ((N0) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (N0)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS reshaped matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zin, 0);   //uint zin0=0,zin1=0,zin2=0,... zin7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0); //uint zero0=0,zero1=0,zero2=0,... zero7=0;

#if defined(REINTERPRET_INPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(N0-1)=0;

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zin);

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(0, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 1 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(1, a, b0, c);
#if K0 > 2
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 2 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(2, a, b0, c);
#endif // K0 > 2
#if K0 > 3
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 3 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(3, a, b0, c);
#endif // K0 > 3
#if K0 > 4
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 4 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(4, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 5 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(5, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 6 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(6, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 7 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(7, a, b0, c);
#endif // K0 > 4
#if K0 > 8
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 8 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(8, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 9 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(9, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 10 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(A, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 11 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(B, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 12 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(C, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 13 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(D, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 14 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(E, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 15 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(F, a, b0, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += K0 * RHS_STEP_X * RHS_STEP_LOOP * sizeof(DATA_TYPE);
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zin0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zin1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zin2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zin3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zin4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zin5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zin6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zin7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(0, a, b0, c);

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += RHS_STEP_X * sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, 1, PARTIAL_STORE_N0, false, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef RHS_STEP_LOOP
}
#endif // defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_POST_ACT_ELTWISE_OP_ACT)

#if defined(OPENCL_IMAGE_SUPPORT) && defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_TEXTURE_POST_ACT_ELTWISE_OP_ACT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops. The RHS matrix is stored in OpenCL image object.
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 * @note (Optional) -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 * @note (Required) -DP2_ELTWISE_OP: The (binary) elementwise post op to perform
 * @note (Required) -DP2_ELTWISE_ARG1_HEIGHT: The height (Y dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Required) -DP2_ELTWISE_ARG1_WIDTH: The width (X dimension) of the eltwise operand matrix of the eltwise post op at slot 2
 * @note (Optional) -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
 *
 * All parameters are similarly defined in kernel gemm_mm_reshaped_only_rhs_nt_texture, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 * @param[in] M                        Number of rows in LHS matrix not reshaped.
 * @param[in] N                        Number of columns in RHS matrix not reshaped.
 * @param[in] K                        Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 */
__kernel void gemm_mm_reshaped_only_rhs_nt_texture_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                           __read_only image2d_t rhs_img,
#if defined(BETA)
                                                                           IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                           IMAGE_DECLARATION(dst),
                                                                           // Post Op arguments
                                                                           IMAGE_DECLARATION(eltwise_operand),
                                                                           uint lhs_stride_z,
                                                                           uint rhs_stride_z,
#if defined(BETA)
                                                                           uint bias_stride_z,
#endif //defined(BETA)
                                                                           uint dst_stride_z,
                                                                           uint eltwise_operand_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                                           ,
                                                                           uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                           ,
                                                                           uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                           ,
                                                                           const int M,
                                                                           const int N,
                                                                           const int K)
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(N0)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (PIXEL_UNIT))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X ((PIXEL_UNIT) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (PIXEL_UNIT)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (z % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (x % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (x / (uint)H0) + z_rhs * RHS_HEIGHT;

    REPEAT_VAR_INIT_TO_CONST(8, uint, zin, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0);

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zin);

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(0, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 1 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(1, a, b0, c);
#if K0 > 2
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 2 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(2, a, b0, c);
#endif // K0 > 2
#if K0 > 3
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 3 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(3, a, b0, c);
#endif // K0 > 3
#if K0 > 4
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 4 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(4, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 5 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(5, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 6 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(6, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 7 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(7, a, b0, c);
#endif // K0 > 4
#if K0 > 8
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 8 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(8, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 9 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(9, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 10 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(A, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 11 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(B, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 12 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(C, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 13 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(D, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 14 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(E, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 15 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(F, a, b0, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        x_rhs += K0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zin0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zin1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zin2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zin3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zin4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zin5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zin6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zin7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));

        VFMA_M0xN0(0, a, b0, c);

        lhs_offset += sizeof(DATA_TYPE);
        x_rhs += RHS_STEP_X;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, 1, PARTIAL_STORE_N0, false, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef RHS_STEP_LOOP
}
#endif // defined(OPENCL_IMAGE_SUPPORT) && defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_TEXTURE_POST_ACT_ELTWISE_OP_ACT)
#endif // defined(P2_ELTWISE_OP) && defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE)
