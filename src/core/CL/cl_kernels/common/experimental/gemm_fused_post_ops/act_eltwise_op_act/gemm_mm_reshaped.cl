/*
 * Copyright (c) 2021 Arm Limited.
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

/** (EXPERIMENTAL_POST_OPS) gemm_mm_reshaped kernel */

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE) && defined(DATA_TYPE_ACCUMULATOR) && defined(M) && defined(N)
#if defined(P2_ELTWISE_OP) && defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)

#if defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
        c += a.s8 * b.s8;   \
        c += a.s9 * b.s9;   \
        c += a.sA * b.sA;   \
        c += a.sB * b.sB;   \
        c += a.sC * b.sC;   \
        c += a.sD * b.sD;   \
        c += a.sE * b.sE;   \
        c += a.sF * b.sF;   \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#else  // defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
        c = fma(a.s8, b.s8, c); \
        c = fma(a.s9, b.s9, c); \
        c = fma(a.sA, b.sA, c); \
        c = fma(a.sB, b.sB, c); \
        c = fma(a.sC, b.sC, c); \
        c = fma(a.sD, b.sD, c); \
        c = fma(a.sE, b.sE, c); \
        c = fma(a.sF, b.sF, c); \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#endif // defined(MIXED_PRECISION)

#if defined(ARM_DOT_K0XN0)
#undef ARM_DOT_K0XN0
#endif // defined(ARM_DOT_K0XN0)

#if N0 == 2
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
        ARM_DOT_K0((a), (b##8), (c.s8)); \
        ARM_DOT_K0((a), (b##9), (c.s9)); \
        ARM_DOT_K0((a), (b##A), (c.sA)); \
        ARM_DOT_K0((a), (b##B), (c.sB)); \
        ARM_DOT_K0((a), (b##C), (c.sC)); \
        ARM_DOT_K0((a), (b##D), (c.sD)); \
        ARM_DOT_K0((a), (b##E), (c.sE)); \
        ARM_DOT_K0((a), (b##F), (c.sF)); \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions

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
 * All parameters are similarly defined in kernel gemm_mm_reshaped_lhs_nt_rhs_t, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 */
__kernel void gemm_mm_reshaped_lhs_nt_rhs_t_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                    IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                                                    IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                    IMAGE_DECLARATION(dst),
                                                                    // Post-Op arguments
                                                                    IMAGE_DECLARATION(eltwise_operand),
                                                                    uint k,
                                                                    uint lhs_stride_z,
                                                                    uint rhs_stride_z,
#if defined(BETA)
                                                                    uint bias_stride_z,
#endif //defined(BETA)
                                                                    uint dst_stride_z,
                                                                    uint eltwise_operand_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                    ,
                                                                    uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                   )
{
    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (K0)
#define LHS_STEP_X ((K0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (K0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

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

#if defined(DUMMY_WORK_ITEMS)
    if((get_global_id(0) * N0 >= N) || (get_global_id(1) * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (get_global_id(1) % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(1) / V0) * (uint)lhs_stride_y +
                               (get_global_id(2) * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (get_global_id(0) % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(0) / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (get_global_id(2) % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += get_global_id(2) * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

    for(int i = 0; i < k; i += K0)
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
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_addr, 0, LHS_STEP_X * sizeof(DATA_TYPE), zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_addr, 0, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(a7, b, c7);
#endif // M0 > 7

        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP) * sizeof(DATA_TYPE);
        rhs_addr += (N0 * RHS_STEP_X * RHS_STEP_LOOP) * sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, get_global_id(1) * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

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
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    MIXED_PRECISION_ELTWISE_OP_BLOCK(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x, c_lp);

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}

#if defined(OPENCL_IMAGE_SUPPORT)
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
 * All parameters are similarly defined in kernel gemm_mm_reshaped_lhs_nt_rhs_t_texture, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 */
__kernel void gemm_mm_reshaped_lhs_nt_rhs_t_texture_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                            __read_only image2d_t rhs_img,
#if defined(BETA)
                                                                            IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                            IMAGE_DECLARATION(dst),
                                                                            // Post-Op arguments
                                                                            IMAGE_DECLARATION(eltwise_operand),
                                                                            uint k,
                                                                            uint lhs_stride_z,
                                                                            uint rhs_stride_z,
#if defined(BETA)
                                                                            uint bias_stride_z,
#endif //defined(BETA)
                                                                            uint dst_stride_z,
                                                                            uint eltwise_operand_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                            ,
                                                                            uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                           )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(K0)

    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (K0)
#define LHS_STEP_X ((K0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (K0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

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

#if defined(DUMMY_WORK_ITEMS)
    if((get_global_id(0) * N0 >= N) || (get_global_id(1) * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (get_global_id(1) % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(1) / V0) * (uint)lhs_stride_y +
                               (get_global_id(2) * lhs_stride_z);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (get_global_id(2) % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = get_global_id(2);
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (get_global_id(0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (get_global_id(0) / (uint)H0) + z_rhs * RHS_HEIGHT;

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

    for(int i = 0; i < K; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_addr, 0, LHS_STEP_X * sizeof(DATA_TYPE), zlhs);

        // Load values from RHS matrix stored in a cl_image
        REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);
        LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

        // Accumulate
        ARM_DOT_K0XN0(a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(a7, b, c7);
#endif // M0 > 7

        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP) * sizeof(DATA_TYPE);

        x_rhs += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, get_global_id(1) * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

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
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    MIXED_PRECISION_ELTWISE_OP_BLOCK(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x, c_lp);

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef PIXEL_UNIT
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}
#endif // defined(OPENCL_IMAGE_SUPPORT)

#if defined(LHS_TRANSPOSE)

#define VTYPE(TYPE, SIZE) VEC_DATA_TYPE(TYPE, SIZE)

#if defined(MIXED_PRECISION)

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))) * (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0)));
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#else // defined(MIXED_PRECISION

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (a) * (b);
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((a), (b), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#endif // defined(MIXED_PRECISION)

#define ARM_VVM_T_NT_1xN0x1(N0, TYPE, a, b, C)         \
    ({                                                 \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a), b, (C##0)); \
    })
#define ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s0), b, (C##0)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s1), b, (C##1)); \
    })
#define ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s2), b, (C##2)); \
    })
#define ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s3), b, (C##3)); \
    })
#define ARM_VVM_T_NT_8xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s4), b, (C##4)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s5), b, (C##5)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s6), b, (C##6)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s7), b, (C##7)); \
    })

// Factory macro for the column-vector (transposed) by row-vector (not transposed) multiplication. K0 = 1
// a is the column-vector (transposed)
// b is the row-vector (not transposed)
// C is the output matrix
// Lower case is a vector (a, b)
// Upper case is a matrix (C)
#define ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, a, b, C) ARM_VVM_T_NT_##M0##xN0x1(N0, TYPE, a, b, C)

#define ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##0), (B##0), C); \
    })
#define ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##1), (B##1), C); \
    })
#define ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##2), (B##2), C); \
    })
#define ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##3), (B##3), C); \
    })
#define ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##4), (B##4), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##5), (B##5), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##6), (B##6), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##7), (B##7), C); \
    })
#define ARM_MM_T_NT_M0xN0x16(M0, N0, TYPE, A, B, C)           \
    ({                                                        \
        ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C);           \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##8), (B##8), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##9), (B##9), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##A), (B##A), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##B), (B##B), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##C), (B##C), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##D), (B##D), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##E), (B##E), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##F), (B##F), C); \
    })

// Factory macro for the matrix (transposed) by matrix (not transposed) multiplication.
// The dimensions for this matrix multiplications are defined through M0, N0 and K0
// The dimensions supported are:
// M0: 1, 2, 3, 4, 8
// N0: 1, 2, 3, 4, 8, 16
// K0: 1, 2, 3, 4, 8, 16
// This macro calls the vector-by-matrix macro K0 times
// A, B and C are matrices
#define ARM_MM_T_NT(M0, N0, K0, TYPE, A, B, C) \
    CONCAT(ARM_MM_T_NT_M0xN0x, K0)             \
    (M0, N0, TYPE, A, B, C)

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
 * All parameters are similarly defined in kernel gemm_mm_reshaped_lhs_t_rhs_nt, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 */
__kernel void gemm_mm_reshaped_lhs_t_rhs_nt_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                    IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                                                    IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                    IMAGE_DECLARATION(dst),
                                                                    // Post-Op arguments
                                                                    IMAGE_DECLARATION(eltwise_operand),
                                                                    uint k,
                                                                    uint lhs_stride_z,
                                                                    uint rhs_stride_z,
#if defined(BETA)
                                                                    uint bias_stride_z,
#endif //defined(BETA)
                                                                    uint dst_stride_z,
                                                                    uint eltwise_operand_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                    ,
                                                                    uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                   )
{
    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (M0)
#define LHS_STEP_X ((M0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (M0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (N0)
#define RHS_STEP_X ((N0) * (H0))
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (N0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint z = get_global_id(2);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

    __global DATA_TYPE *lhs = (__global DATA_TYPE *)(lhs_addr);
    __global DATA_TYPE *rhs = (__global DATA_TYPE *)(rhs_addr);

    for(int i = 0; i < k; i += K0)
    {
        VEC_DATA_TYPE(DATA_TYPE, M0)
        a0;
        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

#if K0 > 1
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 1

#if K0 > 2
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 2

#if K0 > 3
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 3

#if K0 > 4
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 4

#if K0 > 8
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 8

#ifndef LHS_INTERLEAVE
        lhs += (M0 * K0 * (V0 - 1));
#endif // LHS_INTERLEAVE

#ifndef RHS_INTERLEAVE
        rhs += (N0 * K0 * (H0 - 1));
#endif // RHS_INTERLEAVE
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
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
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    MIXED_PRECISION_ELTWISE_OP_BLOCK(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x, c_lp);

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#if defined(OPENCL_IMAGE_SUPPORT)
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
 * All parameters are similarly defined in kernel gemm_mm_reshaped_lhs_t_rhs_nt_texture, with these additions:
 *
 * @param[in] eltwise_operand_ptr      Pointer to the eltwise operand matrix. Supported data type: F16/F32
 * @param[in] eltwise_operand_stride_x Stride of the eltwise operand matrix in X dimension (in bytes)
 * @param[in] eltwise_operand_step_x   eltwise_operand_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_y Stride of the eltwise operand matrix in Y dimension (in bytes)
 * @param[in] eltwise_operand_step_y   eltwise_operand_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] eltwise_operand_stride_z Stride of the eltwise operand tensor in Z dimension (in bytes)
 */
__kernel void gemm_mm_reshaped_lhs_t_rhs_nt_texture_post_act_eltwise_op_act(IMAGE_DECLARATION(lhs),
                                                                            __read_only image2d_t rhs_img,
#if defined(BETA)
                                                                            IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                                            IMAGE_DECLARATION(dst),
                                                                            // Post-Op arguments
                                                                            IMAGE_DECLARATION(eltwise_operand),
                                                                            uint k,
                                                                            uint lhs_stride_z,
                                                                            uint rhs_stride_z,
#if defined(BETA)
                                                                            uint bias_stride_z,
#endif //defined(BETA)
                                                                            uint dst_stride_z,
                                                                            uint eltwise_operand_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                            ,
                                                                            uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                                           )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(N0)

    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (M0)
#define LHS_STEP_X ((M0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (M0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (PIXEL_UNIT))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X ((PIXEL_UNIT) * (H0))
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (PIXEL_UNIT)
#endif // defined(RHS_INTERLEAVE)

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (z % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (x % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (x / (uint)H0) + z_rhs * RHS_HEIGHT;

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

    __global DATA_TYPE *lhs = (__global DATA_TYPE *)(lhs_addr);

    for(int i = 0; i < K; i += K0)
    {
        VEC_DATA_TYPE(DATA_TYPE, M0)
        a0;
        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

#if K0 > 1
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 1 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 1

#if K0 > 2
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 2 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 2

#if K0 > 3
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 3 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 3

#if K0 > 4
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 4 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 5 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 6 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 7 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 4

#if K0 > 8
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 8 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 9 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 10 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 11 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 12 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 13 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 14 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 15 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 8

#ifndef LHS_INTERLEAVE
        lhs += (M0 * K0 * (V0 - 1));
#endif // LHS_INTERLEAVE

        x_rhs += K0 * RHS_STEP_X;
#ifndef RHS_INTERLEAVE
        x_rhs += (PIXEL_UNIT * K0 * (H0 - 1));
#endif // RHS_INTERLEAVE
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
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
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, false, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    MIXED_PRECISION_ELTWISE_OP_BLOCK_BROADCAST(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    MIXED_PRECISION_ELTWISE_OP_BLOCK(ADD, M0, N0, c, bias, DATA_TYPE_ACCUMULATOR, bias_hp);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    // c = act(c)
    POST_OP1_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);
    // c = c + eltwise_operand (mix-precision, broadcast, boundary aware)
    POST_OP2_ELTWISE_OP(P2_ELTWISE_OP, M0, N0, c, eltwise_operand, DATA_TYPE, DATA_TYPE_ACCUMULATOR, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
    // c = act(c)
    POST_OP3_ACTIVATION_OPTIONAL(M0, DATA_TYPE, DATA_TYPE_ACCUMULATOR, N0, c);

    // Store output block
    MIXED_PRECISION_STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x, c_lp);

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef PIXEL_UNIT
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}
#endif // defined(OPENCL_IMAGE_SUPPORT)

#endif // defined(LHS_TRANSPOSE)
#endif // defined(P2_ELTWISE_OP) && defined(P2_ELTWISE_ARG1_HEIGHT) && defined(P2_ELTWISE_ARG1_WIDTH)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE) && defined(DATA_TYPE_ACCUMULATOR) && defined(M) && defined(N)