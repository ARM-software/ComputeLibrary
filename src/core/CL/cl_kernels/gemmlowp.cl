/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "gemm_helpers.h"
#include "helpers_asymm.h"
#include "repeat.h"

#if defined(DATA_TYPE) && defined(ACC_DATA_TYPE)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), (val));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

/** Specialized macros to perform the dot product instruction between two vectors of size N [1,16]. These macros use the dot8 instruction */
#define ARM_DOT1(a, b, c)                                                                                                                               \
    ({                                                                                                                                                  \
        ARM_DOT((VEC_DATA_TYPE(DATA_TYPE, 4))(a, (VEC_DATA_TYPE(DATA_TYPE, 3))0), (VEC_DATA_TYPE(DATA_TYPE, 4))(b, (VEC_DATA_TYPE(DATA_TYPE, 3))0), c); \
    })
#define ARM_DOT2(a, b, c)                                                                                                                               \
    ({                                                                                                                                                  \
        ARM_DOT((VEC_DATA_TYPE(DATA_TYPE, 4))(a, (VEC_DATA_TYPE(DATA_TYPE, 2))0), (VEC_DATA_TYPE(DATA_TYPE, 4))(b, (VEC_DATA_TYPE(DATA_TYPE, 2))0), c); \
    })
#define ARM_DOT3(a, b, c)                                                                                           \
    ({                                                                                                              \
        ARM_DOT((VEC_DATA_TYPE(DATA_TYPE, 4))(a, (DATA_TYPE)0), (VEC_DATA_TYPE(DATA_TYPE, 4))(b, (DATA_TYPE)0), c); \
    })
#define ARM_DOT4(a, b, c) \
    ({                    \
        ARM_DOT(a, b, c); \
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

#else // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

/** Specialized macros to perform the dot product instruction between two vectors of size K0 [1,16] without using the dot8 instruction. */
#define ARM_DOT1(a, b, c)          \
    ({                             \
        c += (ACC_DATA_TYPE)a * b; \
    })
#define ARM_DOT2(a, b, c)                \
    ({                                   \
        c += (ACC_DATA_TYPE)a.s0 * b.s0; \
        c += (ACC_DATA_TYPE)a.s1 * b.s1; \
    })
#define ARM_DOT3(a, b, c)                \
    ({                                   \
        ARM_DOT2(a, b, c);               \
        c += (ACC_DATA_TYPE)a.s2 * b.s2; \
    })
#define ARM_DOT4(a, b, c)                \
    ({                                   \
        ARM_DOT3(a, b, c);               \
        c += (ACC_DATA_TYPE)a.s3 * b.s3; \
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
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

/** Specialized macros to perform a broadcast dot product operation between one vector "a" and N0 vectors "b" of size K0 [1,16] */
#define ARM_DOT_K0X1(k0, a, b, c)         \
    ({                                    \
        ARM_DOT_K0(k0, (a), (b##0), (c)); \
    })
#define ARM_DOT_K0X2(k0, a, b, c)            \
    ({                                       \
        ARM_DOT_K0(k0, (a), (b##0), (c.s0)); \
        ARM_DOT_K0(k0, (a), (b##1), (c.s1)); \
    })
#define ARM_DOT_K0X3(k0, a, b, c)            \
    ({                                       \
        ARM_DOT_K0X2(k0, a, b, c);           \
        ARM_DOT_K0(k0, (a), (b##2), (c.s2)); \
    })
#define ARM_DOT_K0X4(k0, a, b, c)            \
    ({                                       \
        ARM_DOT_K0X3(k0, a, b, c);           \
        ARM_DOT_K0(k0, (a), (b##3), (c.s3)); \
    })
#define ARM_DOT_K0X8(k0, a, b, c)            \
    ({                                       \
        ARM_DOT_K0X4(k0, a, b, c);           \
        ARM_DOT_K0(k0, (a), (b##4), (c.s4)); \
        ARM_DOT_K0(k0, (a), (b##5), (c.s5)); \
        ARM_DOT_K0(k0, (a), (b##6), (c.s6)); \
        ARM_DOT_K0(k0, (a), (b##7), (c.s7)); \
    })
#define ARM_DOT_K0X16(k0, a, b, c)           \
    ({                                       \
        ARM_DOT_K0X8(k0, a, b, c);           \
        ARM_DOT_K0(k0, (a), (b##8), (c.s8)); \
        ARM_DOT_K0(k0, (a), (b##9), (c.s9)); \
        ARM_DOT_K0(k0, (a), (b##A), (c.sA)); \
        ARM_DOT_K0(k0, (a), (b##B), (c.sB)); \
        ARM_DOT_K0(k0, (a), (b##C), (c.sC)); \
        ARM_DOT_K0(k0, (a), (b##D), (c.sD)); \
        ARM_DOT_K0(k0, (a), (b##E), (c.sE)); \
        ARM_DOT_K0(k0, (a), (b##F), (c.sF)); \
    })

/** Specialized macros to perform a partial matrix multiplication with dimensions M0,N0,K0 */
#define ARM_MM_K0XN0X1(n0, k0, a, b, c)           \
    ({                                            \
        ARM_DOT_K0XN0(n0, k0, (a##0), b, (c##0)); \
    })
#define ARM_MM_K0XN0X2(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X1(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##1), b, (c##1)); \
    })
#define ARM_MM_K0XN0X3(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X2(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##2), b, (c##2)); \
    })
#define ARM_MM_K0XN0X4(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X3(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##3), b, (c##3)); \
    })
#define ARM_MM_K0XN0X5(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X4(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##4), b, (c##4)); \
    })
#define ARM_MM_K0XN0X6(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X5(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##5), b, (c##5)); \
    })
#define ARM_MM_K0XN0X7(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X6(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##6), b, (c##6)); \
    })
#define ARM_MM_K0XN0X8(n0, k0, a, b, c)           \
    ({                                            \
        ARM_MM_K0XN0X7(n0, k0, a, b, c);          \
        ARM_DOT_K0XN0(n0, k0, (a##7), b, (c##7)); \
    })

#define ARM_DOT_K0(k0, a, b, c) \
    ({                          \
        CONCAT(ARM_DOT, k0)     \
        ((a), (b), (c));        \
    })

#define ARM_DOT_K0XN0(n0, k0, a, b, c) \
    ({                                 \
        CONCAT(ARM_DOT_K0X, n0)        \
        (k0, (a), b, (c));             \
    })

#define ARM_MM_K0XN0XM0(m0, n0, k0, a, b, c) \
    ({                                       \
        CONCAT(ARM_MM_K0XN0X, m0)            \
        (n0, k0, a, b, c);                   \
    })

/** Specialized macros to perform a broadcast dot product operation between one vector "a" and N0 vectors "b" of size K0 [1,16] */
#define ARM_MUL_N0X1(VECTOR_ACC_TYPE, a, b, c)   \
    ({                                           \
        c += CONVERT(b##0, VECTOR_ACC_TYPE) * a; \
    })
#define ARM_MUL_N0X2(VECTOR_ACC_TYPE, a, b, c)        \
    ({                                                \
        c += CONVERT(b##0, VECTOR_ACC_TYPE) * a.s##0; \
        c += CONVERT(b##1, VECTOR_ACC_TYPE) * a.s##1; \
    })
#define ARM_MUL_N0X3(VECTOR_ACC_TYPE, a, b, c)        \
    ({                                                \
        ARM_MUL_N0X2(VECTOR_ACC_TYPE, a, b, c);       \
        c += CONVERT(b##2, VECTOR_ACC_TYPE) * a.s##2; \
    })
#define ARM_MUL_N0X4(VECTOR_ACC_TYPE, a, b, c)        \
    ({                                                \
        ARM_MUL_N0X3(VECTOR_ACC_TYPE, a, b, c);       \
        c += CONVERT(b##3, VECTOR_ACC_TYPE) * a.s##3; \
    })
#define ARM_MUL_N0X8(VECTOR_ACC_TYPE, a, b, c)        \
    ({                                                \
        ARM_MUL_N0X4(VECTOR_ACC_TYPE, a, b, c);       \
        c += CONVERT(b##4, VECTOR_ACC_TYPE) * a.s##4; \
        c += CONVERT(b##5, VECTOR_ACC_TYPE) * a.s##5; \
        c += CONVERT(b##6, VECTOR_ACC_TYPE) * a.s##6; \
        c += CONVERT(b##7, VECTOR_ACC_TYPE) * a.s##7; \
    })
#define ARM_MUL_N0X16(VECTOR_ACC_TYPE, a, b, c)       \
    ({                                                \
        ARM_MUL_N0X8(VECTOR_ACC_TYPE, a, b, c);       \
        c += CONVERT(b##8, VECTOR_ACC_TYPE) * a.s##8; \
        c += CONVERT(b##9, VECTOR_ACC_TYPE) * a.s##9; \
        c += CONVERT(b##A, VECTOR_ACC_TYPE) * a.s##A; \
        c += CONVERT(b##B, VECTOR_ACC_TYPE) * a.s##B; \
        c += CONVERT(b##C, VECTOR_ACC_TYPE) * a.s##C; \
        c += CONVERT(b##D, VECTOR_ACC_TYPE) * a.s##D; \
        c += CONVERT(b##E, VECTOR_ACC_TYPE) * a.s##E; \
        c += CONVERT(b##F, VECTOR_ACC_TYPE) * a.s##F; \
    })
/** Specialized macros to perform a a partial matrix multiplication with dimensions M0,N0,K0 */
#define ARM_MM_NATIVE_N0XK0X1(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##0), b, (c##0)); \
    })
#define ARM_MM_NATIVE_N0XK0X2(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X1(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##1), b, (c##1)); \
    })
#define ARM_MM_NATIVE_N0XK0X3(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X2(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##2), b, (c##2)); \
    })
#define ARM_MM_NATIVE_N0XK0X4(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X3(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##3), b, (c##3)); \
    })
#define ARM_MM_NATIVE_N0XK0X5(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X4(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##4), b, (c##4)); \
    })
#define ARM_MM_NATIVE_N0XK0X6(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X5(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##5), b, (c##5)); \
    })
#define ARM_MM_NATIVE_N0XK0X7(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X6(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##6), b, (c##6)); \
    })
#define ARM_MM_NATIVE_N0XK0X8(VECTOR_ACC_TYPE, k0, a, b, c)    \
    ({                                                         \
        ARM_MM_NATIVE_N0XK0X7(VECTOR_ACC_TYPE, k0, a, b, c);   \
        ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, (a##7), b, (c##7)); \
    })
#define ARM_MUL_N0XK0(VECTOR_ACC_TYPE, k0, a, b, c) \
    ({                                              \
        CONCAT(ARM_MUL_N0X, k0)                     \
        (VECTOR_ACC_TYPE, (a), b, (c));             \
    })
#define ARM_MM_NATIVE_N0XK0XM0(VECTOR_ACC_TYPE, m0, k0, a, b, c) \
    ({                                                           \
        CONCAT(ARM_MM_NATIVE_N0XK0X, m0)                         \
        (VECTOR_ACC_TYPE, k0, a, b, c);                          \
    })

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(M) && defined(N)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices with QASYMM/QASYMM_SIGNED data type.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The accumulator data type must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions M and N must be passed at compile time using -DM and -DN (i.e. -DM=52 and -DN=90).
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (i.e. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - V0 >= 1
 *  - H0 >= 1
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: QASYMM8/QASYMM_SIGNED
 * @param[in]  lhs_stride_x                      Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                      Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                      Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                      Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS reshaped matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  k                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                      Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad               (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_reshaped_lhs_nt_rhs_t(IMAGE_DECLARATION(lhs),
                                                IMAGE_DECLARATION(rhs),
                                                IMAGE_DECLARATION(dst),
                                                uint k,
                                                uint lhs_stride_z,
                                                uint rhs_stride_z,
                                                uint dst_stride_z
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

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global DATA_TYPE *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

    // Compute RHS matrix address
    __global DATA_TYPE *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zrhs, 0);

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(ACC_DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(ACC_DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    for(int i = 0; i < k; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_addr, 0, LHS_STEP_X, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_addr, 0, RHS_STEP_X, zrhs);

        // Partial matrix multiplication M0,N0,K0
        ARM_MM_K0XN0XM0(M0, N0, K0, a, b, c);

        // Update address
        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP);
        rhs_addr += (N0 * RHS_STEP_X * RHS_STEP_LOOP);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(int)) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Convert and store output block
    CONVERT_STORE_BLOCK(M0, N0, int, c, dst_addr, dst_stride_y, zout);

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(K)

#if defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(K)

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is transposed
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The accumulator data type must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note The number of columns of LHS matrix must be passed at compile time using -DK (i.e. -DK=64)
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (i.e. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (i.e. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: QASYMM8/QASYMM8_SIGNED
 * @param[in]  lhs_stride_x                      Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                      Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                      Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                      Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS reshaped matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                      Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad               (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad               (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_reshaped_only_rhs_t(IMAGE_DECLARATION(lhs),
                                              IMAGE_DECLARATION(rhs),
                                              IMAGE_DECLARATION(dst),
                                              uint lhs_stride_z,
                                              uint rhs_stride_z,
                                              uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                              ,
                                              uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                              ,
                                              uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                             )
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

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + y * M0 * (uint)lhs_stride_y;

    // Compute RHS matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zrhs, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(ACC_DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(ACC_DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(N0-1)=0;

    for(int i = 0; i < K; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X, zrhs);

        // Partial matrix multiplication M0,N0,K0
        ARM_MM_K0XN0XM0(M0, N0, K0, a, b, c);

        lhs_offset += K0;
        rhs_offset += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Convert and store output block
    CONVERT_STORE_BLOCK(M0, N0, int, c, dst_addr, dst_stride_y, zout);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}

#if defined(RESULT_OFFSET) && defined(RESULT_SHIFT) && defined(RESULT_MULTIPLIER)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices with fused output stage using fixed-point arithmetic.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is transposed
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The accumulator data type must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note The number of columns of LHS matrix must be passed at compile time using -DK (i.e. -DK=64)
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (i.e. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (i.e. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @note The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULTIPLIER and -DRESULT_SHIFT
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 * @note In case of per-channel quantization of matrix B, -DPER_CHANNEL_QUANTIZATION must be passed at compile time.
 *
 * @param[in]  lhs_ptr                                          Pointer to the LHS reshaped matrix. Supported data type: QASYMM8/QASYMM8_SIGNED
 * @param[in]  lhs_stride_x                                     Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                                       src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                                     Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                                       src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes                The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                                          Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                                     Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                                       src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                                     Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                                       src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes                The offset of the first element in the RHS reshaped matrix
 * @param[out] dst_ptr                                          Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                                     Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                                       dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                                     Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                                       dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes                The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                                     Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                                     Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                                     Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                              (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                              (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 * @param[in]  sum_col_ptr                                      (Optional) Pointer to the source tensor. Supported data type: S32
 * @param[in]  sum_col_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                                   (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                                   (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                                      (Optional) Pointer to the source tensor. Supported data type: S32
 * @param[in]  sum_row_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                                   (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                                   (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  biases_ptr                                       (Optional) Pointer to the biases tensor. Supported data type: S32
 * @param[in]  biases_stride_x                                  (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases tensor
 * @param[in]  result_multipliers_ptr                           (Optional) Pointer to the output multipliers vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_multipliers_stride_x                      (Optional) Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in]  result_multipliers_step_x                        (Optional) output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_multipliers_offset_first_element_in_bytes (Optional) The offset of the first element in the output multipliers vector
 * @param[in]  result_shifts_ptr                                (Optional) Pointer to the output shifts vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_shifts_stride_x                           (Optional) Stride of the output shifts vector in X dimension (in bytes)
 * @param[in]  result_shifts_step_x                             (Optional) output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_shifts_offset_first_element_in_bytes      (Optional) The offset of the first element in the output shifts vector
 */
__kernel void gemmlowp_mm_reshaped_only_rhs_t_fused_output_stage_fixedpoint(IMAGE_DECLARATION(lhs),
                                                                            IMAGE_DECLARATION(rhs),
                                                                            IMAGE_DECLARATION(dst),
                                                                            uint lhs_stride_z,
                                                                            uint rhs_stride_z,
                                                                            uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                                            ,
                                                                            uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                                            ,
                                                                            uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
#if defined(A_OFFSET)
                                                                            ,
                                                                            IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                                                                            ,
                                                                            IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
                                                                            ,
                                                                            VECTOR_DECLARATION(biases)
#endif // defined(ADD_BIAS)
#if defined(PER_CHANNEL_QUANTIZATION)
                                                                            ,
                                                                            VECTOR_DECLARATION(result_multipliers),
                                                                            VECTOR_DECLARATION(result_shifts)
#endif // defined(PER_CHANNEL_QUANTIZATION)
                                                                           )
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

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + y * M0 * (uint)lhs_stride_y;

    // Compute RHS matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zrhs, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(ACC_DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(ACC_DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(N0-1)=0;

    for(int i = 0; i < K; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X, zrhs);

        // Partial matrix multiplication M0,N0,K0
        ARM_MM_K0XN0XM0(M0, N0, K0, a, b, c);

        lhs_offset += K0;
        rhs_offset += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    // Result of MM is of type DATA_TYPE
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(DATA_TYPE) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Convert result of matrix multiplication to S32
    REPEAT_VAR_INIT_CONVERT_SAT(M0, VEC_DATA_TYPE(int, N0), c, c_int);

    // Offset contribution: c += (A_OFFSET * sum_col) + (B_OFFSET * sum_row) +  K_OFFSET;
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(int, N0), offset_s32_, K_OFFSET);

#if defined(A_OFFSET)
    // Compute the offset contribution due to A_OFFSET
    __global uchar *sum_col_addr = sum_col_ptr + sum_col_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int);

#if defined(SUM_COL_HAS_BATCHES)
    sum_col_addr += z * sum_col_stride_y;
#endif // defined(SUM_COL_HAS_BATCHES)
    VEC_DATA_TYPE(int, N0)
    a_offset_s32 = VLOAD(N0)(0, (__global int *)sum_col_addr);
    a_offset_s32 *= (VEC_DATA_TYPE(int, N0))A_OFFSET;

    REPEAT_ADD_VECTOR_TO_VAR(M0, offset_s32_, a_offset_s32);
#endif // defined(A_OFFSET)

#if defined(B_OFFSET)
    // Compute the offset contribution due to B_OFFSET
    // Note: The sum_row tensor is generated through CLGEMMLowpMatrixAReductionKernel which
    // does not introduce paddings. For this reason is safe to access the tensor in this manner
    // without considering that the coordinate "y" could come from an input 3D tensor
    __global uchar *sum_row_addr = sum_row_ptr + sum_row_offset_first_element_in_bytes + (y * (uint)M0) * sizeof(int) + z * sum_row_stride_y;

    LOAD_SCALAR_AS_VECTOR(M0, N0, int, b_offset_s32_, sum_row_addr, 0, sum_row_stride_x);

    REPEAT_MLA_VAR_WITH_CONST_VEC(M0, offset_s32_, b_offset_s32_, (VEC_DATA_TYPE(int, N0))B_OFFSET);
#endif // defined(B_OFFSET)

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int);

    VEC_DATA_TYPE(int, N0)
    bias_values = VLOAD(N0)(0, (__global int *)bias_addr);
    REPEAT_ADD_VECTOR_TO_VAR(M0, offset_s32_, bias_values);
#endif // defined(ADD_BIAS)

    REPEAT_ADD_TWO_VARS(M0, c_int, offset_s32_);

    // Multiply by result_mult_int and shift
#if defined(PER_CHANNEL_QUANTIZATION)
    __global uchar *result_multipliers_addr = result_multipliers_ptr + result_multipliers_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int);
    __global uchar *result_shifts_addr      = result_shifts_ptr + result_shifts_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int);

    VEC_DATA_TYPE(int, N0)
    res_mul = VLOAD(N0)(0, (__global int *)result_multipliers_addr);
    VEC_DATA_TYPE(int, N0)
    res_shift = VLOAD(N0)(0, (__global int *)result_shifts_addr);

    REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_PER_CHANNEL(M0, N0, c_int, res_mul, res_shift);
#else // defined(PER_CHANNEL_QUANTIZATION)

#if RESULT_SHIFT < 0
    REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(M0, N0, c_int, RESULT_MULTIPLIER, RESULT_SHIFT);
#else  // RESULT_SHIFT >= 0
    REPEAT_ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(M0, N0, c_int, RESULT_MULTIPLIER, RESULT_SHIFT);
#endif // RESULT_SHIFT < 0

#endif // defined(PER_CHANNEL_QUANTIZATION)

    // Add the offset terms to GEMM's result
    REPEAT_ADD_CONST_TO_VAR(M0, VEC_DATA_TYPE(int, N0), c_int, RESULT_OFFSET);

#if defined(MIN_BOUND)
    REPEAT_MAX_CONST_VAR(M0, VEC_DATA_TYPE(int, N0), c_int, MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    REPEAT_MIN_CONST_VAR(M0, VEC_DATA_TYPE(int, N0), c_int, MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Convert and store output block (does convert saturate)
    CONVERT_STORE_BLOCK(M0, N0, DATA_TYPE, c_int, dst_addr, dst_stride_y, zout);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_SHIFT) && defined(RESULT_MULTIPLIER)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE) && defined(K)

#if defined(M0) && defined(N0) && defined(K0) && defined(K)

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is NOT reshaped
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The accumulator data type must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note The number of columns of LHS matrix must be passed at compile time using -DK (i.e. -DK=64)
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (i.e. -DM0=2)
 * @note The number of N0 columns to process must be passed at compile time using -DN0 (i.e. -DN0=2)
 * @note The number of K0 partial accumulations must be passed at compile time using -DK0 (i.e., -DK0=2)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: QASYMM8
 * @param[in]  lhs_stride_x                      Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                      Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                      Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                      Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS reshaped matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                      Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad               (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad               (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_native(IMAGE_DECLARATION(lhs),
                                 IMAGE_DECLARATION(rhs),
                                 IMAGE_DECLARATION(dst),
                                 uint lhs_stride_z,
                                 uint rhs_stride_z,
                                 uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                 ,
                                 uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                 ,
                                 uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + y * M0 * (uint)lhs_stride_y;

    // Compute RHS matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + x * N0;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zrhs, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(ACC_DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(ACC_DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;

    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(K0, N0, DATA_TYPE, b, rhs_ptr, rhs_offset, rhs_stride_y, zrhs);

        // Partial matrix multiplication M0,N0,K0
#if(GPU_ARCH == GPU_ARCH_MIDGARD)
        ARM_MM_NATIVE_N0XK0XM0(VEC_DATA_TYPE(ACC_DATA_TYPE, N0), M0, K0, a, b, c);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        // Transpose the values from RHS matrix
        TRANSPOSE_K0XN0(K0, N0, b_t, b, DATA_TYPE);

        ARM_MM_K0XN0XM0(M0, N0, K0, a, b_t, c);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        // Update the offset
        lhs_offset += K0;
        rhs_offset += K0 * rhs_stride_y;
    }

    // Left-over for loop
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, 1, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(1, N0, DATA_TYPE, b, rhs_ptr, rhs_offset, rhs_stride_y, zrhs);

        // Partial matrix multiplication M0,N0,1
#if(GPU_ARCH == GPU_ARCH_MIDGARD)
        ARM_MM_NATIVE_N0XK0XM0(VEC_DATA_TYPE(ACC_DATA_TYPE, N0), M0, 1, a, b, c);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        // Transpose the values from RHS matrix
        TRANSPOSE_K0XN0(1, N0, b_t, b, DATA_TYPE);

        ARM_MM_K0XN0XM0(M0, N0, 1, a, b_t, c);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        // Update the offset
        lhs_offset += 1;
        rhs_offset += rhs_stride_y;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0) * sizeof(int) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Convert and store output block
    CONVERT_STORE_BLOCK(M0, N0, int, c, dst_addr, dst_stride_y, zout);
}
#endif // defined(M0) && defined(N0) && defined(K0) && defined(K)

#if defined(COLS_A)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 * It is also possible to multiply each reduced row by a scalar value, if SCALAR is passed at compile time.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The data type for the accumulation must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note In case of scaling the scalar value must be passed at compile time using -DSCALAR (e.g. -DSCALAR=3)
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8/QASYMM8_SIGNED/QSYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_matrix_a_reduction(TENSOR3D_DECLARATION(src),
                                          IMAGE_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(ACC_DATA_TYPE, 4)
    sum_row_32            = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))0;
    ACC_DATA_TYPE sum_row = 0;

    __global const DATA_TYPE *matrix_a = (__global const DATA_TYPE *)(src.ptr + get_global_id(0) * src_stride_y + get_global_id(1) * src_stride_z);

    int i = 0;

    // This for loop performs 16 accumulations
    for(; i <= ((int)COLS_A - 16); i += 16)
    {
        const VEC_DATA_TYPE(DATA_TYPE, 16) a0 = vload16(0, matrix_a + i);

        sum_row_32 += CONVERT(a0.s0123, VEC_DATA_TYPE(ACC_DATA_TYPE, 4)) + CONVERT(a0.s4567, VEC_DATA_TYPE(ACC_DATA_TYPE, 4)) + CONVERT(a0.s89AB, VEC_DATA_TYPE(ACC_DATA_TYPE, 4)) + CONVERT(a0.sCDEF,
                      VEC_DATA_TYPE(ACC_DATA_TYPE, 4));
    }

    // This for loop performs the leftover accumulations
    for(; i < COLS_A; ++i)
    {
        sum_row += (ACC_DATA_TYPE)matrix_a[i];
    }

    sum_row += sum_row_32.s0 + sum_row_32.s1 + sum_row_32.s2 + sum_row_32.s3;

#if defined(SCALAR)
    sum_row *= (int)SCALAR;
#endif // defined(SCALAR)
    *((__global int *)dst.ptr) = (int)sum_row;
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A using the arm dot product instruction.
 * It is also possible to multiply each reduced row by a scalar value, if SCALAR is passed at compile time.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The data type for the accumulation must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note In case of scaling the scalar value must be passed at compile time using -DSCALAR (e.g. -DSCALAR=3)
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8/QASYMM8_SIGNED/QSYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_matrix_a_reduction_dot8(TENSOR3D_DECLARATION(src),
                                               IMAGE_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT(dst);

    ACC_DATA_TYPE sum_row = 0;

    __global const DATA_TYPE *matrix_a = (__global const DATA_TYPE *)(src.ptr + get_global_id(0) * src_stride_y + get_global_id(1) * src_stride_z);

    int i = 0;

    // This for loop performs 16 accumulations
    for(; i <= ((int)COLS_A - 32); i += 32)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        a0 = vload16(0, matrix_a + i);

        sum_row += arm_dot(a0.s0123, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.s4567, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.s89AB, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.sCDEF, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));

        a0 = vload16(1, matrix_a + i);

        sum_row += arm_dot(a0.s0123, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.s4567, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.s89AB, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
        sum_row += arm_dot(a0.sCDEF, (VEC_DATA_TYPE(DATA_TYPE, 4))(1));
    }

    // This for loop performs the leftover accumulations
    for(; i < COLS_A; ++i)
    {
        sum_row += (ACC_DATA_TYPE)matrix_a[i];
    }

#if defined(SCALAR)
    sum_row *= (int)SCALAR;
#endif // defined(SCALAR)
    *((__global int *)dst.ptr) = (int)sum_row;
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#endif // defined(COLS_A)

#if defined(COLS_B) && defined(ROWS_B)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 * It is also possible to multiply each reduced column by a scalar value, if SCALAR is passed at compile time.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix B columns and rows needs to be passed at compile time using -DCOLS_B and -DROWS_B
 * @note The input data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=uchar)
 * @note The data type for the accumulation must be passed at compile time using -DACC_DATA_TYPE (i.e. -DACC_DATA_TYPE=uint)
 * @note In case of scaling the scalar value must be passed at compile time using -DSCALAR (i.e. -DSCALAR=3)
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_matrix_b_reduction(TENSOR3D_DECLARATION(src),
                                          IMAGE_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(ACC_DATA_TYPE, 16)
    sum_col_32 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 16))0;

    __global const DATA_TYPE *matrix_b = (__global const DATA_TYPE *)(src.ptr + get_global_id(1) * src_stride_z);

    int i = 0;
    // This for loop performs 4 accumulations
    for(; i <= ((int)ROWS_B - 4); i += 4)
    {
        const VEC_DATA_TYPE(DATA_TYPE, 16)
        b0 = vload16(0, matrix_b + 0 * src_stride_y);
        const VEC_DATA_TYPE(DATA_TYPE, 16)
        b1 = vload16(0, matrix_b + 1 * src_stride_y);
        const VEC_DATA_TYPE(DATA_TYPE, 16)
        b2 = vload16(0, matrix_b + 2 * src_stride_y);
        const VEC_DATA_TYPE(DATA_TYPE, 16)
        b3 = vload16(0, matrix_b + 3 * src_stride_y);

        sum_col_32 += CONVERT(b0, VEC_DATA_TYPE(ACC_DATA_TYPE, 16)) + CONVERT(b1, VEC_DATA_TYPE(ACC_DATA_TYPE, 16)) + CONVERT(b2, VEC_DATA_TYPE(ACC_DATA_TYPE, 16)) + CONVERT(b3, VEC_DATA_TYPE(ACC_DATA_TYPE,
                      16));

        matrix_b += 4 * src_stride_y;
    }

    // This for loop perfoms the leftover accumulations
    for(; i < (int)ROWS_B; ++i)
    {
        const VEC_DATA_TYPE(DATA_TYPE, 16)
        b0 = vload16(0, matrix_b);

        sum_col_32 += CONVERT(b0, VEC_DATA_TYPE(ACC_DATA_TYPE, 16));

        matrix_b += src_stride_y;
    }

#if defined(SCALAR)
    sum_col_32 *= (VEC_DATA_TYPE(ACC_DATA_TYPE, 16))SCALAR;
#endif // defined(SCALAR)
    VSTORE(16)
    (convert_int16(sum_col_32), 0, (__global int *)dst.ptr);
}
#endif // defined(COLS_B) && defined(ROWS_B)

#endif // defined(DATA_TYPE) && defined(ACC_DATA_TYPE)

#if defined(K_OFFSET)

/* Helper function used to calculate the offset contribution after matrix multiplication.
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication),
 * and calculates the offset contribution of matrix A and matrix B.
 *
 * @attention The k_offset = a_offset * b_offset * k (where k is the number of matrix A columns) needs to be passed at compile time using -DK_OFFSET (i.e. -DK_OFFSET=1200)
 * @note In case the offset contribution due to a_offset is required, a_offset needs to be passed at compile time using -DA_OFFSET (i.e. -DA_OFFSET=1)
 * @note In case the offset contribution due to b_offset is required, b_offset needs to be passed at compile time using -DB_OFFSET (i.e. -DB_OFFSET=6)
 * @note In case sum_col has batches, -DSUM_COL_HAS_BATCHES must be passed at compile time. Usually if gemmlowp is used to accelerate convolution layer, sum_col will not have batches
 *
 * @param[in] x                                     get_global_id(0) * 4
 * @param[in] y                                     get_global_id(1)
 * @param[in] z                                     get_global_id(2)
 * @param[in] sum_col_ptr                           (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_col_stride_x                      (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_col_step_x                        (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_col_stride_y                      (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_col_step_y                        (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_col_offset_first_element_in_bytes (Optional) The offset of the first element in the source tensor
 * @param[in] sum_row_ptr                           (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_row_stride_x                      (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_row_step_x                        (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_row_stride_y                      (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_row_step_y                        (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_row_offset_first_element_in_bytes (Optional) The offset of the first element in the source tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases tensor
 */
inline int4 offset_contribution(
    int x,
    int y,
    int z
#if defined(A_OFFSET)
    ,
    IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
    ,
    IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif // defined(ADD_BIAS)
)
{
    int4 a_offset_s32 = (int4)0;
    int4 b_offset_s32 = (int4)0;

    int batch_id = z;
#if defined(DEPTH_INPUT3D)
    batch_id /= (int)DEPTH_INPUT3D;
#endif // defined(DEPTH_INPUT3D)

#if defined(A_OFFSET)
    // Compute the offset contribution due to A_OFFSET
    __global uchar *sum_col_addr = sum_col_ptr + sum_col_offset_first_element_in_bytes + x * sizeof(int);

    // Compute the offset contribution due to A_OFFSET
#if defined(SUM_COL_HAS_BATCHES)
    a_offset_s32 = vload4(0, (__global int *)(sum_col_addr + batch_id * sum_col_stride_y));
#else  // defined(SUM_COL_HAS_BATCHES)
    a_offset_s32 = vload4(0, (__global int *)sum_col_addr);
#endif // defined(SUM_COL_HAS_BATCHES)

    a_offset_s32 *= (int4)A_OFFSET;
#endif // defined(A_OFFSET)

#if defined(B_OFFSET)
    // Compute the offset contribution due to A_OFFSET
    __global uchar *sum_row_addr = sum_row_ptr + sum_row_offset_first_element_in_bytes + y * sizeof(int);

    // Compute the offset contribution due to B_OFFSET
#if defined(HEIGHT_INPUT3D) && defined(DEPTH_INPUT3D)
    b_offset_s32 = (int4) * (((__global int *)(sum_row_addr + batch_id * sum_row_stride_y)) + (z % (int)DEPTH_INPUT3D) * (int)HEIGHT_INPUT3D);
#else  // defined(HEIGHT_INPUT3D) && defined(DEPTH_INPUT3D)
    b_offset_s32 = (int4) * (((__global int *)(sum_row_addr + batch_id * sum_row_stride_y)));
#endif // defined(HEIGHT_INPUT3D) && defined(DEPTH_INPUT3D)
    b_offset_s32 *= (int4)B_OFFSET;
#endif // defined(B_OFFSET)

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + x * sizeof(int);

    int4 biases_values = vload4(0, (__global int *)bias_addr);
    b_offset_s32 += (int4)biases_values;
#endif // defined(ADD_BIAS)

    return (int4)K_OFFSET + a_offset_s32 + b_offset_s32;
}

/* OpenCL kernel used to add the offset contribution after matrix multiplication. The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication),
 * and adds to it the offset contribution of matrix A and matrix B in-place.
 *
 * @attention The k_offset = a_offset * b_offset * k (where k is the number of matrix A columns) needs to be passed at compile time using -DK_OFFSET (i.e. -DK_OFFSET=1200)
 * @note In case the offset contribution due to a_offset is required, a_offset needs to be passed at compile time using -DA_OFFSET (i.e. -DA_OFFSET=1)
 * @note In case the offset contribution due to b_offset is required, b_offset needs to be passed at compile time using -DB_OFFSET (i.e. -DB_OFFSET=6)
 * @note In case sum_col has batches, -DSUM_COL_HAS_BATCHES must be passed at compile time. Usually if gemmlowp is used to accelerate convolution layer, sum_col will not have batches
 *
 * The final result is:
 *
 * mm_result[i][k] = mm_result[i][k] +
 *                   (sum_col[k] * A_OFFSET) +
 *                   (sum_row[i] * B_OFFSET) +
 *                   (K_OFFSET)
 *
 * @param[in] mm_result_ptr                           Pointer to the source tensor. Supported data type: S32
 * @param[in] mm_result_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in] mm_result_step_x                        mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] mm_result_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] mm_result_step_y                        mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] mm_result_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] mm_result_step_z                        mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] mm_result_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] sum_col_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_col_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_col_step_x                          (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_col_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_col_step_y                          (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_col_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in] sum_row_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_row_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_row_step_x                          (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_row_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_row_step_y                          (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_row_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in] biases_ptr                              (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in] biases_stride_x                         (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in] biases_step_x                           (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes    (Optional) The offset of the first element in the biases tensor
 */
__kernel void gemmlowp_offset_contribution(TENSOR3D_DECLARATION(mm_result)
#if defined(A_OFFSET)
                                           ,
                                           IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                                           ,
                                           IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
                                           ,
                                           VECTOR_DECLARATION(biases)
#endif // defined(ADD_BIAS))
                                          )
{
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Compute offset contribution
    int4 offset_term_s32 = offset_contribution(
                               x, y, z
#if defined(A_OFFSET)
                               ,
                               sum_col_ptr,
                               sum_col_stride_x,
                               sum_col_step_x,
                               sum_col_stride_y,
                               sum_col_step_y,
                               sum_col_offset_first_element_in_bytes
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                               ,
                               sum_row_ptr,
                               sum_row_stride_x,
                               sum_row_step_x,
                               sum_row_stride_y,
                               sum_row_step_y,
                               sum_row_offset_first_element_in_bytes
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
                               ,
                               biases_ptr,
                               biases_stride_x,
                               biases_step_x,
                               biases_offset_first_element_in_bytes
#endif // defined(ADD_BIAS)
                           );

    __global uchar *mm_result_addr = mm_result_ptr + mm_result_offset_first_element_in_bytes + x * sizeof(int) + y * mm_result_stride_y + z * mm_result_stride_z;

    int4 in_s32 = vload4(0, (__global int *)mm_result_addr);

    // Add the offset terms to GEMM's result
    in_s32 += offset_term_s32;

    // Store the result with the offset contribution
    vstore4(in_s32, 0, (__global int *)mm_result_addr);
}

#if defined(RESULT_OFFSET) && defined(RESULT_MULTIPLIER) && defined(RESULT_SHIFT) && defined(OUTPUT_DATA_TYPE)
/* OpenCL kernel used to add the offset contribution after @ref CLGEMMLowpMatrixMultiplyKernel and it quantizes down to uint8.
 *
 * This kernel takes a final int32 accumulator value (the output of @CLGEMMLowpMatrixMultiplyKernel), adds to it the offset contribution of matrix A and matrix B and quantizes to uint8 through the output stage.
 *
 *
 * @attention The k_offset = a_offset * b_offset * k (where k is the number of matrix A columns) needs to be passed at compile time using -DK_OFFSET (i.e. -DK_OFFSET=1200)
 * @note In case the offset contribution due to a_offset is required, a_offset needs to be passed at compile time using -DA_OFFSET (i.e. -DA_OFFSET=1)
 * @note In case the offset contribution due to b_offset is required, b_offset needs to be passed at compile time using -DB_OFFSET (i.e. -DB_OFFSET=6)
 * @note In case sum_col has batches, -DSUM_COL_HAS_BATCHES must be passed at compile time. Usually if gemmlowp is used to accelerate convolution layer, sum_col will not have batches
 *
 * The result before the output stage is:
 *
 * mm_result[i][k] = mm_result[i][k] +
 *                   (sum_col[k] * A_OFFSET) +
 *                   (sum_row[i] * B_OFFSET) +
 *                   (K_OFFSET)
 *
 * This result is quantized down to uint8/int8 using the output stage. The output stage computes the following operations:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result (if -DADD_BIAS is passed at compile time)
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds (if -DMIN_BOUND and/or -DMAX_BOUND are passed at compile time)
 *  -# Clamp the resulting int32 values:
 *      - to the [0..255] range and cast to QASYMM8.
 *      - to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  mm_result_ptr                                    Pointer to the source tensor. Supported data type: S32
 * @param[in]  mm_result_stride_x                               Stride of the source tensor in X dimension (in bytes)
 * @param[in]  mm_result_step_x                                 mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mm_result_stride_y                               Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  mm_result_step_y                                 mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  mm_result_stride_z                               Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  mm_result_step_z                                 mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mm_result_offset_first_element_in_bytes          The offset of the first element in the source tensor
 * @param[in]  sum_col_ptr                                      (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_col_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                                   (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                                   (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                                      (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_row_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                                   (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                                   (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  biases_ptr                                       (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                                  (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                                          Pointer to the destination tensor Supported data type: QASYMM8/QASYMM8_SIGNED
 * @param[in]  dst_stride_x                                     Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                                       dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                                     Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                                       dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                                     Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                                       src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes                The offset of the first element in the destination tensor
 * @param[in]  result_multipliers_ptr                           (Optional) Pointer to the output multipliers vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_multipliers_stride_x                      (Optional) Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in]  result_multipliers_step_x                        (Optional) output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_multipliers_offset_first_element_in_bytes (Optional) The offset of the first element in the output multipliers vector
 * @param[in]  result_shifts_ptr                                (Optional) Pointer to the output shifts vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_shifts_stride_x                           (Optional) Stride of the output shifts vector in X dimension (in bytes)
 * @param[in]  result_shifts_step_x                             (Optional) output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_shifts_offset_first_element_in_bytes      (Optional) The offset of the first element in the output shifts vector
 */
__kernel void gemmlowp_offset_contribution_quantize_down(TENSOR3D_DECLARATION(mm_result)
#if defined(A_OFFSET)
                                                         ,
                                                         IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                                                         ,
                                                         IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
                                                         ,
#if defined(ADD_BIAS)
                                                         VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                         TENSOR3D_DECLARATION(dst)
#if defined(PER_CHANNEL_QUANTIZATION)
                                                         ,
                                                         VECTOR_DECLARATION(result_multipliers),
                                                         VECTOR_DECLARATION(result_shifts)
#endif // defined(PER_CHANNEL_QUANTIZATION)
                                                        )
{
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x + y * dst_stride_y + z * dst_stride_z;

    // Compute offset contribution
    int4 offset_term_s32 = offset_contribution(
                               x, y, z
#if defined(A_OFFSET)
                               ,
                               sum_col_ptr,
                               sum_col_stride_x,
                               sum_col_step_x,
                               sum_col_stride_y,
                               sum_col_step_y,
                               sum_col_offset_first_element_in_bytes
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                               ,
                               sum_row_ptr,
                               sum_row_stride_x,
                               sum_row_step_x,
                               sum_row_stride_y,
                               sum_row_step_y,
                               sum_row_offset_first_element_in_bytes
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
                               ,
                               biases_ptr,
                               biases_stride_x,
                               biases_step_x,
                               biases_offset_first_element_in_bytes
#endif // defined(ADD_BIAS)
                           );

    __global uchar *mm_result_addr = mm_result_ptr + mm_result_offset_first_element_in_bytes + x * sizeof(int) + y * mm_result_stride_y + z * mm_result_stride_z;

    int4 in_s32 = vload4(0, (__global int *)mm_result_addr);

    // Add the offset terms to GEMM's result
    in_s32 += offset_term_s32;

    // -------------- OUTPUT STAGE

    // Add the offset terms to GEMM's result
    in_s32 += (int4)RESULT_OFFSET;

    // Multiply by result_mult_int and shift
#if defined(PER_CHANNEL_QUANTIZATION)
    __global uchar *result_multipliers_addr   = result_multipliers_ptr + result_multipliers_offset_first_element_in_bytes + x * sizeof(int);
    __global uchar *result_shifts_addr        = result_shifts_ptr + result_shifts_offset_first_element_in_bytes + x * sizeof(int);
    int4            result_multipliers_values = vload4(0, (__global int *)result_multipliers_addr);
    int4            result_shifts_values      = vload4(0, (__global int *)result_shifts_addr);

    in_s32 *= result_multipliers_values;
    in_s32 >>= result_shifts_values;
#else  // defined(PER_CHANNEL_QUANTIZATION)
    in_s32 *= RESULT_MULTIPLIER;

    in_s32 >>= RESULT_SHIFT;
#endif // defined(PER_CHANNEL_QUANTIZATION)

    VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4)
    res = CONVERT_SAT(in_s32, VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4));

#if defined(MIN_BOUND)
    res = max(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global OUTPUT_DATA_TYPE *)dst_addr);
}

/* OpenCL kernel used to add the offset contribution after matrix multiplication and it quantizes down to uint8.
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication), adds to it the offset contribution of matrix A and matrix B and quantizes to uint8 through the output stage.
 *
 *
 * @attention The k_offset = a_offset * b_offset * k (where k is the number of matrix A columns) needs to be passed at compile time using -DK_OFFSET (i.e. -DK_OFFSET=1200)
 * @note In case the offset contribution due to a_offset is required, a_offset needs to be passed at compile time using -DA_OFFSET (i.e. -DA_OFFSET=1)
 * @note In case the offset contribution due to b_offset is required, b_offset needs to be passed at compile time using -DB_OFFSET (i.e. -DB_OFFSET=6)
 * @note In case sum_col has batches, -DSUM_COL_HAS_BATCHES must be passed at compile time. Usually if gemmlowp is used to accelerate convolution layer, sum_col will not have batches
 *
 * The result before the output stage is:
 *
 * mm_result[i][k] = mm_result[i][k] +
 *                   (sum_col[k] * A_OFFSET) +
 *                   (sum_row[i] * B_OFFSET) +
 *                   (K_OFFSET)
 *
 * This result is quantized down to uint8/int8 using the output stage. The output stage computes the following operations:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values:
 *      - to the [0..255] range and cast to QASYMM8.
 *      - to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  mm_result_ptr                                    Pointer to the source tensor. Supported data type: S32
 * @param[in]  mm_result_stride_x                               Stride of the source tensor in X dimension (in bytes)
 * @param[in]  mm_result_step_x                                 mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mm_result_stride_y                               Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  mm_result_step_y                                 mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  mm_result_stride_z                               Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  mm_result_step_z                                 mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mm_result_offset_first_element_in_bytes          The offset of the first element in the source tensor
 * @param[in]  sum_col_ptr                                      (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_col_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                                   (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                                   (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                                      (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_row_stride_x                                 (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                                   (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                                 (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                                   (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes            (Optional) The offset of the first element in the source tensor
 * @param[in]  biases_ptr                                       (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                                  (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                                          Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                                     Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                                       dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                                     Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                                       dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                                     Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                                       src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes                The offset of the first element in the destination tensor
 * @param[in]  result_multipliers_ptr                           (Optional) Pointer to the output multipliers vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_multipliers_stride_x                      (Optional) Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in]  result_multipliers_step_x                        (Optional) output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_multipliers_offset_first_element_in_bytes (Optional) The offset of the first element in the output multipliers vector
 * @param[in]  result_shifts_ptr                                (Optional) Pointer to the output shifts vector for per-channel quantization. Supported data types: S32
 * @param[in]  result_shifts_stride_x                           (Optional) Stride of the output shifts vector in X dimension (in bytes)
 * @param[in]  result_shifts_step_x                             (Optional) output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  result_shifts_offset_first_element_in_bytes      (Optional) The offset of the first element in the output shifts vector
 */
__kernel void gemmlowp_offset_contribution_quantize_down_fixedpoint(TENSOR3D_DECLARATION(mm_result)
#if defined(A_OFFSET)
                                                                    ,
                                                                    IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                                                                    ,
                                                                    IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
                                                                    ,
#if defined(ADD_BIAS)
                                                                    VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                                    TENSOR3D_DECLARATION(dst)
#if defined(PER_CHANNEL_QUANTIZATION)
                                                                    ,
                                                                    VECTOR_DECLARATION(result_multipliers),
                                                                    VECTOR_DECLARATION(result_shifts)
#endif // defined(PER_CHANNEL_QUANTIZATION)
                                                                   )
{
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Compute offset contribution
    int4 offset_term_s32 = offset_contribution(
                               x, y, z
#if defined(A_OFFSET)
                               ,
                               sum_col_ptr,
                               sum_col_stride_x,
                               sum_col_step_x,
                               sum_col_stride_y,
                               sum_col_step_y,
                               sum_col_offset_first_element_in_bytes
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                               ,
                               sum_row_ptr,
                               sum_row_stride_x,
                               sum_row_step_x,
                               sum_row_stride_y,
                               sum_row_step_y,
                               sum_row_offset_first_element_in_bytes
#endif // defined(B_OFFSET)
#if defined(ADD_BIAS)
                               ,
                               biases_ptr,
                               biases_stride_x,
                               biases_step_x,
                               biases_offset_first_element_in_bytes
#endif // defined(ADD_BIAS)
                           );

    __global uchar *mm_result_addr = mm_result_ptr + mm_result_offset_first_element_in_bytes + x * sizeof(int) + y * mm_result_stride_y + z * mm_result_stride_z;

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x + y * dst_stride_y + z * dst_stride_z;

    int4 in_s32 = vload4(0, (__global int *)mm_result_addr);

    // Add the offset terms to GEMM's result
    in_s32 += offset_term_s32;

    // -------------- OUTPUT STAGE

    // Multiply by result_mult_int and shift
#if defined(PER_CHANNEL_QUANTIZATION)
    __global uchar *result_multipliers_addr   = result_multipliers_ptr + result_multipliers_offset_first_element_in_bytes + x * sizeof(int);
    __global uchar *result_shifts_addr        = result_shifts_ptr + result_shifts_offset_first_element_in_bytes + x * sizeof(int);
    int4            result_multipliers_values = vload4(0, (__global int *)result_multipliers_addr);
    int4            result_shifts_values      = vload4(0, (__global int *)result_shifts_addr);

    int4 in_s32_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(in_s32, result_multipliers_values, result_shifts_values, 4);
    int4 in_s32_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(in_s32, result_multipliers_values, result_shifts_values, 4);
    in_s32                = select(in_s32_shift_lt0, in_s32_shift_gt0, result_shifts_values >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)

#if RESULT_SHIFT < 0
    in_s32 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(in_s32, RESULT_MULTIPLIER, RESULT_SHIFT, 4);
#else  // RESULT_SHIFT >= 0
    in_s32 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(in_s32, RESULT_MULTIPLIER, RESULT_SHIFT, 4);
#endif // RESULT_SHIFT < 0

#endif // defined(PER_CHANNEL_QUANTIZATION)

    // Add the offset terms to GEMM's result
    in_s32 += (int4)RESULT_OFFSET;

    VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4)
    res = CONVERT_SAT(in_s32, VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4));

#if defined(MIN_BOUND)
    res = max(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global OUTPUT_DATA_TYPE *)dst_addr);
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_MULTIPLIER) && defined(RESULT_SHIFT) && defined(OUTPUT_DATA_TYPE)

#endif // defined(K_OFFSET)

#if defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8/QASYMM8_SIGNED
 *
 * This kernel takes a final int32 accumulator value and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result (if -DADD_BIAS is passed at compile time)
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds (if -DMIN_BOUND and/or -DMAX_BOUND are passed at compile time)
 *  -# Clamp the resulting int32 values:
 *  -#  - to the [0..255] range and cast to QASYMM8.
 *  -#  - to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8/QASYMM8_SIGNED
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                  VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                  TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(int) + y * src_stride_y + z * src_stride_z;

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x + y * dst_stride_y + z * dst_stride_z;

    int4 input_values = vload4(0, (__global int *)src_addr);

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + x * sizeof(int);

    int4 biases_values = vload4(0, (__global int *)bias_addr);
    input_values += (int4)biases_values;
#endif // defined(ADD_BIAS)

    // Add the offset terms to GEMM's result
    input_values += (int4)RESULT_OFFSET;

    // Multiply by result_mult_int and shift
    input_values *= RESULT_MULT_INT;

#if RESULT_SHIFT < 0
    input_values >>= -RESULT_SHIFT;
#else  // RESULT_SHIFT >= 0
    input_values >>= RESULT_SHIFT;
#endif // RESULT_SHIFT < 0

    VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4)
    res = CONVERT_SAT(input_values, VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4));

#if defined(MIN_BOUND)
    res = max(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global OUTPUT_DATA_TYPE *)dst_addr);
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)

#if defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8/QASYMM8_SIGNED
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication), and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values:
 *      - to the [0..255] range and cast to QASYMM8.
 *      - to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET_AFTER_SHIFT, -DRESULT_FIXEDPOINT_MULTIPLIER and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8/QASYMM8_SIGNED
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down_fixedpoint(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                             VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                             TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(int) + y * src_stride_y + z * src_stride_z;

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x + y * dst_stride_y + z * dst_stride_z;

    int4 input_values = vload4(0, (__global int *)src_addr);

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + x * sizeof(int);

    int4 biases_values = vload4(0, (__global int *)bias_addr);
    input_values += (int4)biases_values;
#endif // defined(ADD_BIAS)

    // Multiply by result_mult_int and shift
#if RESULT_SHIFT < 0
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);
#else  // RESULT_SHIFT >= 0
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);
#endif // RESULT_SHIFT < 0

    // Add the offset terms to GEMM's result
    input_values += (int4)RESULT_OFFSET_AFTER_SHIFT;

    VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4)
    res = CONVERT_SAT(input_values, VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4));

#if defined(MIN_BOUND)
    res = max(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global OUTPUT_DATA_TYPE *)dst_addr);
}
#endif // defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)

#if defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)

/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QSYMM16
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication), and processes it to obtain the final QSYMM16 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [-32768..32767] range and cast to QSYMM16.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_FIXEDPOINT_MULTIPLIER and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QSYMM16
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down_fixedpoint_qsymm16(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                                     VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                                     TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(int) + y * src_stride_y + z * src_stride_z;

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * 2 + y * dst_stride_y + z * dst_stride_z;

    int4 input_values = vload4(0, (__global int *)src_addr);

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + x * sizeof(int);

    int4 biases_values = vload4(0, (__global int *)bias_addr);
    input_values += (int4)biases_values;
#endif // defined(ADD_BIAS)

    // Multiply by result_mult_int and shift
#if RESULT_SHIFT < 0
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);
#else  // RESULT_SHIFT >= 0
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);
#endif // RESULT_SHIFT < 0

    short4 res = convert_short4_sat(input_values);

#if defined(MIN_BOUND)
    res = max(res, (short4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (short4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global short *)dst_addr);
}
#endif // defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)

#if defined(REAL_MULTIPLIER) && defined(OUTPUT_OFFSET)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8/QASYMM8_SIGNED
 *
 * This kernel takes a final int32 accumulator value (the output of matrix multiplication), and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Requantize
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values:
 *      - to the [0..255] range and cast to QASYMM8.
 *      - to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 * @attention The offset and scalar scale factor must be passed at compile time using -DRESULT_OFFSET, -DREAL_MULTIPLIER
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note The output datatype should be passed at compile time using -DOUTPUT_DATA_TYPE
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                         Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                           src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down_float(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                        VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
#if defined(DST_HEIGHT)
                                                        TENSOR4D_DECLARATION(dst))
#else  // defined(DST_HEIGHT)
                                                        TENSOR3D_DECLARATION(dst))
#endif // defined(DST_HEIGHT)
{
    // Compute source and destination addresses
    int x = get_global_id(0) * 4;
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(int) + y * src_stride_y + z * src_stride_z;

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x + y * dst_stride_y + z * dst_stride_z;

    int4 input_values = vload4(0, (__global int *)src_addr);

#if defined(ADD_BIAS)
    // Add bias
    __global uchar *bias_addr = biases_ptr + biases_offset_first_element_in_bytes + x * sizeof(int);

    int4 biases_values = vload4(0, (__global int *)bias_addr);
    input_values += (int4)biases_values;
#endif // defined(ADD_BIAS)

    // Convert to float
    float4 input_values_f = convert_float4(input_values);
    input_values_f        = round(input_values_f * (float)REAL_MULTIPLIER + (float)OUTPUT_OFFSET);

    VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4)
    res = CONVERT_SAT(input_values_f, VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4));

#if defined(MIN_BOUND)
    res = max(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (VEC_DATA_TYPE(OUTPUT_DATA_TYPE, 4))MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, (__global OUTPUT_DATA_TYPE *)dst_addr);
}
#endif // defined(REAL_MULTIPLIER) && defined(OUTPUT_OFFSET)
