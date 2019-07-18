/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), (val));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

/** Specialized macros to perform the dot product instruction between two vectors of size N [1,16]. These macros use the dot8 instruction */
#define ARM_DOT1(a, b, c)                                           \
    ({                                                              \
        ARM_DOT((uchar4)(a, (uchar3)0), (uchar4)(b, (uchar3)0), c); \
    })
#define ARM_DOT2(a, b, c)                                           \
    ({                                                              \
        ARM_DOT((uchar4)(a, (uchar2)0), (uchar4)(b, (uchar2)0), c); \
    })
#define ARM_DOT3(a, b, c)                                         \
    ({                                                            \
        ARM_DOT((uchar4)(a, (uchar)0), (uchar4)(b, (uchar)0), c); \
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
#define ARM_DOT1(a, b, c) \
    ({                    \
        c += (uint)a * b; \
    })
#define ARM_DOT2(a, b, c)       \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
    })
#define ARM_DOT3(a, b, c)       \
    ({                          \
        ARM_DOT2(a, b, c);      \
        c += (uint)a.s2 * b.s2; \
    })
#define ARM_DOT4(a, b, c)       \
    ({                          \
        ARM_DOT3(a, b, c);      \
        c += (uint)a.s3 * b.s3; \
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

/** Specialized macros to perform a a partial matrix multiplication with dimensions M0,N0,K0 */
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

#if defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && defined(COLS_A)
#define VECTOR_UCHAR VEC_DATA_TYPE(uchar, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_UINT VEC_DATA_TYPE(uint, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_INT VEC_DATA_TYPE(int, NUM_ELEMS_PROCESSED_PER_THREAD_X)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type: QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the output tensor (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_midgard(IMAGE_DECLARATION(src0),
                                  IMAGE_DECLARATION(src1),
                                  IMAGE_DECLARATION(dst),
                                  uint src0_stride_z,
                                  uint src1_stride_z,
                                  uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                  ,
                                  uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                  ,
                                  uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                 )
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx;

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    int end_row_vec_a = src_addr.s0 + COLS_A;

    VECTOR_UINT acc0 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VECTOR_UINT acc1 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VECTOR_UINT acc2 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VECTOR_UINT acc3 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    VECTOR_UINT acc4 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

    for(; src_addr.s0 <= (end_row_vec_a - 2); src_addr += (int2)(2, 2 * src1_stride_y))
    {
        // Load values from matrix A
        uchar2 a0 = vload2(0, src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar2 a1 = vload2(0, src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar2 a2 = vload2(0, src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar2 a3 = vload2(0, src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar2 a4 = vload2(0, src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        VECTOR_UCHAR b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1);
        VECTOR_UCHAR b1 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1 + src1_stride_y);

        // Accumulate
        acc0 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a0.s0;
        acc0 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a0.s1;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a1.s0;
        acc1 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a1.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a2.s0;
        acc2 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a2.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a3.s0;
        acc3 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a3.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        acc4 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a4.s0;
        acc4 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a4.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(1, src1_stride_y))
    {
        // Load values from matrix A
        uchar a0 = *(src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar a1 = *(src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar a2 = *(src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar a3 = *(src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar a4 = *(src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        VECTOR_UCHAR b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1);

        // Accumulate
        acc0 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a2;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        acc4 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a4;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    const int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint8 zout = ((uint8)(0, 1, 2, 3, 4, 5, 6, 7) + (uint8)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint8)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst.ptr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the result
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc0, VECTOR_INT), 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc1, VECTOR_INT), 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc2, VECTOR_INT), 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc3, VECTOR_INT), 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc4, VECTOR_INT), 0, (__global int *)(dst.ptr + 4 * dst_stride_y + zout.s4));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst.ptr += z * dst_stride_z;

    // Store the result
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc0, VECTOR_INT), 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc1, VECTOR_INT), 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc2, VECTOR_INT), 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc3, VECTOR_INT), 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc4, VECTOR_INT), 0, (__global int *)(dst.ptr + 4 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}
#endif // defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && defined(COLS_A)

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(M) && defined(N)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices with QASYMM data type.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
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
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as @p lhs_ptr
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
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zrhs, 0);

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(uint, N0), c, 0); //VEC_DATA_TYPE(uint, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    for(int i = 0; i < k; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, uchar, a, lhs_addr, 0, LHS_STEP_X, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, uchar, b, rhs_addr, 0, RHS_STEP_X, zrhs);

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
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: F16/F32
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
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as @p lhs_ptr
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
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(uint, N0), c, 0); //VEC_DATA_TYPE(uint, N0)    c0=0,c1=0,c2=0,... c(N0-1)=0;

    for(int i = 0; i < K; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, uchar, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, uchar, b, rhs_ptr, rhs_offset, RHS_STEP_X, zrhs);

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
#endif // defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE) && defined(K)

#if defined(M0) && defined(N0) && defined(K0) && defined(K)

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is NOT reshaped
 *
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
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: F16/F32
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
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as @p lhs_ptr
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
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(uint, N0), c, 0); //VEC_DATA_TYPE(uint, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;

    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, uchar, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(K0, N0, uchar, b, rhs_ptr, rhs_offset, rhs_stride_y, zrhs);

        // Transpose the values from RHS matrix
        TRANSPOSE_K0XN0(K0, N0, b_t, b);

        // Partial matrix multiplication M0,N0,K0
        ARM_MM_K0XN0XM0(M0, N0, K0, a, b_t, c);

        // Update the offset
        lhs_offset += K0;
        rhs_offset += K0 * rhs_stride_y;
    }

    // Left-over for loop
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, 1, uchar, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(1, N0, uchar, b, rhs_ptr, rhs_offset, rhs_stride_y, zrhs);

        // Transpose the values from RHS matrix
        TRANSPOSE_K0XN0(1, N0, b_t, b);

        // Partial matrix multiplication M0,N0,1
        ARM_MM_K0XN0XM0(M0, N0, 1, a, b_t, c);

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
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8
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

    uint4 sum_row_u32 = (uint4)0;
    uint  sum_row     = 0;

    __global const uchar *matrix_a = (__global const uchar *)(src.ptr + get_global_id(0) * src_stride_y + get_global_id(1) * src_stride_z);

    int i = 0;

    // This for loop performs 16 accumulations
    for(; i <= ((int)COLS_A - 16); i += 16)
    {
        const uchar16 a0_u8 = vload16(0, matrix_a + i);

        sum_row_u32 += convert_uint4(a0_u8.s0123) + convert_uint4(a0_u8.s4567) + convert_uint4(a0_u8.s89AB) + convert_uint4(a0_u8.sCDEF);
    }

    // This for loop performs the leftover accumulations
    for(; i < COLS_A; ++i)
    {
        sum_row += matrix_a[i];
    }

    sum_row += sum_row_u32.s0 + sum_row_u32.s1 + sum_row_u32.s2 + sum_row_u32.s3;

    *((__global int *)dst.ptr) = (int)sum_row;
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A using the arm dot product instruction
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8
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

    uint sum_row = 0;

    __global const uchar *matrix_a = (__global const uchar *)(src.ptr + get_global_id(0) * src_stride_y + get_global_id(1) * src_stride_z);

    int i = 0;

    // This for loop performs 16 accumulations
    for(; i <= ((int)COLS_A - 32); i += 32)
    {
        uchar16 a0_u8 = vload16(0, matrix_a + i);

        sum_row += arm_dot(a0_u8.s0123, (uchar4)(1));
        sum_row += arm_dot(a0_u8.s4567, (uchar4)(1));
        sum_row += arm_dot(a0_u8.s89AB, (uchar4)(1));
        sum_row += arm_dot(a0_u8.sCDEF, (uchar4)(1));

        a0_u8 = vload16(1, matrix_a + i);

        sum_row += arm_dot(a0_u8.s0123, (uchar4)(1));
        sum_row += arm_dot(a0_u8.s4567, (uchar4)(1));
        sum_row += arm_dot(a0_u8.s89AB, (uchar4)(1));
        sum_row += arm_dot(a0_u8.sCDEF, (uchar4)(1));
    }

    // This for loop performs the leftover accumulations
    for(; i < COLS_A; ++i)
    {
        sum_row += matrix_a[i];
    }

    *((__global int *)dst.ptr) = (int)sum_row;
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#endif // defined(COLS_A)

#if defined(COLS_B) && defined(ROWS_B)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix B columns and rows needs to be passed at compile time using -DCOLS_B and -DROWS_B
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8
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

    uint16 sum_col_u32 = (uint16)0;

    __global const uchar *matrix_b = (__global const uchar *)(src.ptr + get_global_id(1) * src_stride_z);

    int i = 0;
    // This for loop performs 4 accumulations
    for(; i <= ((int)ROWS_B - 4); i += 4)
    {
        const uchar16 b0_u8 = vload16(0, matrix_b + 0 * src_stride_y);
        const uchar16 b1_u8 = vload16(0, matrix_b + 1 * src_stride_y);
        const uchar16 b2_u8 = vload16(0, matrix_b + 2 * src_stride_y);
        const uchar16 b3_u8 = vload16(0, matrix_b + 3 * src_stride_y);

        sum_col_u32 += convert_uint16(b0_u8) + convert_uint16(b1_u8) + convert_uint16(b2_u8) + convert_uint16(b3_u8);

        matrix_b += 4 * src_stride_y;
    }

    // This for loop perfoms the leftover accumulations
    for(; i < (int)ROWS_B; ++i)
    {
        const uchar16 b0_u8 = vload16(0, matrix_b);

        sum_col_u32 += convert_uint16(b0_u8);

        matrix_b += src_stride_y;
    }

    vstore16(convert_int16(sum_col_u32), 0, (__global int *)dst.ptr);
}
#endif // defined(COLS_B) && defined(ROWS_B)

#if defined(K_OFFSET)

/* Helper function used to calculate the offset contribution after @ref CLGEMMLowpMatrixMultiplyKernel.
 *
 * This kernel takes a final int32 accumulator value (the output of @CLGEMMLowpMatrixMultiplyKernel),
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

/* OpenCL kernel used to add the offset contribution after @ref CLGEMMLowpMatrixMultiplyKernel. The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of @CLGEMMLowpMatrixMultiplyKernel),
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

#if defined(RESULT_OFFSET) && defined(RESULT_MULTIPLIER) && defined(RESULT_SHIFT)
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
 * This result is quantized down to uint8 using the output stage. The output stage computes the following operations:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result (if -DADD_BIAS is passed at compile time)
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds (if -DMIN_BOUND and/or -DMAX_BOUND are passed at compile time)
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  mm_result_ptr                           Pointer to the source tensor. Supported data type: S32
 * @param[in]  mm_result_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  mm_result_step_x                        mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mm_result_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  mm_result_step_y                        mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  mm_result_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  mm_result_step_z                        mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mm_result_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_col_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_col_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                          (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                          (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_row_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                          (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                          (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in]  biases_ptr                              (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                         (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                           (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes    (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                                 Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                            Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                              dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                            Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                              dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                            Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                              src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes       The offset of the first element in the destination tensor
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
                                                         TENSOR3D_DECLARATION(dst))
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
    in_s32 *= RESULT_MULTIPLIER;

    in_s32 >>= RESULT_SHIFT;

    uchar4 res = convert_uchar4_sat(in_s32);

#if defined(MIN_BOUND)
    res = max(res, (uchar4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, dst_addr);
}

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
 * This result is quantized down to uint8 using the output stage. The output stage computes the following operations:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  mm_result_ptr                           Pointer to the source tensor. Supported data type: S32
 * @param[in]  mm_result_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  mm_result_step_x                        mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mm_result_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  mm_result_step_y                        mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  mm_result_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  mm_result_step_z                        mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mm_result_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_col_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_col_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                          (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                          (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                             (Optional) Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in]  sum_row_stride_x                        (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                          (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                        (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                          (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes   (Optional) The offset of the first element in the source tensor
 * @param[in]  biases_ptr                              (Optional) Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                         (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                           (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes    (Optional) The offset of the first element in the biases tensor
 * @param[out] dst_ptr                                 Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                            Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                              dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                            Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                              dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                            Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                              src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes       The offset of the first element in the destination tensor
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
                                                                    TENSOR3D_DECLARATION(dst))
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
    in_s32 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(in_s32, RESULT_MULTIPLIER, RESULT_SHIFT, 4);

    // Add the offset terms to GEMM's result
    in_s32 += (int4)RESULT_OFFSET;

    uchar4 res = convert_uchar4_sat(in_s32);

#if defined(MIN_BOUND)
    res = max(res, (uchar4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, dst_addr);
}
#endif // defined(K_OFFSET) && defined(RESULT_OFFSET) && defined(RESULT_MULTIPLIER) && defined(RESULT_SHIFT)
#endif // defined(K_OFFSET)

#if defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result (if -DADD_BIAS is passed at compile time)
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds (if -DMIN_BOUND and/or -DMAX_BOUND are passed at compile time)
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
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
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
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

    input_values >>= RESULT_SHIFT;

    uchar4 res = convert_uchar4_sat(input_values);

#if defined(MIN_BOUND)
    res = max(res, (uchar4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, dst_addr);
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)

#if defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET_AFTER_SHIFT, -DRESULT_FIXEDPOINT_MULTIPLIER and -DRESULT_SHIFT
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
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
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
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);

    // Add the offset terms to GEMM's result
    input_values += (int4)RESULT_OFFSET_AFTER_SHIFT;

    uchar4 res = convert_uchar4_sat(input_values);

#if defined(MIN_BOUND)
    res = max(res, (uchar4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, dst_addr);
}
#endif // defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)

#if defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)

/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QSYMM16
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QSYMM16 value.
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
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
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
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 4);

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
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Requantize
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset and scalar scale factor must be passed at compile time using -DRESULT_OFFSET, -DREAL_MULTIPLIER
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
    float16 input_values_f = convert_float4(input_values);
    input_values_f         = round(input_values_f * (float)REAL_MULTIPLIER + (float)OUTPUT_OFFSET);

    uchar4 res = convert_uchar4_sat(input_values_f);

#if defined(MIN_BOUND)
    res = max(res, (uchar4)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar4)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore4(res, 0, dst_addr);
}
#endif // defined(REAL_MULTIPLIER) && defined(OUTPUT_OFFSET)
