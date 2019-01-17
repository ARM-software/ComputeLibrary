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
#include "helpers.h"
#include "helpers_asymm.h"
#include "repeat.h"

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), (val));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(COLS_B) && defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(TRANSPOSE1XW_WIDTH_STEP)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel before running the matrix multiplication
 *
 * @note The number of matrix B columns needs to be passed at compile time using -DCOLS_B: e.g. -DCOLS_B=1024
 * @note The transposition width step (mult_transpose1xW_width * 4) must be passed at compile time using -DTRANSPOSE1XW_WIDTH_STEP (i.e. -DTRANSPOSE1XW_WIDTH_STEP=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
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
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_interleaved_transposed_midgard(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    const int x = get_global_id(0) / TRANSPOSE1XW_WIDTH_STEP;
    const int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    const int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % TRANSPOSE1XW_WIDTH_STEP) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global uchar *src_addr_a = (__global uchar *)(src0_ptr + z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes);
    __global uchar *src_addr_b = (__global uchar *)(src1_ptr + x * src1_stride_y + src1_offset_first_element_in_bytes);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr_b += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr_b += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute end row address for matrix B
    __global uchar *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    int4 c00 = 0;
    int4 c10 = 0;
    int4 c20 = 0;
    int4 c30 = 0;

    for(; src_addr_b <= (src_end_addr_b - (int)(8 * TRANSPOSE1XW_WIDTH_STEP)); src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 8 * TRANSPOSE1XW_WIDTH_STEP)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int4 a0 = convert_int4(vload4(0, src_addr_a));
        int4 b0 = convert_int4(vload4(0, src_addr_b));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;

        a0 = convert_int4(vload4(0, src_addr_a + 4 * MULT_INTERLEAVE4X4_HEIGHT));
        b0 = convert_int4(vload4(0, src_addr_b + 4 * TRANSPOSE1XW_WIDTH_STEP));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += (4 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (4 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int4 a0 = convert_int4(vload4(0, src_addr_a));
        int4 b0 = convert_int4(vload4(0, src_addr_b));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;
    }

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

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst.ptr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x4 block
    vstore4(c00, 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
    vstore4(c10, 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
    vstore4(c20, 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
    vstore4(c30, 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst.ptr += z * dst_stride_z;

    // Store 4x4 block
    vstore4(c00, 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
    vstore4(c10, 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
    vstore4(c20, 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
    vstore4(c30, 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

/** This OpenCL kernel is optimized for Bifrost and computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel before running the matrix multiplication
 *
 * @attention The number of matrix B columns needs to be passed at compile time using -DCOLS_B
 * @note The transposition width step (mult_transpose1xW_width * 4) must be passed at compile time using -DTRANSPOSE1XW_WIDTH_STEP (i.e. -DTRANSPOSE1XW_WIDTH_STEP=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
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
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_interleaved_transposed_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    const int x = get_global_id(0) / TRANSPOSE1XW_WIDTH_STEP;
    const int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    const int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % TRANSPOSE1XW_WIDTH_STEP) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global uchar *src_addr_a = (__global uchar *)(src0_ptr + z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes);
    __global uchar *src_addr_b = (__global uchar *)(src1_ptr + x * src1_stride_y + src1_offset_first_element_in_bytes);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr_b += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr_b += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute end row address for matrix B
    __global uchar *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    uint c00 = 0;
    uint c01 = 0;
    uint c02 = 0;
    uint c03 = 0;
    uint c10 = 0;
    uint c11 = 0;
    uint c12 = 0;
    uint c13 = 0;
    uint c20 = 0;
    uint c21 = 0;
    uint c22 = 0;
    uint c23 = 0;
    uint c30 = 0;
    uint c31 = 0;
    uint c32 = 0;
    uint c33 = 0;

#if MULT_INTERLEAVE4X4_HEIGHT == 1
    for(; src_addr_b <= (src_end_addr_b - (int)(32 * TRANSPOSE1XW_WIDTH_STEP)); src_addr_a += (32 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (32 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar16 a0 = vload16(0, src_addr_a);
        uchar4  b0 = vload4(0, src_addr_b);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 4 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s4 * b0.s0;
        c01 += (ushort)a0.s4 * b0.s1;
        c02 += (ushort)a0.s4 * b0.s2;
        c03 += (ushort)a0.s4 * b0.s3;

        c10 += (ushort)a0.s5 * b0.s0;
        c11 += (ushort)a0.s5 * b0.s1;
        c12 += (ushort)a0.s5 * b0.s2;
        c13 += (ushort)a0.s5 * b0.s3;

        c20 += (ushort)a0.s6 * b0.s0;
        c21 += (ushort)a0.s6 * b0.s1;
        c22 += (ushort)a0.s6 * b0.s2;
        c23 += (ushort)a0.s6 * b0.s3;

        c30 += (ushort)a0.s7 * b0.s0;
        c31 += (ushort)a0.s7 * b0.s1;
        c32 += (ushort)a0.s7 * b0.s2;
        c33 += (ushort)a0.s7 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 8 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s8 * b0.s0;
        c01 += (ushort)a0.s8 * b0.s1;
        c02 += (ushort)a0.s8 * b0.s2;
        c03 += (ushort)a0.s8 * b0.s3;

        c10 += (ushort)a0.s9 * b0.s0;
        c11 += (ushort)a0.s9 * b0.s1;
        c12 += (ushort)a0.s9 * b0.s2;
        c13 += (ushort)a0.s9 * b0.s3;

        c20 += (ushort)a0.sA * b0.s0;
        c21 += (ushort)a0.sA * b0.s1;
        c22 += (ushort)a0.sA * b0.s2;
        c23 += (ushort)a0.sA * b0.s3;

        c30 += (ushort)a0.sB * b0.s0;
        c31 += (ushort)a0.sB * b0.s1;
        c32 += (ushort)a0.sB * b0.s2;
        c33 += (ushort)a0.sB * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 12 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.sC * b0.s0;
        c01 += (ushort)a0.sC * b0.s1;
        c02 += (ushort)a0.sC * b0.s2;
        c03 += (ushort)a0.sC * b0.s3;

        c10 += (ushort)a0.sD * b0.s0;
        c11 += (ushort)a0.sD * b0.s1;
        c12 += (ushort)a0.sD * b0.s2;
        c13 += (ushort)a0.sD * b0.s3;

        c20 += (ushort)a0.sE * b0.s0;
        c21 += (ushort)a0.sE * b0.s1;
        c22 += (ushort)a0.sE * b0.s2;
        c23 += (ushort)a0.sE * b0.s3;

        c30 += (ushort)a0.sF * b0.s0;
        c31 += (ushort)a0.sF * b0.s1;
        c32 += (ushort)a0.sF * b0.s2;
        c33 += (ushort)a0.sF * b0.s3;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload16(0, src_addr_a + 16);
        b0 = vload4(0, src_addr_b + 16 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 20 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s4 * b0.s0;
        c01 += (ushort)a0.s4 * b0.s1;
        c02 += (ushort)a0.s4 * b0.s2;
        c03 += (ushort)a0.s4 * b0.s3;

        c10 += (ushort)a0.s5 * b0.s0;
        c11 += (ushort)a0.s5 * b0.s1;
        c12 += (ushort)a0.s5 * b0.s2;
        c13 += (ushort)a0.s5 * b0.s3;

        c20 += (ushort)a0.s6 * b0.s0;
        c21 += (ushort)a0.s6 * b0.s1;
        c22 += (ushort)a0.s6 * b0.s2;
        c23 += (ushort)a0.s6 * b0.s3;

        c30 += (ushort)a0.s7 * b0.s0;
        c31 += (ushort)a0.s7 * b0.s1;
        c32 += (ushort)a0.s7 * b0.s2;
        c33 += (ushort)a0.s7 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 24 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s8 * b0.s0;
        c01 += (ushort)a0.s8 * b0.s1;
        c02 += (ushort)a0.s8 * b0.s2;
        c03 += (ushort)a0.s8 * b0.s3;

        c10 += (ushort)a0.s9 * b0.s0;
        c11 += (ushort)a0.s9 * b0.s1;
        c12 += (ushort)a0.s9 * b0.s2;
        c13 += (ushort)a0.s9 * b0.s3;

        c20 += (ushort)a0.sA * b0.s0;
        c21 += (ushort)a0.sA * b0.s1;
        c22 += (ushort)a0.sA * b0.s2;
        c23 += (ushort)a0.sA * b0.s3;

        c30 += (ushort)a0.sB * b0.s0;
        c31 += (ushort)a0.sB * b0.s1;
        c32 += (ushort)a0.sB * b0.s2;
        c33 += (ushort)a0.sB * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 28 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.sC * b0.s0;
        c01 += (ushort)a0.sC * b0.s1;
        c02 += (ushort)a0.sC * b0.s2;
        c03 += (ushort)a0.sC * b0.s3;

        c10 += (ushort)a0.sD * b0.s0;
        c11 += (ushort)a0.sD * b0.s1;
        c12 += (ushort)a0.sD * b0.s2;
        c13 += (ushort)a0.sD * b0.s3;

        c20 += (ushort)a0.sE * b0.s0;
        c21 += (ushort)a0.sE * b0.s1;
        c22 += (ushort)a0.sE * b0.s2;
        c23 += (ushort)a0.sE * b0.s3;

        c30 += (ushort)a0.sF * b0.s0;
        c31 += (ushort)a0.sF * b0.s1;
        c32 += (ushort)a0.sF * b0.s2;
        c33 += (ushort)a0.sF * b0.s3;
    }
#endif // MULT_INTERLEAVE4X4_HEIGHT == 1

    for(; src_addr_b < src_end_addr_b; src_addr_a += (4 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (4 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar4 a0 = vload4(0, src_addr_a);
        uchar4 b0 = vload4(0, src_addr_b);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;
    }

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

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst.ptr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x4 block
    vstore4((int4)(c00, c01, c02, c03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
    vstore4((int4)(c10, c11, c12, c13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
    vstore4((int4)(c20, c21, c22, c23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
    vstore4((int4)(c30, c31, c32, c33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst.ptr += z * dst_stride_z;

    // Store 4x4 block
    vstore4((int4)(c00, c01, c02, c03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
    vstore4((int4)(c10, c11, c12, c13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
    vstore4((int4)(c20, c21, c22, c23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
    vstore4((int4)(c30, c31, c32, c33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
/** This OpenCL kernel is optimized for Bifrost and computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel before running the matrix multiplication
 *
 * @attention The number of matrix B columns needs to be passed at compile time using -DCOLS_B
 * @note The transposition width step (mult_transpose1xW_width * 4) must be passed at compile time using -DTRANSPOSE1XW_WIDTH_STEP (i.e. -DTRANSPOSE1XW_WIDTH_STEP=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
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
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_interleaved_transposed_bifrost_dot8(IMAGE_DECLARATION(src0),
                                                              IMAGE_DECLARATION(src1),
                                                              IMAGE_DECLARATION(dst),
                                                              uint src0_stride_z,
                                                              uint src1_stride_z,
                                                              uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                              ,
                                                              uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                             )
{
    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % TRANSPOSE1XW_WIDTH_STEP) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global uchar *src_addr_a = (__global uchar *)(src0_ptr + (get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT) * src0_stride_y + get_global_id(2) * src0_stride_z + src0_offset_first_element_in_bytes);
    __global uchar *src_addr_b = (__global uchar *)(src1_ptr + (get_global_id(0) / TRANSPOSE1XW_WIDTH_STEP) * src1_stride_y + src1_offset_first_element_in_bytes);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr_b += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr_b += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    uint c00 = 0;
    uint c01 = 0;
    uint c02 = 0;
    uint c03 = 0;

    uint c10 = 0;
    uint c11 = 0;
    uint c12 = 0;
    uint c13 = 0;

    uint c20 = 0;
    uint c21 = 0;
    uint c22 = 0;
    uint c23 = 0;

    uint c30 = 0;
    uint c31 = 0;
    uint c32 = 0;
    uint c33 = 0;

#define COLS_MTX_B (COLS_B / (16 * MULT_TRANSPOSE1XW_WIDTH))

#if MULT_INTERLEAVE4X4_HEIGHT == 1
    int i = 0;
    for(; i <= (int)(COLS_MTX_B - 8); i += 8)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar16 a0 = vload16(0, src_addr_a);
        uchar4  b0 = vload4(0, src_addr_b);
        uchar4  b1 = vload4(0, src_addr_b + 4 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b2 = vload4(0, src_addr_b + 8 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b3 = vload4(0, src_addr_b + 12 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b4 = vload4(0, src_addr_b + 16 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b5 = vload4(0, src_addr_b + 20 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b6 = vload4(0, src_addr_b + 24 * TRANSPOSE1XW_WIDTH_STEP);
        uchar4  b7 = vload4(0, src_addr_b + 28 * TRANSPOSE1XW_WIDTH_STEP);

        // Accumulate
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), c00);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), c01);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), c02);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), c03);

        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), c10);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), c11);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), c12);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), c13);

        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), c20);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), c21);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), c22);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), c23);

        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), c30);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), c31);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), c32);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), c33);

        // Accumulate
        a0 = vload16(0, src_addr_a + 16);

        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b4.s0, b5.s0, b6.s0, b7.s0), c00);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b4.s1, b5.s1, b6.s1, b7.s1), c01);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b4.s2, b5.s2, b6.s2, b7.s2), c02);
        ARM_DOT((uchar4)(a0.s0123), (uchar4)(b4.s3, b5.s3, b6.s3, b7.s3), c03);

        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b4.s0, b5.s0, b6.s0, b7.s0), c10);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b4.s1, b5.s1, b6.s1, b7.s1), c11);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b4.s2, b5.s2, b6.s2, b7.s2), c12);
        ARM_DOT((uchar4)(a0.s4567), (uchar4)(b4.s3, b5.s3, b6.s3, b7.s3), c13);

        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b4.s0, b5.s0, b6.s0, b7.s0), c20);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b4.s1, b5.s1, b6.s1, b7.s1), c21);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b4.s2, b5.s2, b6.s2, b7.s2), c22);
        ARM_DOT((uchar4)(a0.s89AB), (uchar4)(b4.s3, b5.s3, b6.s3, b7.s3), c23);

        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b4.s0, b5.s0, b6.s0, b7.s0), c30);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b4.s1, b5.s1, b6.s1, b7.s1), c31);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b4.s2, b5.s2, b6.s2, b7.s2), c32);
        ARM_DOT((uchar4)(a0.sCDEF), (uchar4)(b4.s3, b5.s3, b6.s3, b7.s3), c33);

        src_addr_a += 32;
        src_addr_b += 32 * TRANSPOSE1XW_WIDTH_STEP;
    }
#endif // MULT_INTERLEAVE4X4_HEIGHT == 1
    int i_left_over = 0;
    for(; i < (int)(COLS_MTX_B); ++i)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar16 a0 = vload16(0, src_addr_a + (i_left_over % 4) + ((i_left_over / 4) * 16));
        uchar4  b0 = vload4(0, src_addr_b);

        c00 += a0.s0 * b0.s0;
        c01 += a0.s0 * b0.s1;
        c02 += a0.s0 * b0.s2;
        c03 += a0.s0 * b0.s3;

        c10 += a0.s4 * b0.s0;
        c11 += a0.s4 * b0.s1;
        c12 += a0.s4 * b0.s2;
        c13 += a0.s4 * b0.s3;

        c20 += a0.s8 * b0.s0;
        c21 += a0.s8 * b0.s1;
        c22 += a0.s8 * b0.s2;
        c23 += a0.s8 * b0.s3;

        c30 += a0.sC * b0.s0;
        c31 += a0.sC * b0.s1;
        c32 += a0.sC * b0.s2;
        c33 += a0.sC * b0.s3;

        i_left_over++;
        src_addr_b += 4 * TRANSPOSE1XW_WIDTH_STEP;
    }

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

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst.ptr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x4 block
    vstore4((int4)(c00, c01, c02, c03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
    vstore4((int4)(c10, c11, c12, c13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
    vstore4((int4)(c20, c21, c22, c23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
    vstore4((int4)(c30, c31, c32, c33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst.ptr += get_global_id(2) * dst_stride_z;

    // Store 4x4 block
    vstore4((int4)(c00, c01, c02, c03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
    vstore4((int4)(c10, c11, c12, c13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
    vstore4((int4)(c20, c21, c22, c23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
    vstore4((int4)(c30, c31, c32, c33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#endif // defined(COLS_B) && defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(TRANSPOSE1XW_WIDTH_STEP)

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

/** OpenCL kernel optimized for Bifrost architectures that computes the matrix multiplication between matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
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
__kernel void gemmlowp_mm_bifrost(IMAGE_DECLARATION(src0),
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

    uint acc00 = 0;
    uint acc01 = 0;
    uint acc02 = 0;
    uint acc03 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    uint acc10 = 0;
    uint acc11 = 0;
    uint acc12 = 0;
    uint acc13 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    uint acc20 = 0;
    uint acc21 = 0;
    uint acc22 = 0;
    uint acc23 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    uint acc30 = 0;
    uint acc31 = 0;
    uint acc32 = 0;
    uint acc33 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    uint acc40 = 0;
    uint acc41 = 0;
    uint acc42 = 0;
    uint acc43 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

    for(; src_addr.s0 <= (end_row_vec_a - 4); src_addr += (int2)(4, 4 * src1_stride_y))
    {
        // Load values from matrix A
        uchar4 a0 = vload4(0, src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar4 a1 = vload4(0, src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar4 a2 = vload4(0, src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar4 a3 = vload4(0, src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar4 a4 = vload4(0, src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        uchar4 b0 = vload4(0, src1_ptr + src_addr.s1 + 0 * src1_stride_y);
        uchar4 b1 = vload4(0, src1_ptr + src_addr.s1 + 1 * src1_stride_y);
        uchar4 b2 = vload4(0, src1_ptr + src_addr.s1 + 2 * src1_stride_y);
        uchar4 b3 = vload4(0, src1_ptr + src_addr.s1 + 3 * src1_stride_y);

        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a0.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a0.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a0.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a0.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a0.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a0.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a0.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a0.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a0.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a0.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a0.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a0.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a0.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a0.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a0.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a0.s3;

            acc00 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc01 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc02 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc03 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a1.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a1.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a1.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a1.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a1.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a1.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a1.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a1.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a1.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a1.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a1.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a1.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a1.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a1.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a1.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a1.s3;

            acc10 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc11 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc12 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc13 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a2.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a2.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a2.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a2.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a2.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a2.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a2.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a2.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a2.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a2.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a2.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a2.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a2.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a2.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a2.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a2.s3;

            acc20 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc21 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc22 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc23 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a3.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a3.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a3.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a3.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a3.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a3.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a3.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a3.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a3.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a3.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a3.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a3.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a3.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a3.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a3.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a3.s3;

            acc30 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc31 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc32 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc33 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a4.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a4.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a4.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a4.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a4.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a4.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a4.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a4.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a4.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a4.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a4.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a4.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a4.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a4.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a4.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a4.s3;

            acc40 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc41 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc42 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc43 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
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
        uchar4 b0 = vload4(0, src1_ptr + src_addr.s1);

        // Accumulate
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a0;

            acc00 += ((uint)tmp0);
            acc01 += ((uint)tmp1);
            acc02 += ((uint)tmp2);
            acc03 += ((uint)tmp3);
        }
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a1;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a1;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a1;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a1;

            acc10 += ((uint)tmp0);
            acc11 += ((uint)tmp1);
            acc12 += ((uint)tmp2);
            acc13 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a2;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a2;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a2;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a2;

            acc20 += ((uint)tmp0);
            acc21 += ((uint)tmp1);
            acc22 += ((uint)tmp2);
            acc23 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a3;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a3;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a3;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a3;

            acc30 += ((uint)tmp0);
            acc31 += ((uint)tmp1);
            acc32 += ((uint)tmp2);
            acc33 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a4;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a4;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a4;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a4;

            acc40 += ((uint)tmp0);
            acc41 += ((uint)tmp1);
            acc42 += ((uint)tmp2);
            acc43 += ((uint)tmp3);
        }
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
    vstore4((int4)(acc00, acc01, acc02, acc03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((int4)(acc10, acc11, acc12, acc13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((int4)(acc20, acc21, acc22, acc23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((int4)(acc30, acc31, acc32, acc33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    vstore4((int4)(acc40, acc41, acc42, acc43), 0, (__global int *)(dst.ptr + 4 * dst_stride_y + zout.s4));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst.ptr += z * dst_stride_z;

    // Store the result
    vstore4((int4)(acc00, acc01, acc02, acc03), 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((int4)(acc10, acc11, acc12, acc13), 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((int4)(acc20, acc21, acc22, acc23), 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((int4)(acc30, acc31, acc32, acc33), 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    vstore4((int4)(acc40, acc41, acc42, acc43), 0, (__global int *)(dst.ptr + 4 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
/** OpenCL kernel optimized to use dot product that computes the matrix multiplication between matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
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
__kernel void gemmlowp_mm_bifrost_dot8(IMAGE_DECLARATION(src0),
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
#endif // REINTERPRET_OUTPUT_AS_3D)
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

    zin += ((uint4)(0, 1, 2, 3)) * src0_stride_y;

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

    uint acc00 = 0;
    uint acc01 = 0;
    uint acc02 = 0;
    uint acc03 = 0;
    uint acc04 = 0;
    uint acc05 = 0;
    uint acc06 = 0;
    uint acc07 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    uint acc10 = 0;
    uint acc11 = 0;
    uint acc12 = 0;
    uint acc13 = 0;
    uint acc14 = 0;
    uint acc15 = 0;
    uint acc16 = 0;
    uint acc17 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    uint acc20 = 0;
    uint acc21 = 0;
    uint acc22 = 0;
    uint acc23 = 0;
    uint acc24 = 0;
    uint acc25 = 0;
    uint acc26 = 0;
    uint acc27 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    uint acc30 = 0;
    uint acc31 = 0;
    uint acc32 = 0;
    uint acc33 = 0;
    uint acc34 = 0;
    uint acc35 = 0;
    uint acc36 = 0;
    uint acc37 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    // A and B src indices get incremented at the same time.
    int i = 0;
    for(; i <= ((int)COLS_A - 8); i += 8)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        uchar8 a0 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar8 a1 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar8 a2 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar8 a3 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        uchar8 a0 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar8 a1 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar8 a2 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar8 a3 = vload8(0, (__global uchar *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        uchar8 b0 = vload8(0, src1_ptr + src_addr.s1 + 0 * src1_stride_y);
        uchar8 b1 = vload8(0, src1_ptr + src_addr.s1 + 1 * src1_stride_y);
        uchar8 b2 = vload8(0, src1_ptr + src_addr.s1 + 2 * src1_stride_y);
        uchar8 b3 = vload8(0, src1_ptr + src_addr.s1 + 3 * src1_stride_y);
        src_addr.s1 += 4 * src1_stride_y;

        ARM_DOT(a0.s0123, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc00);
        ARM_DOT(a0.s0123, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc01);
        ARM_DOT(a0.s0123, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc02);
        ARM_DOT(a0.s0123, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc03);
        ARM_DOT(a0.s0123, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc04);
        ARM_DOT(a0.s0123, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc05);
        ARM_DOT(a0.s0123, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc06);
        ARM_DOT(a0.s0123, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc07);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        ARM_DOT(a1.s0123, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc10);
        ARM_DOT(a1.s0123, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc11);
        ARM_DOT(a1.s0123, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc12);
        ARM_DOT(a1.s0123, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc13);
        ARM_DOT(a1.s0123, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc14);
        ARM_DOT(a1.s0123, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc15);
        ARM_DOT(a1.s0123, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc16);
        ARM_DOT(a1.s0123, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc17);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        ARM_DOT(a2.s0123, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc20);
        ARM_DOT(a2.s0123, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc21);
        ARM_DOT(a2.s0123, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc22);
        ARM_DOT(a2.s0123, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc23);
        ARM_DOT(a2.s0123, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc24);
        ARM_DOT(a2.s0123, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc25);
        ARM_DOT(a2.s0123, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc26);
        ARM_DOT(a2.s0123, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc27);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        ARM_DOT(a3.s0123, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc30);
        ARM_DOT(a3.s0123, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc31);
        ARM_DOT(a3.s0123, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc32);
        ARM_DOT(a3.s0123, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc33);
        ARM_DOT(a3.s0123, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc34);
        ARM_DOT(a3.s0123, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc35);
        ARM_DOT(a3.s0123, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc36);
        ARM_DOT(a3.s0123, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc37);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = vload8(0, src1_ptr + src_addr.s1 + 0 * src1_stride_y);
        b1 = vload8(0, src1_ptr + src_addr.s1 + 1 * src1_stride_y);
        b2 = vload8(0, src1_ptr + src_addr.s1 + 2 * src1_stride_y);
        b3 = vload8(0, src1_ptr + src_addr.s1 + 3 * src1_stride_y);
        src_addr.s1 += 4 * src1_stride_y;

        ARM_DOT(a0.s4567, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc00);
        ARM_DOT(a0.s4567, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc01);
        ARM_DOT(a0.s4567, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc02);
        ARM_DOT(a0.s4567, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc03);
        ARM_DOT(a0.s4567, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc04);
        ARM_DOT(a0.s4567, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc05);
        ARM_DOT(a0.s4567, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc06);
        ARM_DOT(a0.s4567, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc07);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        ARM_DOT(a1.s4567, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc10);
        ARM_DOT(a1.s4567, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc11);
        ARM_DOT(a1.s4567, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc12);
        ARM_DOT(a1.s4567, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc13);
        ARM_DOT(a1.s4567, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc14);
        ARM_DOT(a1.s4567, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc15);
        ARM_DOT(a1.s4567, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc16);
        ARM_DOT(a1.s4567, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc17);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        ARM_DOT(a2.s4567, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc20);
        ARM_DOT(a2.s4567, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc21);
        ARM_DOT(a2.s4567, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc22);
        ARM_DOT(a2.s4567, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc23);
        ARM_DOT(a2.s4567, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc24);
        ARM_DOT(a2.s4567, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc25);
        ARM_DOT(a2.s4567, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc26);
        ARM_DOT(a2.s4567, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc27);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        ARM_DOT(a3.s4567, (uchar4)(b0.s0, b1.s0, b2.s0, b3.s0), acc30);
        ARM_DOT(a3.s4567, (uchar4)(b0.s1, b1.s1, b2.s1, b3.s1), acc31);
        ARM_DOT(a3.s4567, (uchar4)(b0.s2, b1.s2, b2.s2, b3.s2), acc32);
        ARM_DOT(a3.s4567, (uchar4)(b0.s3, b1.s3, b2.s3, b3.s3), acc33);
        ARM_DOT(a3.s4567, (uchar4)(b0.s4, b1.s4, b2.s4, b3.s4), acc34);
        ARM_DOT(a3.s4567, (uchar4)(b0.s5, b1.s5, b2.s5, b3.s5), acc35);
        ARM_DOT(a3.s4567, (uchar4)(b0.s6, b1.s6, b2.s6, b3.s6), acc36);
        ARM_DOT(a3.s4567, (uchar4)(b0.s7, b1.s7, b2.s7, b3.s7), acc37);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += 8;
    }

    for(; i < (int)COLS_A; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        uchar a0 = *((__global uchar *)(src0_ptr + src_addr.s0 + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar a1 = *((__global uchar *)(src0_ptr + src_addr.s0 + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar a2 = *((__global uchar *)(src0_ptr + src_addr.s0 + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar a3 = *((__global uchar *)(src0_ptr + src_addr.s0 + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        uchar a0 = *((__global uchar *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar a1 = *((__global uchar *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar a2 = *((__global uchar *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar a3 = *((__global uchar *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        uchar8 b0 = vload8(0, src1_ptr + src_addr.s1);
        src_addr.s1 += src1_stride_y;

        acc00 += (uint)a0 * b0.s0;
        acc01 += (uint)a0 * b0.s1;
        acc02 += (uint)a0 * b0.s2;
        acc03 += (uint)a0 * b0.s3;
        acc04 += (uint)a0 * b0.s4;
        acc05 += (uint)a0 * b0.s5;
        acc06 += (uint)a0 * b0.s6;
        acc07 += (uint)a0 * b0.s7;

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc10 += (uint)a1 * b0.s0;
        acc11 += (uint)a1 * b0.s1;
        acc12 += (uint)a1 * b0.s2;
        acc13 += (uint)a1 * b0.s3;
        acc14 += (uint)a1 * b0.s4;
        acc15 += (uint)a1 * b0.s5;
        acc16 += (uint)a1 * b0.s6;
        acc17 += (uint)a1 * b0.s7;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc20 += (uint)a2 * b0.s0;
        acc21 += (uint)a2 * b0.s1;
        acc22 += (uint)a2 * b0.s2;
        acc23 += (uint)a2 * b0.s3;
        acc24 += (uint)a2 * b0.s4;
        acc25 += (uint)a2 * b0.s5;
        acc26 += (uint)a2 * b0.s6;
        acc27 += (uint)a2 * b0.s7;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc30 += (uint)a3 * b0.s0;
        acc31 += (uint)a3 * b0.s1;
        acc32 += (uint)a3 * b0.s2;
        acc33 += (uint)a3 * b0.s3;
        acc34 += (uint)a3 * b0.s4;
        acc35 += (uint)a3 * b0.s5;
        acc36 += (uint)a3 * b0.s6;
        acc37 += (uint)a3 * b0.s7;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += 1;
    }

    int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = dst.ptr;

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
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the result
    vstore4((int4)(acc00, acc01, acc02, acc03), 0, (__global int *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore4((int4)(acc04, acc05, acc06, acc07), 1, (__global int *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((int4)(acc10, acc11, acc12, acc13), 0, (__global int *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore4((int4)(acc14, acc15, acc16, acc17), 1, (__global int *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((int4)(acc20, acc21, acc22, acc23), 0, (__global int *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore4((int4)(acc24, acc25, acc26, acc27), 1, (__global int *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((int4)(acc30, acc31, acc32, acc33), 0, (__global int *)(dst_addr + 3 * dst_stride_y + zout.s3));
    vstore4((int4)(acc34, acc35, acc36, acc37), 0, (__global int *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store the result
    vstore4((int4)(acc00, acc01, acc02, acc03), 0, (__global int *)(dst_addr + 0 * dst_stride_y));
    vstore4((int4)(acc04, acc05, acc06, acc07), 1, (__global int *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((int4)(acc10, acc11, acc12, acc13), 0, (__global int *)(dst_addr + 1 * dst_stride_y));
    vstore4((int4)(acc14, acc15, acc16, acc17), 1, (__global int *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((int4)(acc20, acc21, acc22, acc23), 0, (__global int *)(dst_addr + 2 * dst_stride_y));
    vstore4((int4)(acc24, acc25, acc26, acc27), 1, (__global int *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((int4)(acc30, acc31, acc32, acc33), 0, (__global int *)(dst_addr + 3 * dst_stride_y));
    vstore4((int4)(acc34, acc35, acc36, acc37), 0, (__global int *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#endif // defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && defined(COLS_A)

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if K0 == 2
#define ARM_DOT_K0(a, b, c)                                         \
    ({                                                              \
        ARM_DOT((uchar4)(a, (uchar2)0), (uchar4)(b, (uchar2)0), c); \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c)                                       \
    ({                                                            \
        ARM_DOT((uchar4)(a, (uchar)0), (uchar4)(b, (uchar)0), c); \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        ARM_DOT(a, b, c);   \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c)           \
    ({                                \
        ARM_DOT(a.s0123, b.s0123, c); \
        ARM_DOT(a.s4567, b.s4567, c); \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c)           \
    ({                                \
        ARM_DOT(a.s0123, b.s0123, c); \
        ARM_DOT(a.s4567, b.s4567, c); \
        ARM_DOT(a.s89AB, b.s89AB, c); \
        ARM_DOT(a.sCDEF, b.sCDEF, c); \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0

#else // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if K0 == 2
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
        c += (uint)a.s2 * b.s2; \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
        c += (uint)a.s2 * b.s2; \
        c += (uint)a.s3 * b.s3; \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
        c += (uint)a.s2 * b.s2; \
        c += (uint)a.s3 * b.s3; \
        c += (uint)a.s4 * b.s4; \
        c += (uint)a.s5 * b.s5; \
        c += (uint)a.s6 * b.s6; \
        c += (uint)a.s7 * b.s7; \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c += (uint)a.s0 * b.s0; \
        c += (uint)a.s1 * b.s1; \
        c += (uint)a.s2 * b.s2; \
        c += (uint)a.s3 * b.s3; \
        c += (uint)a.s4 * b.s4; \
        c += (uint)a.s5 * b.s5; \
        c += (uint)a.s6 * b.s6; \
        c += (uint)a.s7 * b.s7; \
        c += (uint)a.s8 * b.s8; \
        c += (uint)a.s9 * b.s9; \
        c += (uint)a.sA * b.sA; \
        c += (uint)a.sB * b.sB; \
        c += (uint)a.sC * b.sC; \
        c += (uint)a.sD * b.sD; \
        c += (uint)a.sE * b.sE; \
        c += (uint)a.sF * b.sF; \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0

#endif //defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

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

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (i.e. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
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

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (get_global_id(1) % V0) * (uint)LHS_OFFSET_X + (get_global_id(1) / V0) * (uint)lhs_stride_y + (get_global_id(
                                   2)
                               * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (get_global_id(0) % H0) * (uint)RHS_OFFSET_X + (get_global_id(0) / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (get_global_id(2) % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += get_global_id(2) * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(uint, N0), c, 0); //VEC_DATA_TYPE(uint, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    for(int i = 0; i < k; i += K0)
    {
        // Supported cases (M0, K0):
        // 2,4 - 2,8 - 2,16
        // 3,4 - 3,8 - 3,16
        // 4,4 - 4,8 - 4,16
        // 5,4 - 5,8 - 5,16
        // 6,4 - 6,8 - 6,16
        // Load values from LHS matrix
        VEC_DATA_TYPE(uchar, K0)
        a0 = VLOAD(K0)(0, lhs_addr + 0 * LHS_STEP_X);
#if M0 > 1
        VEC_DATA_TYPE(uchar, K0)
        a1 = VLOAD(K0)(0, lhs_addr + 1 * LHS_STEP_X);
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(uchar, K0)
        a2 = VLOAD(K0)(0, lhs_addr + 2 * LHS_STEP_X);
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(uchar, K0)
        a3 = VLOAD(K0)(0, lhs_addr + 3 * LHS_STEP_X);
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(uchar, K0)
        a4 = VLOAD(K0)(0, lhs_addr + 4 * LHS_STEP_X);
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(uchar, K0)
        a5 = VLOAD(K0)(0, lhs_addr + 5 * LHS_STEP_X);
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(uchar, K0)
        a6 = VLOAD(K0)(0, lhs_addr + 6 * LHS_STEP_X);
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(uchar, K0)
        a7 = VLOAD(K0)(0, lhs_addr + 7 * LHS_STEP_X);
#endif // M0 > 7

        // Load values from RHS matrix
        VEC_DATA_TYPE(uchar, K0)
        b0 = VLOAD(K0)(0, rhs_addr + 0 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        b1 = VLOAD(K0)(0, rhs_addr + 1 * RHS_STEP_X);
#if N0 > 2
        VEC_DATA_TYPE(uchar, K0)
        b2 = VLOAD(K0)(0, rhs_addr + 2 * RHS_STEP_X);
#endif // N0 > 2
#if N0 > 3
        VEC_DATA_TYPE(uchar, K0)
        b3 = VLOAD(K0)(0, rhs_addr + 3 * RHS_STEP_X);
#endif // N0 > 3
#if N0 > 4
        VEC_DATA_TYPE(uchar, K0)
        b4 = VLOAD(K0)(0, rhs_addr + 4 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        b5 = VLOAD(K0)(0, rhs_addr + 5 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        b6 = VLOAD(K0)(0, rhs_addr + 6 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        b7 = VLOAD(K0)(0, rhs_addr + 7 * RHS_STEP_X);
#endif // N0 > 4
#if N0 > 8
        VEC_DATA_TYPE(uchar, K0)
        b8 = VLOAD(K0)(0, rhs_addr + 8 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        b9 = VLOAD(K0)(0, rhs_addr + 9 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bA = VLOAD(K0)(0, rhs_addr + 10 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bB = VLOAD(K0)(0, rhs_addr + 11 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bC = VLOAD(K0)(0, rhs_addr + 12 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bD = VLOAD(K0)(0, rhs_addr + 13 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bE = VLOAD(K0)(0, rhs_addr + 14 * RHS_STEP_X);
        VEC_DATA_TYPE(uchar, K0)
        bF = VLOAD(K0)(0, rhs_addr + 15 * RHS_STEP_X);
#endif // N0 > 8

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

        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP);
        rhs_addr += (N0 * RHS_STEP_X * RHS_STEP_LOOP);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(int)) + (get_global_id(1) * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

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

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    zout0 = (0 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout0 = min((uint)(DEPTH_GEMM3D - 1), zout0);
    zout0 *= (dst_cross_plane_pad * dst_stride_y);
#if M0 > 1
    zout1 = (1 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout1 = min((uint)(DEPTH_GEMM3D - 1), zout1);
    zout1 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 1
#if M0 > 2
    zout2 = (2 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout2 = min((uint)(DEPTH_GEMM3D - 1), zout2);
    zout2 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 2
#if M0 > 3
    zout3 = (3 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout3 = min((uint)(DEPTH_GEMM3D - 1), zout3);
    zout3 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 3
#if M0 > 4
    zout4 = (4 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout4 = min((uint)(DEPTH_GEMM3D - 1), zout4);
    zout4 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 4
#if M0 > 5
    zout5 = (5 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout5 = min((uint)(DEPTH_GEMM3D - 1), zout5);
    zout5 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 5
#if M0 > 6
    zout6 = (6 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout6 = min((uint)(DEPTH_GEMM3D - 1), zout6);
    zout6 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 6
#if M0 > 7
    zout7 = (7 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout7 = min((uint)(DEPTH_GEMM3D - 1), zout7);
    zout7 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 7

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Store output block
    VSTORE(N0)
    (CONVERT_SAT(c0, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 0 * dst_stride_y + zout0));
#if M0 > 1
    VSTORE(N0)
    (CONVERT_SAT(c1, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 1 * dst_stride_y + zout1));
#endif // M0 > 1
#if M0 > 2
    VSTORE(N0)
    (CONVERT_SAT(c2, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 2 * dst_stride_y + zout2));
#endif // M0 > 2
#if M0 > 3
    VSTORE(N0)
    (CONVERT_SAT(c3, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 3 * dst_stride_y + zout3));
#endif // M0 > 3
#if M0 > 4
    VSTORE(N0)
    (CONVERT_SAT(c4, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 4 * dst_stride_y + zout4));
#endif // M0 > 4
#if M0 > 5
    VSTORE(N0)
    (CONVERT_SAT(c5, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 5 * dst_stride_y + zout5));
#endif // M0 > 5
#if M0 > 6
    VSTORE(N0)
    (CONVERT_SAT(c6, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 6 * dst_stride_y + zout6));
#endif // M0 > 6
#if M0 > 7
    VSTORE(N0)
    (CONVERT_SAT(c7, VEC_DATA_TYPE(int, N0)), 0, (__global int *)(dst_addr + 7 * dst_stride_y + zout7));
#endif // M0 > 7

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices unsing the dot8 instruction.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (i.e. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
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
__kernel void gemmlowp_mm_reshaped_lhs_nt_rhs_t_dot8(IMAGE_DECLARATION(lhs),
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
    // Note: ARM_DOT_K0XN0 is generated with the dot8 instruction
    gemmlowp_mm_reshaped_lhs_nt_rhs_t(lhs_ptr,
                                      lhs_stride_x,
                                      lhs_step_x,
                                      lhs_stride_y,
                                      lhs_step_y,
                                      lhs_offset_first_element_in_bytes,
                                      rhs_ptr,
                                      rhs_stride_x,
                                      rhs_step_x,
                                      rhs_stride_y,
                                      rhs_step_y,
                                      rhs_offset_first_element_in_bytes,
                                      dst_ptr,
                                      dst_stride_x,
                                      dst_step_x,
                                      dst_stride_y,
                                      dst_step_y,
                                      dst_offset_first_element_in_bytes,
                                      k,
                                      lhs_stride_z,
                                      rhs_stride_z,
                                      dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                      ,
                                      dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                     );
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(K)

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
