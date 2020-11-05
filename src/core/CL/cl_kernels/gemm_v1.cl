/*
 * Copyright (c) 2020 Arm Limited.
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
#include "repeat.h"

#if defined(M) && defined(N) && defined(K) && defined(H0) && defined(V0) && defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)
/** This OpenCL kernel is optimised for Midgard. It computes the matrix multiplication between matrix A reshaped (src0) and matrix B reshaped (src1)
 *
 * @note The number of rows of destination matrix must be passed at compile time using -DM
 * @note The number of columns of the destination matrix must be passed at compile time using -DN
 * @note The number of rows of the *un-reshaped* matrix B (K) must be passed at compile time using -DK
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note The multiplication factor for the transposition width (H0) must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f32(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                 IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
#if defined(BETA)
                                                 uint src2_stride_z,
#endif //defined(BETA)
                                                 uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int x = get_global_id(0) / H0;
    int y = get_global_id(1) / V0;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % V0) * 4;
    const int offset_row_b = (get_global_id(0) % H0) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global float *src_addr_a = (__global float *)(src0_ptr + src0_addr_in_bytes);
    __global float *src_addr_b = (__global float *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global float *src_end_addr_b = src_addr_b + (src1_stride_y / sizeof(float));

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float4 c0 = 0.0f;
    float4 c1 = 0.0f;
    float4 c2 = 0.0f;
    float4 c3 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(8 * H0)); src_addr_a += 8 * V0, src_addr_b += 8 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c0 += (float4)a0.s0 * b0;
        c1 += (float4)a0.s1 * b0;
        c2 += (float4)a0.s2 * b0;
        c3 += (float4)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 4 * V0);
        b0 = vload4(0, src_addr_b + 4 * H0);

        c0 += (float4)a0.s0 * b0;
        c1 += (float4)a0.s1 * b0;
        c2 += (float4)a0.s2 * b0;
        c3 += (float4)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * V0, src_addr_b += 4 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c0 += (float4)a0.s0 * b0;
        c1 += (float4)a0.s1 * b0;
        c2 += (float4)a0.s2 * b0;
        c3 += (float4)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    uint4 zout = 0;

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
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(4, float, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(4, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float));

    LOAD_BLOCK(1, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(4, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float)) + (get_global_id(1) * (uint)4 * src2_stride_y) + get_global_id(
                                    2) * src2_stride_z;

    LOAD_BLOCK(4, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(4, float, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(4, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(4, ACTIVATION_TYPE, float, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store 4x4 block
    const bool cond_y = ((get_global_id(1) + 1) * 4 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * 4 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(4, 4, float, c, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

/** This OpenCL kernel is optimized for Bifrost and tt computes the matrix multiplication between matrix A reshaped (src0) and matrix B reshaped (src1)
 *
 * @note The number of rows of destination matrix must be passed at compile time using -DM
 * @note The number of columns of the destination matrix must be passed at compile time using -DN
 * @note The number of rows of the *un-reshaped* matrix B (K) must be passed at compile time using -DK
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note The multiplication factor for the transposition width (H0) must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f32_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                         IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
#if defined(BETA)
                                                         uint src2_stride_z,
#endif //defined(BETA)
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    int x = get_global_id(0) / H0;
    int y = get_global_id(1) / V0;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % V0) * 4;
    const int offset_row_b = (get_global_id(0) % H0) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global float *src_addr_a = (__global float *)(src0_ptr + src0_addr_in_bytes);
    __global float *src_addr_b = (__global float *)(src1_ptr + src1_addr_in_bytes);

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float4 c0 = 0.0f;
    float4 c1 = 0.0f;
    float4 c2 = 0.0f;
    float4 c3 = 0.0f;

    int i = 0;
    for(; i <= (int)(K - 4); i += 4)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 4 * H0;

        c0.s0 = fma(a0.s0, b0.s0, c0.s0);
        c0.s1 = fma(a0.s0, b0.s1, c0.s1);
        c0.s2 = fma(a0.s0, b0.s2, c0.s2);
        c0.s3 = fma(a0.s0, b0.s3, c0.s3);

        c1.s0 = fma(a0.s1, b0.s0, c1.s0);
        c1.s1 = fma(a0.s1, b0.s1, c1.s1);
        c1.s2 = fma(a0.s1, b0.s2, c1.s2);
        c1.s3 = fma(a0.s1, b0.s3, c1.s3);

        c2.s0 = fma(a0.s2, b0.s0, c2.s0);
        c2.s1 = fma(a0.s2, b0.s1, c2.s1);
        c2.s2 = fma(a0.s2, b0.s2, c2.s2);
        c2.s3 = fma(a0.s2, b0.s3, c2.s3);

        c3.s0 = fma(a0.s3, b0.s0, c3.s0);
        c3.s1 = fma(a0.s3, b0.s1, c3.s1);
        c3.s2 = fma(a0.s3, b0.s2, c3.s2);
        c3.s3 = fma(a0.s3, b0.s3, c3.s3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 4 * H0;

        c0.s0 = fma(a0.s0, b0.s0, c0.s0);
        c0.s1 = fma(a0.s0, b0.s1, c0.s1);
        c0.s2 = fma(a0.s0, b0.s2, c0.s2);
        c0.s3 = fma(a0.s0, b0.s3, c0.s3);

        c1.s0 = fma(a0.s1, b0.s0, c1.s0);
        c1.s1 = fma(a0.s1, b0.s1, c1.s1);
        c1.s2 = fma(a0.s1, b0.s2, c1.s2);
        c1.s3 = fma(a0.s1, b0.s3, c1.s3);

        c2.s0 = fma(a0.s2, b0.s0, c2.s0);
        c2.s1 = fma(a0.s2, b0.s1, c2.s1);
        c2.s2 = fma(a0.s2, b0.s2, c2.s2);
        c2.s3 = fma(a0.s2, b0.s3, c2.s3);

        c3.s0 = fma(a0.s3, b0.s0, c3.s0);
        c3.s1 = fma(a0.s3, b0.s1, c3.s1);
        c3.s2 = fma(a0.s3, b0.s2, c3.s2);
        c3.s3 = fma(a0.s3, b0.s3, c3.s3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 4 * H0;

        c0.s0 = fma(a0.s0, b0.s0, c0.s0);
        c0.s1 = fma(a0.s0, b0.s1, c0.s1);
        c0.s2 = fma(a0.s0, b0.s2, c0.s2);
        c0.s3 = fma(a0.s0, b0.s3, c0.s3);

        c1.s0 = fma(a0.s1, b0.s0, c1.s0);
        c1.s1 = fma(a0.s1, b0.s1, c1.s1);
        c1.s2 = fma(a0.s1, b0.s2, c1.s2);
        c1.s3 = fma(a0.s1, b0.s3, c1.s3);

        c2.s0 = fma(a0.s2, b0.s0, c2.s0);
        c2.s1 = fma(a0.s2, b0.s1, c2.s1);
        c2.s2 = fma(a0.s2, b0.s2, c2.s2);
        c2.s3 = fma(a0.s2, b0.s3, c2.s3);

        c3.s0 = fma(a0.s3, b0.s0, c3.s0);
        c3.s1 = fma(a0.s3, b0.s1, c3.s1);
        c3.s2 = fma(a0.s3, b0.s2, c3.s2);
        c3.s3 = fma(a0.s3, b0.s3, c3.s3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 4 * H0;

        c0.s0 = fma(a0.s0, b0.s0, c0.s0);
        c0.s1 = fma(a0.s0, b0.s1, c0.s1);
        c0.s2 = fma(a0.s0, b0.s2, c0.s2);
        c0.s3 = fma(a0.s0, b0.s3, c0.s3);

        c1.s0 = fma(a0.s1, b0.s0, c1.s0);
        c1.s1 = fma(a0.s1, b0.s1, c1.s1);
        c1.s2 = fma(a0.s1, b0.s2, c1.s2);
        c1.s3 = fma(a0.s1, b0.s3, c1.s3);

        c2.s0 = fma(a0.s2, b0.s0, c2.s0);
        c2.s1 = fma(a0.s2, b0.s1, c2.s1);
        c2.s2 = fma(a0.s2, b0.s2, c2.s2);
        c2.s3 = fma(a0.s2, b0.s3, c2.s3);

        c3.s0 = fma(a0.s3, b0.s0, c3.s0);
        c3.s1 = fma(a0.s3, b0.s1, c3.s1);
        c3.s2 = fma(a0.s3, b0.s2, c3.s2);
        c3.s3 = fma(a0.s3, b0.s3, c3.s3);
    }

    for(; i < (int)K; ++i)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 4 * H0;

        c0.s0 = fma(a0.s0, b0.s0, c0.s0);
        c0.s1 = fma(a0.s0, b0.s1, c0.s1);
        c0.s2 = fma(a0.s0, b0.s2, c0.s2);
        c0.s3 = fma(a0.s0, b0.s3, c0.s3);

        c1.s0 = fma(a0.s1, b0.s0, c1.s0);
        c1.s1 = fma(a0.s1, b0.s1, c1.s1);
        c1.s2 = fma(a0.s1, b0.s2, c1.s2);
        c1.s3 = fma(a0.s1, b0.s3, c1.s3);

        c2.s0 = fma(a0.s2, b0.s0, c2.s0);
        c2.s1 = fma(a0.s2, b0.s1, c2.s1);
        c2.s2 = fma(a0.s2, b0.s2, c2.s2);
        c2.s3 = fma(a0.s2, b0.s3, c2.s3);

        c3.s0 = fma(a0.s3, b0.s0, c3.s0);
        c3.s1 = fma(a0.s3, b0.s1, c3.s1);
        c3.s2 = fma(a0.s3, b0.s2, c3.s2);
        c3.s3 = fma(a0.s3, b0.s3, c3.s3);
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    uint4 zout = 0;

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
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(4, float, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(4, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float));

    LOAD_BLOCK(1, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(4, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float)) + (get_global_id(1) * (uint)4 * src2_stride_y) + get_global_id(
                                    2) * src2_stride_z;

    LOAD_BLOCK(4, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(4, float, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(4, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(4, ACTIVATION_TYPE, float, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store 4x4 block
    const bool cond_y = ((get_global_id(1) + 1) * 4 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * 4 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(4, 4, float, c, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
/** This OpenCL kernel computes the matrix multiplication between matrix A reshaped (src0) and matrix B reshaped (src1)
 *
 * @note The number of rows of destination matrix must be passed at compile time using -DM
 * @note The number of columns of the destination matrix must be passed at compile time using -DN
 * @note The number of rows of the *un-reshaped* matrix B (K) must be passed at compile time using -DK
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note The multiplication factor for the transposition width (H0) must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                 IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
#if defined(BETA)
                                                 uint src2_stride_z,
#endif //defined(BETA)
                                                 uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int x = get_global_id(0) / H0;
    int y = get_global_id(1) / V0;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % V0) * 4;
    const int offset_row_b = (get_global_id(0) % H0) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global half *src_end_addr_b = src_addr_b + (src1_stride_y / sizeof(half));

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    half8 c0 = 0.0f;
    half8 c1 = 0.0f;
    half8 c2 = 0.0f;
    half8 c3 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(16 * H0)); src_addr_a += 8 * V0, src_addr_b += 16 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        c0 += (half8)a0.s0 * b0;
        c1 += (half8)a0.s1 * b0;
        c2 += (half8)a0.s2 * b0;
        c3 += (half8)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 4 * V0);
        b0 = vload8(0, src_addr_b + 8 * H0);

        c0 += (half8)a0.s0 * b0;
        c1 += (half8)a0.s1 * b0;
        c2 += (half8)a0.s2 * b0;
        c3 += (half8)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * V0, src_addr_b += 8 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        c0 += (half8)a0.s0 * b0;
        c1 += (half8)a0.s1 * b0;
        c2 += (half8)a0.s2 * b0;
        c3 += (half8)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    uint4 zout = 0;

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
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(4, half, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(4, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half));

    LOAD_BLOCK(1, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, half, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(4, c, bias0);

#else // defined(BROADCAST_BIAS)

    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (get_global_id(1) * (uint)4 * src2_stride_y) + get_global_id(
                                    2) * src2_stride_z;

    LOAD_BLOCK(4, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(4, half, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(4, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(4, ACTIVATION_TYPE, half, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store 4x8 block
    const bool cond_y = ((get_global_id(1) + 1) * 4 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * 8 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(4, 8, half, c, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

/** This OpenCL kernel computes the matrix multiplication between matrix A reshaped (src0) and matrix B reshaped (src1) while accumulating the result in a 32 floating point variable.
 *
 * @note The number of rows of destination matrix must be passed at compile time using -DM
 * @note The number of columns of the destination matrix must be passed at compile time using -DN
 * @note The number of rows of the *un-reshaped* matrix B (K) must be passed at compile time using -DK
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note The multiplication factor for the transposition width (H0) must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16_acc32(IMAGE_DECLARATION(src0),
                                                       IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                       IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                       IMAGE_DECLARATION(dst),
                                                       uint src0_stride_z,
                                                       uint src1_stride_z,
#if defined(BETA)
                                                       uint src2_stride_z,
#endif //defined(BETA)
                                                       uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                       ,
                                                       uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                      )
{
    int x = get_global_id(0) / H0;
    int y = get_global_id(1) / V0;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % V0) * 4;
    const int offset_row_b = (get_global_id(0) % H0) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global half *src_end_addr_b = src_addr_b + (src1_stride_y / sizeof(half));

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float8 c0 = 0.0f;
    float8 c1 = 0.0f;
    float8 c2 = 0.0f;
    float8 c3 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(16 * H0)); src_addr_a += 8 * V0, src_addr_b += 16 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = convert_float4(vload4(0, src_addr_a));
        float8 b0 = convert_float8(vload8(0, src_addr_b));

        c0 += (float8)a0.s0 * b0;
        c1 += (float8)a0.s1 * b0;
        c2 += (float8)a0.s2 * b0;
        c3 += (float8)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = convert_float4(vload4(0, src_addr_a + 4 * V0));
        b0 = convert_float8(vload8(0, src_addr_b + 8 * H0));

        c0 += (float8)a0.s0 * b0;
        c1 += (float8)a0.s1 * b0;
        c2 += (float8)a0.s2 * b0;
        c3 += (float8)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * V0, src_addr_b += 8 * H0)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = convert_float4(vload4(0, src_addr_a));
        float8 b0 = convert_float8(vload8(0, src_addr_b));

        c0 += (float8)a0.s0 * b0;
        c1 += (float8)a0.s1 * b0;
        c2 += (float8)a0.s2 * b0;
        c3 += (float8)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    uint4 zout = 0;

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
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(4, float, c, ALPHA);
#endif // defined(ALPHA)

#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(4, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half));

    LOAD_BLOCK(1, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

    float8 bias_f0 = convert_float8(bias0);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias_f, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(4, c, bias_f0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (get_global_id(1) * (uint)4 * src2_stride_y) + get_global_id(
                                    2) * src2_stride_z;

    LOAD_BLOCK(4, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

    float8 bias_f0 = convert_float8(bias0);
    float8 bias_f1 = convert_float8(bias1);
    float8 bias_f2 = convert_float8(bias2);
    float8 bias_f3 = convert_float8(bias3);

#ifndef UNIT_BETA
    SCALE_BLOCK(4, float, bias_f, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(4, c, bias_f);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    half8 c_h0 = convert_half8(c0);
    half8 c_h1 = convert_half8(c1);
    half8 c_h2 = convert_half8(c2);
    half8 c_h3 = convert_half8(c3);

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(4, ACTIVATION_TYPE, half, VEC_SIZE, c_h, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store 4x8 block
    const bool cond_y = ((get_global_id(1) + 1) * 4 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * 8 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(4, 8, half, c_h, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

/** This OpenCL kernel optimized for Bifrost architectures computes the matrix multiplication between matrix A reshaped (src0) and matrix B reshaped (src1)
 *
 * @note The number of rows of destination matrix must be passed at compile time using -DM
 * @note The number of columns of the destination matrix must be passed at compile time using -DN
 * @note The number of rows of the *un-reshaped* matrix B (K) must be passed at compile time using -DK
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note The multiplication factor for the transposition width (H0) must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                         IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
#if defined(BETA)
                                                         uint src2_stride_z,
#endif //defined(BETA)
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    int x = get_global_id(0) / H0;
    int y = get_global_id(1) / V0;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % V0) * 4;
    const int offset_row_b = (get_global_id(0) % H0) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    half8 c0 = 0.0f;
    half8 c1 = 0.0f;
    half8 c2 = 0.0f;
    half8 c3 = 0.0f;

    int i = 0;
    for(; i <= (int)(K - 4); i += 4)
    {
#if V0 == 1
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half8 a0 = vload8(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 8 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);

        // Load values from matrix B (transposed)
        b0 = vload8(0, src_addr_b);

        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s4, b0, c0);
        c1 = fma((half8)a0.s5, b0, c1);
        c2 = fma((half8)a0.s6, b0, c2);
        c3 = fma((half8)a0.s7, b0, c3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload8(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 8 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);

        // Load values from matrix B (transposed)
        b0 = vload8(0, src_addr_b);

        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s4, b0, c0);
        c1 = fma((half8)a0.s5, b0, c1);
        c2 = fma((half8)a0.s6, b0, c2);
        c3 = fma((half8)a0.s7, b0, c3);
#else  // V0 == 1
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);
#endif // V0 == 1
    }

    for(; i < (int)K; ++i)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * V0;
        src_addr_b += 8 * H0;

        c0 = fma((half8)a0.s0, b0, c0);
        c1 = fma((half8)a0.s1, b0, c1);
        c2 = fma((half8)a0.s2, b0, c2);
        c3 = fma((half8)a0.s3, b0, c3);
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    uint4 zout = 0;

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
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(4, half, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(4, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half));

    LOAD_BLOCK(1, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, half, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(4, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (get_global_id(1) * (uint)4 * src2_stride_y) + get_global_id(
                                    2) * src2_stride_z;

    LOAD_BLOCK(4, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(4, half, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(4, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(4, ACTIVATION_TYPE, half, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store 4x8 block
    const bool cond_y = ((get_global_id(1) + 1) * 4 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * 8 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(4, 8, half, c, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)

#endif // defined(M) && defined(N) && defined(K) && defined(H0) && defined(V0) && defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)

#if defined(N) && defined(K) && defined(M0) && defined(N0) && defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)
#if defined(DATA_TYPE)
#define VECTOR_TYPE VEC_DATA_TYPE(DATA_TYPE, N0)
/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped.
 *
 * @note This OpenCL kernel works with floating point data types (F16/F32)
 * @note The floating point data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DN0 and -DM0
 * @note The number of columns of matrix A and the number of columns of the matrix B need to be passed at compile time using -DK and -DN
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16/F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the output tensor (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point(IMAGE_DECLARATION(src0),
                                     IMAGE_DECLARATION(src1),
#if defined(BETA)
                                     IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                     IMAGE_DECLARATION(dst),
                                     uint src0_stride_z,
                                     uint src1_stride_z,
#if defined(BETA)
                                     uint src2_stride_z,
#endif //defined(BETA)
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
    int idx = get_global_id(0) * N0;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * src0_stride_y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(DATA_TYPE);

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

    // The plane (zin) is calculated dividing row by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
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

    int end_row_vec_a = src_addr.s0 + (K * sizeof(DATA_TYPE));

    VECTOR_TYPE acc0 = 0.0f;
#if M0 > 1
    VECTOR_TYPE acc1 = 0.0f;
#endif // M0 > 1
#if M0 > 2
    VECTOR_TYPE acc2 = 0.0f;
#endif // M0 > 2
#if M0 > 3
    VECTOR_TYPE acc3 = 0.0f;
#endif // M0 > 3

    for(; src_addr.s0 <= (end_row_vec_a - 2 * (int)sizeof(DATA_TYPE)); src_addr += (int2)(2 * sizeof(DATA_TYPE), 2 * src1_stride_y))
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        LOAD_BLOCK(M0, 2, DATA_TYPE, a, src0_ptr, src_addr.s0, src0_stride_y, zin.s);
#else // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        VECTOR_TYPE b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1));
        VECTOR_TYPE b1 = VLOAD(N0)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1 + src1_stride_y));

        // Accumulate
        acc0 += b0 * (VECTOR_TYPE)a0.s0;
        acc0 += b1 * (VECTOR_TYPE)a0.s1;
#if M0 > 1
        acc1 += b0 * (VECTOR_TYPE)a1.s0;
        acc1 += b1 * (VECTOR_TYPE)a1.s1;
#endif // M0 > 1
#if M0 > 2
        acc2 += b0 * (VECTOR_TYPE)a2.s0;
        acc2 += b1 * (VECTOR_TYPE)a2.s1;
#endif // M0 > 2
#if M0 > 3
        acc3 += b0 * (VECTOR_TYPE)a3.s0;
        acc3 += b1 * (VECTOR_TYPE)a3.s1;
#endif // M0 > 3
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(DATA_TYPE), src1_stride_y))
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        DATA_TYPE a0 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if M0 > 1
        DATA_TYPE a1 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // M0 > 1
#if M0 > 2
        DATA_TYPE a2 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // M0 > 2
#if M0 > 3
        DATA_TYPE a3 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // M0 > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        DATA_TYPE a0 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        DATA_TYPE a1 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        DATA_TYPE a2 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        DATA_TYPE a3 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        VECTOR_TYPE b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1));

        // Accumulate
        acc0 += b0 * (VECTOR_TYPE)a0;
#if M0 > 1
        acc1 += b0 * (VECTOR_TYPE)a1;
#endif // M0 > 1
#if M0 > 2
        acc2 += b0 * (VECTOR_TYPE)a2;
#endif // M0 > 2
#if M0 > 3
        acc3 += b0 * (VECTOR_TYPE)a3;
#endif // M0 > 3
    }

    int z = get_global_id(2);

    // Compute dst address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                               PARTIAL_STORE_M0)
                               * dst_stride_y);

    uint4 zout = 0;

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

    // The plane (zout) is calculated dividing row by HEIGHT_GEMM3D
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, acc, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, acc, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                                PARTIAL_STORE_M0)
                                * src2_stride_y)
                                + z * src2_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, acc, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, acc, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store output block
    const bool cond_y = get_global_id(1) == 0;
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, acc, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}
#endif // defined(DATA_TYPE)

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped
 *
 * @note This OpenCL kernel works with the 32-bit floating point data type (float) and uses the fma units.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DN0 and -DM0.
 * @note This kernel processed a fixed number of elements along x: -DN0=4.
 * @note The number of columns of matrix A and the number of columns of the matrix B need to be passed at compile time using -DK and -DN
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f32_bifrost(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                 IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
#if defined(BETA)
                                                 uint src2_stride_z,
#endif //defined(BETA)
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
    int idx = get_global_id(0) * N0;

    // Compute starting address for matrix A and matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for matrix A
    src_addr.s0 += COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * src0_stride_y;

    // Update address for matrix B
    src_addr.s1 += idx * sizeof(float);

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

    // The plane (zin) is calculated dividing row by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
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

    // Initialize accumulators
    float4 acc0 = 0.0f;

#if M0 > 1
    float4 acc1 = 0.0f;
#endif // M0 > 1

#if M0 > 2
    float4 acc2 = 0.0f;
#endif // M0 > 2

#if M0 > 3
    float4 acc3 = 0.0f;
#endif // M0 > 3

    // A and B src indices get incremented at the same time.
    int i = 0;
    for(; i <= ((int)K - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        LOAD_BLOCK(M0, 4, float, a, src0_ptr, src_addr.s0, src0_stride_y, zin.s);
#else // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        float4 a0 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        float4 a1 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        float4 a2 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        float4 a3 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0.s0, b0.s0, acc0.s0);
        acc0.s1 = fma(a0.s0, b0.s1, acc0.s1);
        acc0.s2 = fma(a0.s0, b0.s2, acc0.s2);
        acc0.s3 = fma(a0.s0, b0.s3, acc0.s3);

#if M0 > 1

        acc1.s0 = fma(a1.s0, b0.s0, acc1.s0);
        acc1.s1 = fma(a1.s0, b0.s1, acc1.s1);
        acc1.s2 = fma(a1.s0, b0.s2, acc1.s2);
        acc1.s3 = fma(a1.s0, b0.s3, acc1.s3);

#endif // M0 > 1
#if M0 > 2

        acc2.s0 = fma(a2.s0, b0.s0, acc2.s0);
        acc2.s1 = fma(a2.s0, b0.s1, acc2.s1);
        acc2.s2 = fma(a2.s0, b0.s2, acc2.s2);
        acc2.s3 = fma(a2.s0, b0.s3, acc2.s3);

#endif // M0 > 2
#if M0 > 3

        acc3.s0 = fma(a3.s0, b0.s0, acc3.s0);
        acc3.s1 = fma(a3.s0, b0.s1, acc3.s1);
        acc3.s2 = fma(a3.s0, b0.s2, acc3.s2);
        acc3.s3 = fma(a3.s0, b0.s3, acc3.s3);
#endif // M0 > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0.s1, b0.s0, acc0.s0);
        acc0.s1 = fma(a0.s1, b0.s1, acc0.s1);
        acc0.s2 = fma(a0.s1, b0.s2, acc0.s2);
        acc0.s3 = fma(a0.s1, b0.s3, acc0.s3);

#if M0 > 1

        acc1.s0 = fma(a1.s1, b0.s0, acc1.s0);
        acc1.s1 = fma(a1.s1, b0.s1, acc1.s1);
        acc1.s2 = fma(a1.s1, b0.s2, acc1.s2);
        acc1.s3 = fma(a1.s1, b0.s3, acc1.s3);

#endif // M0 > 1
#if M0 > 2

        acc2.s0 = fma(a2.s1, b0.s0, acc2.s0);
        acc2.s1 = fma(a2.s1, b0.s1, acc2.s1);
        acc2.s2 = fma(a2.s1, b0.s2, acc2.s2);
        acc2.s3 = fma(a2.s1, b0.s3, acc2.s3);

#endif // M0 > 2
#if M0 > 3

        acc3.s0 = fma(a3.s1, b0.s0, acc3.s0);
        acc3.s1 = fma(a3.s1, b0.s1, acc3.s1);
        acc3.s2 = fma(a3.s1, b0.s2, acc3.s2);
        acc3.s3 = fma(a3.s1, b0.s3, acc3.s3);
#endif // M0 > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0.s2, b0.s0, acc0.s0);
        acc0.s1 = fma(a0.s2, b0.s1, acc0.s1);
        acc0.s2 = fma(a0.s2, b0.s2, acc0.s2);
        acc0.s3 = fma(a0.s2, b0.s3, acc0.s3);

#if M0 > 1

        acc1.s0 = fma(a1.s2, b0.s0, acc1.s0);
        acc1.s1 = fma(a1.s2, b0.s1, acc1.s1);
        acc1.s2 = fma(a1.s2, b0.s2, acc1.s2);
        acc1.s3 = fma(a1.s2, b0.s3, acc1.s3);

#endif // M0 > 1
#if M0 > 2

        acc2.s0 = fma(a2.s2, b0.s0, acc2.s0);
        acc2.s1 = fma(a2.s2, b0.s1, acc2.s1);
        acc2.s2 = fma(a2.s2, b0.s2, acc2.s2);
        acc2.s3 = fma(a2.s2, b0.s3, acc2.s3);

#endif // M0 > 2
#if M0 > 3

        acc3.s0 = fma(a3.s2, b0.s0, acc3.s0);
        acc3.s1 = fma(a3.s2, b0.s1, acc3.s1);
        acc3.s2 = fma(a3.s2, b0.s2, acc3.s2);
        acc3.s3 = fma(a3.s2, b0.s3, acc3.s3);
#endif // M0 > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0.s3, b0.s0, acc0.s0);
        acc0.s1 = fma(a0.s3, b0.s1, acc0.s1);
        acc0.s2 = fma(a0.s3, b0.s2, acc0.s2);
        acc0.s3 = fma(a0.s3, b0.s3, acc0.s3);

#if M0 > 1

        acc1.s0 = fma(a1.s3, b0.s0, acc1.s0);
        acc1.s1 = fma(a1.s3, b0.s1, acc1.s1);
        acc1.s2 = fma(a1.s3, b0.s2, acc1.s2);
        acc1.s3 = fma(a1.s3, b0.s3, acc1.s3);

#endif // M0 > 1
#if M0 > 2

        acc2.s0 = fma(a2.s3, b0.s0, acc2.s0);
        acc2.s1 = fma(a2.s3, b0.s1, acc2.s1);
        acc2.s2 = fma(a2.s3, b0.s2, acc2.s2);
        acc2.s3 = fma(a2.s3, b0.s3, acc2.s3);

#endif // M0 > 2
#if M0 > 3

        acc3.s0 = fma(a3.s3, b0.s0, acc3.s0);
        acc3.s1 = fma(a3.s3, b0.s1, acc3.s1);
        acc3.s2 = fma(a3.s3, b0.s2, acc3.s2);
        acc3.s3 = fma(a3.s3, b0.s3, acc3.s3);
#endif // M0 > 3

        src_addr.s0 += 4 * sizeof(float);
    }

    for(; i < (int)K; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if M0 > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // M0 > 1
#if M0 > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // M0 > 2
#if M0 > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // M0 > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0, b0.s0, acc0.s0);
        acc0.s1 = fma(a0, b0.s1, acc0.s1);
        acc0.s2 = fma(a0, b0.s2, acc0.s2);
        acc0.s3 = fma(a0, b0.s3, acc0.s3);
#if M0 > 1
        acc1.s0 = fma(a1, b0.s0, acc1.s0);
        acc1.s1 = fma(a1, b0.s1, acc1.s1);
        acc1.s2 = fma(a1, b0.s2, acc1.s2);
        acc1.s3 = fma(a1, b0.s3, acc1.s3);
#endif // M0 > 1
#if M0 > 2
        acc2.s0 = fma(a2, b0.s0, acc2.s0);
        acc2.s1 = fma(a2, b0.s1, acc2.s1);
        acc2.s2 = fma(a2, b0.s2, acc2.s2);
        acc2.s3 = fma(a2, b0.s3, acc2.s3);
#endif // M0 > 2
#if M0 > 3
        acc3.s0 = fma(a3, b0.s0, acc3.s0);
        acc3.s1 = fma(a3, b0.s1, acc3.s1);
        acc3.s2 = fma(a3, b0.s2, acc3.s2);
        acc3.s3 = fma(a3, b0.s3, acc3.s3);
#endif // M0 > 3

        src_addr.s0 += sizeof(float);
    }

    int z = get_global_id(2);

    // Compute dst address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                               PARTIAL_STORE_M0) * dst_stride_y);

    uint4 zout = 0;

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

    // The plane (zout) is calculated dividing row by HEIGHT_GEMM3D
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, float, acc, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float));

    LOAD_BLOCK(1, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, acc, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)4 * sizeof(float)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                                PARTIAL_STORE_M0)
                                * src2_stride_y)
                                + z * src2_stride_z;

    LOAD_BLOCK(M0, 4, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, float, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias
    ADD_BLOCK(M0, acc, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, float, VEC_SIZE, acc, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store the output block
    const bool cond_y = get_global_id(1) == 0;
    const bool cond_x = ((get_global_id(0) + 1) * 4 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(M0, 4, float, acc, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped
 *
 * @note This OpenCL kernel works with the 32-bit floating point data type (float) and uses the fma units.
 * This OpenCL kernel is optimized for Bifrost when the number of matrix B columns is less or equal to 1000.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DN0 and -DM0.
 * @note This kernel processed a fixed number of elements along x: -DN0=2.
 * @note The number of columns of matrix A and the number of columns of the matrix B need to be passed at compile time using -DK and -DN
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f32_bifrost_1000(IMAGE_DECLARATION(src0),
                                                      IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                      IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                      IMAGE_DECLARATION(dst),
                                                      uint src0_stride_z,
                                                      uint src1_stride_z,
#if defined(BETA)
                                                      uint src2_stride_z,
#endif //defined(BETA)
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
    // Requires 2 N0, C vect2, A vect4, B (2 vload2) // to fix for M0 > 1
    int idx = get_global_id(0) * N0;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * src0_stride_y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(float);

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

    // The plane (zin) is calculated dividing row by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
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

    // Initialize accumulators
    float2 acc0 = 0.0f;
#if M0 > 1
    float2 acc1 = 0.0f;
#endif // M0 > 1
#if M0 > 2
    float2 acc2 = 0.0f;
#endif // M0 > 2
#if M0 > 3
    float2 acc3 = 0.0f;
#endif // M0 > 3

    // A and B src indices get incremented at the same time.
    int i = 0;
    for(; i <= ((int)K - 8); i += 8)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float8 a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + zin.s0));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float8 a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0));
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float2 b0 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b1 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b2 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b3 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b4 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b5 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b6 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b7 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0.s0, b0.s0, acc0.s0);
        acc0.s0 = fma(a0.s1, b1.s0, acc0.s0);
        acc0.s0 = fma(a0.s2, b2.s0, acc0.s0);
        acc0.s0 = fma(a0.s3, b3.s0, acc0.s0);
        acc0.s0 = fma(a0.s4, b4.s0, acc0.s0);
        acc0.s0 = fma(a0.s5, b5.s0, acc0.s0);
        acc0.s0 = fma(a0.s6, b6.s0, acc0.s0);
        acc0.s0 = fma(a0.s7, b7.s0, acc0.s0);

        acc0.s1 = fma(a0.s0, b0.s1, acc0.s1);
        acc0.s1 = fma(a0.s1, b1.s1, acc0.s1);
        acc0.s1 = fma(a0.s2, b2.s1, acc0.s1);
        acc0.s1 = fma(a0.s3, b3.s1, acc0.s1);
        acc0.s1 = fma(a0.s4, b4.s1, acc0.s1);
        acc0.s1 = fma(a0.s5, b5.s1, acc0.s1);
        acc0.s1 = fma(a0.s6, b6.s1, acc0.s1);
        acc0.s1 = fma(a0.s7, b7.s1, acc0.s1);

#if M0 > 1
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0                    = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc1.s0 = fma(a0.s0, b0.s0, acc1.s0);
        acc1.s0 = fma(a0.s1, b1.s0, acc1.s0);
        acc1.s0 = fma(a0.s2, b2.s0, acc1.s0);
        acc1.s0 = fma(a0.s3, b3.s0, acc1.s0);
        acc1.s0 = fma(a0.s4, b4.s0, acc1.s0);
        acc1.s0 = fma(a0.s5, b5.s0, acc1.s0);
        acc1.s0 = fma(a0.s6, b6.s0, acc1.s0);
        acc1.s0 = fma(a0.s7, b7.s0, acc1.s0);

        acc1.s1 = fma(a0.s0, b0.s1, acc1.s1);
        acc1.s1 = fma(a0.s1, b1.s1, acc1.s1);
        acc1.s1 = fma(a0.s2, b2.s1, acc1.s1);
        acc1.s1 = fma(a0.s3, b3.s1, acc1.s1);
        acc1.s1 = fma(a0.s4, b4.s1, acc1.s1);
        acc1.s1 = fma(a0.s5, b5.s1, acc1.s1);
        acc1.s1 = fma(a0.s6, b6.s1, acc1.s1);
        acc1.s1 = fma(a0.s7, b7.s1, acc1.s1);
#endif // M0 > 1
#if M0 > 2
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0                    = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc2.s0 = fma(a0.s0, b0.s0, acc2.s0);
        acc2.s0 = fma(a0.s1, b1.s0, acc2.s0);
        acc2.s0 = fma(a0.s2, b2.s0, acc2.s0);
        acc2.s0 = fma(a0.s3, b3.s0, acc2.s0);
        acc2.s0 = fma(a0.s4, b4.s0, acc2.s0);
        acc2.s0 = fma(a0.s5, b5.s0, acc2.s0);
        acc2.s0 = fma(a0.s6, b6.s0, acc2.s0);
        acc2.s0 = fma(a0.s7, b7.s0, acc2.s0);

        acc2.s1 = fma(a0.s0, b0.s1, acc2.s1);
        acc2.s1 = fma(a0.s1, b1.s1, acc2.s1);
        acc2.s1 = fma(a0.s2, b2.s1, acc2.s1);
        acc2.s1 = fma(a0.s3, b3.s1, acc2.s1);
        acc2.s1 = fma(a0.s4, b4.s1, acc2.s1);
        acc2.s1 = fma(a0.s5, b5.s1, acc2.s1);
        acc2.s1 = fma(a0.s6, b6.s1, acc2.s1);
        acc2.s1 = fma(a0.s7, b7.s1, acc2.s1);
#endif // M0 > 2
#if M0 > 3
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0                    = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc3.s0 = fma(a0.s0, b0.s0, acc3.s0);
        acc3.s0 = fma(a0.s1, b1.s0, acc3.s0);
        acc3.s0 = fma(a0.s2, b2.s0, acc3.s0);
        acc3.s0 = fma(a0.s3, b3.s0, acc3.s0);
        acc3.s0 = fma(a0.s4, b4.s0, acc3.s0);
        acc3.s0 = fma(a0.s5, b5.s0, acc3.s0);
        acc3.s0 = fma(a0.s6, b6.s0, acc3.s0);
        acc3.s0 = fma(a0.s7, b7.s0, acc3.s0);

        acc3.s1 = fma(a0.s0, b0.s1, acc3.s1);
        acc3.s1 = fma(a0.s1, b1.s1, acc3.s1);
        acc3.s1 = fma(a0.s2, b2.s1, acc3.s1);
        acc3.s1 = fma(a0.s3, b3.s1, acc3.s1);
        acc3.s1 = fma(a0.s4, b4.s1, acc3.s1);
        acc3.s1 = fma(a0.s5, b5.s1, acc3.s1);
        acc3.s1 = fma(a0.s6, b6.s1, acc3.s1);
        acc3.s1 = fma(a0.s7, b7.s1, acc3.s1);
#endif // M0 > 3

        src_addr.s0 += sizeof(float) * 8;
    }
    // float size increment
    for(; i < (int)K; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if M0 > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // M0 > 1
#if M0 > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // M0 > 2
#if M0 > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // M0 > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float2 b0 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc0.s0 = fma(a0, b0.s0, acc0.s0);
        acc0.s1 = fma(a0, b0.s1, acc0.s1);
#if M0 > 1
        acc1.s0 = fma(a1, b0.s0, acc1.s0);
        acc1.s1 = fma(a1, b0.s1, acc1.s1);
#endif // M0 > 1
#if M0 > 2
        acc2.s0 = fma(a2, b0.s0, acc2.s0);
        acc2.s1 = fma(a2, b0.s1, acc2.s1);
#endif // M0 > 2
#if M0 > 3
        acc3.s0 = fma(a3, b0.s0, acc3.s0);
        acc3.s1 = fma(a3, b0.s1, acc3.s1);
#endif // M0 > 3

        src_addr.s0 += sizeof(float);
    }

    int z = get_global_id(2);

    // Compute dst address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)2 * sizeof(float)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                               PARTIAL_STORE_M0) * dst_stride_y);

    uint4 zout = 0;

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

    // The plane (zout) is calculated dividing row by HEIGHT_GEMM3D
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, float, acc, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)2 * sizeof(float));

    LOAD_BLOCK(1, 2, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, acc, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)2 * sizeof(float)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                                PARTIAL_STORE_M0)
                                * src2_stride_y)
                                + z * src2_stride_z;

    LOAD_BLOCK(M0, 2, float, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, float, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias
    ADD_BLOCK(M0, acc, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, float, VEC_SIZE, acc, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store the output block
    const bool cond_y = get_global_id(1) == 0;
    const bool cond_x = ((get_global_id(0) + 1) * 2 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(M0, 2, float, acc, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * @note This OpenCL kernel works with the 16-bit floating point data type (half) and accumulating the result in a 32 floating point variable.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DN0 and -DM0.
 * @note This kernel processed a fixed number of elements along x: -DN0=8.
 * @note The number of columns of matrix A and the number of columns of the matrix B need to be passed at compile time using -DK and -DN
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f16_bifrost_acc32(IMAGE_DECLARATION(src0),
                                                       IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                       IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                       IMAGE_DECLARATION(dst),
                                                       uint src0_stride_z,
                                                       uint src1_stride_z,
#if defined(BETA)
                                                       uint src2_stride_z,
#endif //defined(BETA)
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
    int idx = get_global_id(0) * N0;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * src0_stride_y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(half);

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

    // The plane (zin) is calculated dividing row by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
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

    float8 acc0 = 0.0h;
#if M0 > 1
    float8 acc1 = 0.0h;
#endif // M0 > 1
#if M0 > 2
    float8 acc2 = 0.0h;
#endif // M0 > 2
#if M0 > 3
    float8 acc3 = 0.0h;
#endif // M0 > 3

    int i = 0;
    for(; i <= ((int)K - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        LOAD_BLOCK(M0, 4, half, a, src0_ptr, src_addr.s0, src0_stride_y, zin.s);
#else // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float8 b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;

        // Accumulate
        acc0 = fma(b0, (float8)a0.s0, acc0);
#if M0 > 1
        acc1 = fma(b0, (float8)a1.s0, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (float8)a2.s0, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (float8)a3.s0, acc3);
#endif // M0 > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s1, acc0);
#if M0 > 1
        acc1 = fma(b0, (float8)a1.s1, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (float8)a2.s1, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (float8)a3.s1, acc3);
#endif // M0 > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s2, acc0);
#if M0 > 1
        acc1 = fma(b0, (float8)a1.s2, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (float8)a2.s2, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (float8)a3.s2, acc3);
#endif // M0 > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s3, acc0);
#if M0 > 1
        acc1 = fma(b0, (float8)a1.s3, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (float8)a2.s3, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (float8)a3.s3, acc3);
#endif // M0 > 3

        src_addr.s0 += 4 * sizeof(half);
    }

    for(; i < (int)K; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if M0 > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // M0 > 1
#if M0 > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // M0 > 2
#if M0 > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // M0 > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float8 b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));

        src_addr += (int2)(sizeof(half), src1_stride_y);

        // Accumulate
        acc0 = fma(b0, (float8)a0, acc0); // b0 * (half8)a0;
#if M0 > 1
        acc1 = fma(b0, (float8)a1, acc1); // b0 * (half8)a1;
#endif                                    // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (float8)a2, acc2); // b0 * (half8)a2;
#endif                                    // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (float8)a3, acc3); // b0 * (half8)a3;
#endif                                    // M0 > 3
    }

    int z = get_global_id(2);

    // Compute dst address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * dst_stride_y);

    uint4 zout = 0;

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

    // The plane (zout) is calculated dividing row by HEIGHT_GEMM3D
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, float, acc, ALPHA);
#endif // defined(ALPHA)

#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half));

    LOAD_BLOCK(1, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

    float8 bias_f0 = convert_float8(bias0);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, float, bias_f, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, acc, bias_f0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                                PARTIAL_STORE_M0)
                                * src2_stride_y)
                                + z * src2_stride_z;

    LOAD_BLOCK(M0, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

    float8 bias_f0 = convert_float8(bias0);
#if M0 > 1
    float8 bias_f1 = convert_float8(bias1);
#endif // M0 > 1
#if M0 > 2
    float8 bias_f2 = convert_float8(bias2);
#endif // M0 > 2
#if M0 > 3
    float8 bias_f3 = convert_float8(bias3);
#endif // M0 > 3

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, float, bias_f, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias
    ADD_BLOCK(M0, acc, bias_f);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    half8 acc_h0 = convert_half8(acc0);
#if M0 > 1
    half8 acc_h1 = convert_half8(acc1);
#endif // M0 > 1
#if M0 > 2
    half8 acc_h2 = convert_half8(acc2);
#endif // M0 > 2
#if M0 > 3
    half8 acc_h3 = convert_half8(acc3);
#endif // M0 > 3

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, half, VEC_SIZE, acc_h, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store the output block
    const bool cond_y = get_global_id(1) == 0;
    const bool cond_x = ((get_global_id(0) + 1) * 8 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(M0, 8, half, acc_h, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * @note This OpenCL kernel works with the 16-bit floating point data type (half) and uses the fma units.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DN0 and -DM0.
 * @note This kernel processed a fixed number of elements along x: -DN0=8.
 * @note The number of columns of matrix A and the number of columns of the matrix B need to be passed at compile time using -DK and -DN
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note The optional alpha's value need to be passed at compile time using -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (e.g. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (e.g. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  src2_step_y                        (Optional) src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src2_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f16_bifrost(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(BETA)
                                                 IMAGE_DECLARATION(src2),
#endif // defined(BETA)
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
#if defined(BETA)
                                                 uint src2_stride_z,
#endif //defined(BETA)
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
    int idx = get_global_id(0) * N0;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * src0_stride_y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(half);

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

    // The plane (zin) is calculated dividing row by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
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

    half8 acc0 = 0.0h;
#if M0 > 1
    half8 acc1 = 0.0h;
#endif // M0 > 1
#if M0 > 2
    half8 acc2 = 0.0h;
#endif // M0 > 2
#if M0 > 3
    half8 acc3 = 0.0h;
#endif // M0 > 3

    int i = 0;
    for(; i <= ((int)K - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        LOAD_BLOCK(M0, 4, half, a, src0_ptr, src_addr.s0, src0_stride_y, zin.s);
#else // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Accumulate
        acc0 = fma(b0, (half8)a0.s0, acc0);
#if M0 > 1
        acc1 = fma(b0, (half8)a1.s0, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (half8)a2.s0, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (half8)a3.s0, acc3);
#endif // M0 > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s1, acc0);
#if M0 > 1
        acc1 = fma(b0, (half8)a1.s1, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (half8)a2.s1, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (half8)a3.s1, acc3);
#endif // M0 > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s2, acc0);
#if M0 > 1
        acc1 = fma(b0, (half8)a1.s2, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (half8)a2.s2, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (half8)a3.s2, acc3);
#endif // M0 > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s3, acc0);
#if M0 > 1
        acc1 = fma(b0, (half8)a1.s3, acc1);
#endif // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (half8)a2.s3, acc2);
#endif // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (half8)a3.s3, acc3);
#endif // M0 > 3

        src_addr.s0 += 4 * sizeof(half);
    }

    for(; i < (int)K; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if M0 > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // M0 > 1
#if M0 > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // M0 > 2
#if M0 > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // M0 > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if M0 > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // M0 > 1
#if M0 > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // M0 > 2
#if M0 > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // M0 > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));

        src_addr += (int2)(sizeof(half), src1_stride_y);

        // Accumulate
        acc0 = fma(b0, (half8)a0, acc0); // b0 * (half8)a0;
#if M0 > 1
        acc1 = fma(b0, (half8)a1, acc1); // b0 * (half8)a1;
#endif                                   // M0 > 1
#if M0 > 2
        acc2 = fma(b0, (half8)a2, acc2); // b0 * (half8)a2;
#endif                                   // M0 > 2
#if M0 > 3
        acc3 = fma(b0, (half8)a3, acc3); // b0 * (half8)a3;
#endif                                   // M0 > 3
    }

    int z = get_global_id(2);

    // Compute dst address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0) * dst_stride_y);

    uint4 zout = 0;

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

    // The plane (zout) is calculated dividing row by HEIGHT_GEMM3D
    zout = ((uint4)(0, 1, 2, 3) + (uint4)(COMPUTE_M0_START_ROW(get_global_id(1), M0, PARTIAL_STORE_M0))) / (uint4)HEIGHT_GEMM3D;
    zout = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;
#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;
#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, half, acc, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

#if defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half));

    LOAD_BLOCK(1, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, half, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, acc, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *src2_addr = src2_ptr + src2_offset_first_element_in_bytes + (get_global_id(0) * (uint)8 * sizeof(half)) + (COMPUTE_M0_START_ROW(get_global_id(1), M0,
                                PARTIAL_STORE_M0)
                                * src2_stride_y)
                                + z * src2_stride_z;

    LOAD_BLOCK(M0, 8, half, bias, src2_addr, 0, src2_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, half, bias, BETA);
#endif // UNIT_BIAS

    // acc = acc + bias
    ADD_BLOCK(M0, acc, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, half, VEC_SIZE, acc, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    // Store the output block
    const bool cond_y = get_global_id(1) == 0;
    const bool cond_x = ((get_global_id(0) + 1) * 8 >= N);
    STORE_BLOCK_BOUNDARY_AWARE(M0, 8, half, acc, dst_addr, dst_stride_y, zout.s, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
}
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)

#endif // defined(N) && defined(K) && defined(M0) && defined(N0) && defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)