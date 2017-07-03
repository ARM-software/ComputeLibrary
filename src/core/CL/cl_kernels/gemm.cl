/*
 * Copyright (c) 2017 ARM Limited.
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
#include "fixed_point.h"
#include "helpers.h"

/** This OpenCL kernel computes the "vector" 1x4 transposition of input matrix
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_transpose1x4(IMAGE_DECLARATION(src),
                                IMAGE_DECLARATION(dst))
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    /* Compute address for Matrix B - source */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    uint dst_addr_in_bytes = y * 16 + ((x * dst_stride_y + dst_offset_first_element_in_bytes));

    uint4 b0 = vload4(0, (__global uint *)src.ptr);

    vstore4(b0, 0, (__global uint *)(dst_ptr + dst_addr_in_bytes));
}

/** This OpenCL kernel computes the "vector" 1x8 transposition of input matrix
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U16/S16/QS16/F16
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_transpose1x8(IMAGE_DECLARATION(src),
                                IMAGE_DECLARATION(dst))
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    /* Compute address for Matrix B - source */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    uint dst_addr_in_bytes = y * 16 + ((x * dst_stride_y + dst_offset_first_element_in_bytes));

    ushort8 b0 = vload8(0, (__global ushort *)src.ptr);

    vstore8(b0, 0, (__global ushort *)(dst_ptr + dst_addr_in_bytes));
}

/** This OpenCL kernel computes the "vector" 1x16 transposition of input matrix
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/QS8
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_transpose1x16(IMAGE_DECLARATION(src),
                                 IMAGE_DECLARATION(dst))
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    /* Compute address for Matrix B - source */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    uint dst_addr_in_bytes = y * 16 + ((x * dst_stride_y + dst_offset_first_element_in_bytes));

    uchar16 b0 = vload16(0, (__global uchar *)src.ptr);

    vstore16(b0, 0, (__global uchar *)(dst_ptr + dst_addr_in_bytes));
}

/** This OpenCL kernel reshapes the input matrix transposing each 4x4 block and interleaving the values
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_interleave4x4_32bit(IMAGE_DECLARATION(src),
                                       IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from Matrix A */
    float4 a0 = vload4(0, (__global float *)(offset(&src, 0, 0)));
    float4 a1 = vload4(0, (__global float *)(offset(&src, 0, 1)));
    float4 a2 = vload4(0, (__global float *)(offset(&src, 0, 2)));
    float4 a3 = vload4(0, (__global float *)(offset(&src, 0, 3)));

    float4 val0 = (float4)(a0.s0, a1.s0, a2.s0, a3.s0);
    vstore4(val0, 0, ((__global float *)dst.ptr) + 0);

    val0 = (float4)(a0.s1, a1.s1, a2.s1, a3.s1);
    vstore4(val0, 0, ((__global float *)dst.ptr) + 4);

    val0 = (float4)(a0.s2, a1.s2, a2.s2, a3.s2);
    vstore4(val0, 0, ((__global float *)dst.ptr) + 8);

    val0 = (float4)(a0.s3, a1.s3, a2.s3, a3.s3);
    vstore4(val0, 0, ((__global float *)dst.ptr) + 12);
}

/** This OpenCL kernel reshapes the input matrix transposing each 4x4 block and interleaving the values
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U16/S16/QS16/F16
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_interleave4x4_16bit(IMAGE_DECLARATION(src),
                                       IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from Matrix A */
    half8 a0 = vload8(0, (__global half *)(offset(&src, 0, 0)));
    half8 a1 = vload8(0, (__global half *)(offset(&src, 0, 1)));
    half8 a2 = vload8(0, (__global half *)(offset(&src, 0, 2)));
    half8 a3 = vload8(0, (__global half *)(offset(&src, 0, 3)));

    half8 val0 = (half8)((half4)(a0.s0, a1.s0, a2.s0, a3.s0), (half4)(a0.s1, a1.s1, a2.s1, a3.s1));
    vstore8(val0, 0, ((__global half *)dst.ptr) + 0);

    val0 = (half8)((half4)(a0.s2, a1.s2, a2.s2, a3.s2), (half4)(a0.s3, a1.s3, a2.s3, a3.s3));
    vstore8(val0, 0, ((__global half *)dst.ptr) + 8);

    val0 = (half8)((half4)(a0.s4, a1.s4, a2.s4, a3.s4), (half4)(a0.s5, a1.s5, a2.s5, a3.s5));
    vstore8(val0, 0, ((__global half *)dst.ptr) + 16);

    val0 = (half8)((half4)(a0.s6, a1.s6, a2.s6, a3.s6), (half4)(a0.s7, a1.s7, a2.s7, a3.s7));
    vstore8(val0, 0, ((__global half *)dst.ptr) + 24);
}

/** This OpenCL kernel reshapes the input matrix transposing each 4x4 block and interleaving the values
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/QS8
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_interleave4x4_8bit(IMAGE_DECLARATION(src),
                                      IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from Matrix A */
    uchar16 a0 = vload16(0, (__global uchar *)(offset(&src, 0, 0)));
    uchar16 a1 = vload16(0, (__global uchar *)(offset(&src, 0, 1)));
    uchar16 a2 = vload16(0, (__global uchar *)(offset(&src, 0, 2)));
    uchar16 a3 = vload16(0, (__global uchar *)(offset(&src, 0, 3)));

    uchar16 val0 = (uchar16)((uchar4)(a0.s0, a1.s0, a2.s0, a3.s0), (uchar4)(a0.s1, a1.s1, a2.s1, a3.s1),
                             (uchar4)(a0.s2, a1.s2, a2.s2, a3.s2), (uchar4)(a0.s3, a1.s3, a2.s3, a3.s3));
    vstore16(val0, 0, ((__global uchar *)dst.ptr) + 0);

    val0 = (uchar16)((uchar4)(a0.s4, a1.s4, a2.s4, a3.s4), (uchar4)(a0.s5, a1.s5, a2.s5, a3.s5),
                     (uchar4)(a0.s6, a1.s6, a2.s6, a3.s6), (uchar4)(a0.s7, a1.s7, a2.s7, a3.s7));
    vstore16(val0, 0, ((__global uchar *)dst.ptr) + 16);

    val0 = (uchar16)((uchar4)(a0.s8, a1.s8, a2.s8, a3.s8), (uchar4)(a0.s9, a1.s9, a2.s9, a3.s9),
                     (uchar4)(a0.sA, a1.sA, a2.sA, a3.sA), (uchar4)(a0.sB, a1.sB, a2.sB, a3.sB));
    vstore16(val0, 0, ((__global uchar *)dst.ptr) + 32);

    val0 = (uchar16)((uchar4)(a0.sC, a1.sC, a2.sC, a3.sC), (uchar4)(a0.sD, a1.sD, a2.sD, a3.sD),
                     (uchar4)(a0.sE, a1.sE, a2.sE, a3.sE), (uchar4)(a0.sF, a1.sF, a2.sF, a3.sF));
    vstore16(val0, 0, ((__global uchar *)dst.ptr) + 48);
}

/** This kernel accumulates each row with the biases vector
 *
 * @note The data type must be passed at compile time -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in, out] accum_ptr                            Pointer to the accumulate tensor. Supported data type: U8/S8/QS8/U16/S16/F16/U32/S32/F32
 * @param[in]      accum_stride_x                       Stride of the accmulate tensor in X dimension (in bytes)
 * @param[in]      accum_step_x                         accum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      accum_stride_y                       Stride of the accumlulate tensor in Y dimension (in bytes)
 * @param[in]      accum_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      accum_offset_first_element_in_bytes  The offset of the first element in the accumulate tensor
 * @param[in]      biases_ptr                           Pointer to the biases vector. Same as @p accum_ptr
 * @param[in]      biases_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]      biases_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      biases_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
#ifdef DATA_TYPE
__kernel void gemm_accumulate_biases(
    IMAGE_DECLARATION(accum),
    VECTOR_DECLARATION(biases))
{
    Image  accum  = CONVERT_TO_IMAGE_STRUCT(accum);
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    accum_value = vload16(0, (__global DATA_TYPE *)accum.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    biases_value = vload16(0, (__global DATA_TYPE *)biases.ptr);
    accum_value  = biases_value + accum_value;

    // Store result in the accummulate buffer
    vstore16(accum_value, 0, (__global DATA_TYPE *)accum.ptr);
}
#endif /* DATA_TYPE */

#ifdef WIDTH_MATRIX_B
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_8bit and @ref gemm_transpose1x16 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_MATRIX_B
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported formats: U8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported formats: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported formats: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  a_offset                           Offset to be added to each element of the matrix A
 * @param[in]  b_offset                           Offset to be added to each element of the matrix B.
 * @param[in]  c_offset                           Offset to be added to each element of the matrix C.
 * @param[in]  c_mult_int                         Multiplied with each element of the matrix C.
 * @param[in]  shift                              Number of bits to shift right the result.
 */
__kernel void gemm_mm_u8(IMAGE_DECLARATION(src0),
                         IMAGE_DECLARATION(src1),
                         IMAGE_DECLARATION(dst),
                         int a_offset,
                         int b_offset,
                         int c_offset,
                         int c_mult_int,
                         int shift)
{
    /* src_addr.s0 = address of matrix A */
    /* src_addr.s1 = address of matrix B */

    /* Compute address for matrix A and B */
    int2 src_addr = (int2)(get_global_id(1), get_global_id(0)) * (int2)((src0_stride_y),
                                                                        (src1_stride_y));

    /* Add offset_first_element_in_bytes */
    src_addr = src_addr + ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    /* Compute end row address for matrix B */
    int end_row_mtx_b = src_addr.s1 + WIDTH_MATRIX_B;

    /* Reset accumulators */
    int16 c00 = 0.0f;
    int16 c10 = 0.0f;
    int16 c20 = 0.0f;
    int16 c30 = 0.0f;

    for(; src_addr.s1 <= (end_row_mtx_b - 8); src_addr += (int2)(8, 32))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        int8 a0  = (int8)a_offset + convert_int8(vload8(0, ((__global uchar *)src0_ptr) + src_addr.s0));
        int16 b0 = (int16)b_offset + convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1));

        c00 += (int16)a0.s0 * b0;
        c10 += (int16)a0.s1 * b0;
        c20 += (int16)a0.s2 * b0;
        c30 += (int16)a0.s3 * b0;

        int16 b1 = (int16)b_offset + convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1 + 16));

        c00 += (int16)a0.s4 * b1;
        c10 += (int16)a0.s5 * b1;
        c20 += (int16)a0.s6 * b1;
        c30 += (int16)a0.s7 * b1;
    }

    for(; src_addr.s1 < end_row_mtx_b; src_addr += (int2)(4, 16))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        int4 a0  = (int4)a_offset + convert_int4(vload4(0, ((__global uchar *)src0_ptr) + src_addr.s0));
        int16 b0 = (int16)b_offset + convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1));

        c00 += (int16)a0.s0 * b0;
        c10 += (int16)a0.s1 * b0;
        c20 += (int16)a0.s2 * b0;
        c30 += (int16)a0.s3 * b0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of matrix product */
    c00 = (((int16)c_offset + c00) * (int16)c_mult_int) >> shift;
    c10 = (((int16)c_offset + c10) * (int16)c_mult_int) >> shift;
    c20 = (((int16)c_offset + c20) * (int16)c_mult_int) >> shift;
    c30 = (((int16)c_offset + c30) * (int16)c_mult_int) >> shift;

    /* Store 4x16 block */
    vstore16(convert_uchar16_sat(c00), 0, (__global uchar *)(offset(&dst, 0, 0)));
    vstore16(convert_uchar16_sat(c10), 0, (__global uchar *)(offset(&dst, 0, 1)));
    vstore16(convert_uchar16_sat(c20), 0, (__global uchar *)(offset(&dst, 0, 2)));
    vstore16(convert_uchar16_sat(c30), 0, (__global uchar *)(offset(&dst, 0, 3)));
}
#endif /* WIDTH_MATRIX_B */

#if defined(WIDTH_MATRIX_B) && defined(ALPHA)
/** This OpenCL kernel is optimised for Midgard. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_MATRIX_B and -DALPHA
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_mm_f32_midgard(IMAGE_DECLARATION(src0),
                                  IMAGE_DECLARATION(src1),
                                  IMAGE_DECLARATION(dst))
{
    /* src_addr.s0 = address of matrix A */
    /* src_addr.s1 = address of matrix B */

    /* Compute address for matrix A and B */
    int2 src_addr = (int2)(get_global_id(1), get_global_id(0)) * (int2)((src0_stride_y),
                                                                        (src1_stride_y));

    /* Add offset_first_element_in_bytes */
    src_addr = src_addr + ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    /* Divide by 4 in order to get the src_addr in unit of float */
    src_addr = src_addr >> 2;

    /* Compute end row address for matrix B */
    int end_row_mtx_b = src_addr.s1 + WIDTH_MATRIX_B;

    /* Reset accumulators */
    float4 c00 = 0.0f;
    float4 c10 = 0.0f;
    float4 c20 = 0.0f;
    float4 c30 = 0.0f;

    for(; src_addr.s1 <= (end_row_mtx_b - 8); src_addr += (int2)(8, 8))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        float4 a0 = vload4(0, ((__global float *)src0_ptr) + src_addr.s0);
        float4 b0 = vload4(0, ((__global float *)src1_ptr) + src_addr.s1);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;

        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        a0 = vload4(0, ((__global float *)src0_ptr) + src_addr.s0 + 4);
        b0 = vload4(0, ((__global float *)src1_ptr) + src_addr.s1 + 4);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;
    }

    for(; src_addr.s1 < end_row_mtx_b; src_addr += (int2)(4, 4))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        float4 a0 = vload4(0, ((__global float *)src0_ptr) + src_addr.s0);
        float4 b0 = vload4(0, ((__global float *)src1_ptr) + src_addr.s1);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of matrix product */
    c00 = c00 * (float4)ALPHA;
    c10 = c10 * (float4)ALPHA;
    c20 = c20 * (float4)ALPHA;
    c30 = c30 * (float4)ALPHA;

    /* Store 4x4 block */
    vstore4(c00, 0, (__global float *)(offset(&dst, 0, 0)));
    vstore4(c10, 0, (__global float *)(offset(&dst, 0, 1)));
    vstore4(c20, 0, (__global float *)(offset(&dst, 0, 2)));
    vstore4(c30, 0, (__global float *)(offset(&dst, 0, 3)));
}

/** This OpenCL kernel is optimised for Bifrost. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_MATRIX_B and -DALPHA
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_mm_f32_bifrost(IMAGE_DECLARATION(src0),
                                  IMAGE_DECLARATION(src1),
                                  IMAGE_DECLARATION(dst))
{
    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global float *src_addr_a = (__global float *)(src0_ptr + get_global_id(1) * src0_stride_y + src0_offset_first_element_in_bytes);
    __global float *src_addr_b = (__global float *)(src1_ptr + get_global_id(0) * src1_stride_y + src1_offset_first_element_in_bytes);

    // Compute end row address for matrix B
    __global float *src_end_addr_b = src_addr_b + WIDTH_MATRIX_B;

    // Reset accumulators
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c02 = 0.0f;
    float c03 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    float c12 = 0.0f;
    float c13 = 0.0f;
    float c20 = 0.0f;
    float c21 = 0.0f;
    float c22 = 0.0f;
    float c23 = 0.0f;
    float c30 = 0.0f;
    float c31 = 0.0f;
    float c32 = 0.0f;
    float c33 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - 16); src_addr_a += 16, src_addr_b += 16)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 4);
        b0 = vload4(0, src_addr_b + 4);

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 8);
        b0 = vload4(0, src_addr_b + 8);

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 12);
        b0 = vload4(0, src_addr_b + 12);

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4, src_addr_b += 4)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Multiply by the weight of matrix product
    c00 = c00 * ALPHA;
    c01 = c01 * ALPHA;
    c02 = c02 * ALPHA;
    c03 = c03 * ALPHA;
    c10 = c10 * ALPHA;
    c11 = c11 * ALPHA;
    c12 = c12 * ALPHA;
    c13 = c13 * ALPHA;
    c20 = c20 * ALPHA;
    c21 = c21 * ALPHA;
    c22 = c22 * ALPHA;
    c23 = c23 * ALPHA;
    c30 = c30 * ALPHA;
    c31 = c31 * ALPHA;
    c32 = c32 * ALPHA;
    c33 = c33 * ALPHA;

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Store 4x4 block
    vstore4((float4)(c00, c01, c02, c03), 0, (__global float *)(offset(&dst, 0, 0)));
    vstore4((float4)(c10, c11, c12, c13), 0, (__global float *)(offset(&dst, 0, 1)));
    vstore4((float4)(c20, c21, c22, c23), 0, (__global float *)(offset(&dst, 0, 2)));
    vstore4((float4)(c30, c31, c32, c33), 0, (__global float *)(offset(&dst, 0, 3)));
}

/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_16bit and @ref gemm_transpose1x8 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_MATRIX_B and -DALPHA
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_mm_f16(IMAGE_DECLARATION(src0),
                          IMAGE_DECLARATION(src1),
                          IMAGE_DECLARATION(dst))
{
    /* src_addr.s0 = address of matrix A */
    /* src_addr.s1 = address of matrix B */

    /* Compute address for matrix A and B */
    int2 src_addr = (int2)(get_global_id(1), get_global_id(0)) * (int2)((src0_stride_y),
                                                                        (src1_stride_y));

    /* Add offset_first_element_in_bytes */
    src_addr = src_addr + ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    /* Divide by 2 in order to get the src_addr in unit of half */
    src_addr = src_addr >> 1;

    /* Compute end row address for matrix B */
    int end_row_mtx_b = src_addr.s1 + WIDTH_MATRIX_B;

    /* Reset accumulators */
    half8 c00 = 0.0f;
    half8 c10 = 0.0f;
    half8 c20 = 0.0f;
    half8 c30 = 0.0f;

    for(; src_addr.s1 <= (end_row_mtx_b - 8); src_addr += (int2)(8, 16))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        half4 a0 = vload4(0, ((__global half *)src0_ptr) + src_addr.s0);
        half8 b0 = vload8(0, ((__global half *)src1_ptr) + src_addr.s1);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;

        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        a0 = vload4(0, ((__global half *)src0_ptr) + src_addr.s0 + 4);
        b0 = vload8(0, ((__global half *)src1_ptr) + src_addr.s1 + 8);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;
    }

    for(; src_addr.s1 < end_row_mtx_b; src_addr += (int2)(4, 8))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        half4 a0 = vload4(0, ((__global half *)src0_ptr) + src_addr.s0);
        half8 b0 = vload8(0, ((__global half *)src1_ptr) + src_addr.s1);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of matrix product */
    c00 = c00 * (half8)ALPHA;
    c10 = c10 * (half8)ALPHA;
    c20 = c20 * (half8)ALPHA;
    c30 = c30 * (half8)ALPHA;

    /* Store 4x8 block */
    vstore8(c00, 0, (__global half *)(offset(&dst, 0, 0)));
    vstore8(c10, 0, (__global half *)(offset(&dst, 0, 1)));
    vstore8(c20, 0, (__global half *)(offset(&dst, 0, 2)));
    vstore8(c30, 0, (__global half *)(offset(&dst, 0, 3)));
}

#ifdef FIXED_POINT_POSITION
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1) in 8 bit fixed point precision
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_8bit and @ref gemm_transpose1x16 before running the matrix multiplication
 *
 * @attention The width of matrix B, the alpha's value and fixed point position need to be passed at compile time using -DWIDTH_MATRIX_B -DALPHA and -DFIXED_POINT_POSITION
 *
 * @note: ALPHA must be passed in 8 bit fixed point format
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: QS8
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_mm_qs8(IMAGE_DECLARATION(src0),
                          IMAGE_DECLARATION(src1),
                          IMAGE_DECLARATION(dst))
{
    /* src_addr.s0 = address of matrix A */
    /* src_addr.s1 = address of matrix B */

    /* Compute address for matrix A and B */
    int2 src_addr = (int2)(get_global_id(1), get_global_id(0)) * (int2)((src0_stride_y),
                                                                        (src1_stride_y));

    /* Add offset_first_element_in_bytes */
    src_addr = src_addr + ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    /* Compute end row address for matrix B */
    int end_row_mtx_b = src_addr.s1 + WIDTH_MATRIX_B;

    /* Reset accumulators */
    short8 c00 = 0.0f;
    short8 c10 = 0.0f;
    short8 c20 = 0.0f;
    short8 c30 = 0.0f;
    short8 c01 = 0.0f;
    short8 c11 = 0.0f;
    short8 c21 = 0.0f;
    short8 c31 = 0.0f;

    /* This for loop performs 1 accumulation for each iteration */
    for(; src_addr.s1 <= (end_row_mtx_b - 16); src_addr += (int2)(4, 16))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        char4  a0 = vload4(0, ((__global char *)src0_ptr) + src_addr.s0);
        char16 b0 = vload16(0, ((__global char *)src1_ptr) + src_addr.s1);

        c00 = mlal_sat_qs8x8(c00, (char8)a0.s0, b0.s01234567, FIXED_POINT_POSITION);
        c10 = mlal_sat_qs8x8(c10, (char8)a0.s1, b0.s01234567, FIXED_POINT_POSITION);
        c20 = mlal_sat_qs8x8(c20, (char8)a0.s2, b0.s01234567, FIXED_POINT_POSITION);
        c30 = mlal_sat_qs8x8(c30, (char8)a0.s3, b0.s01234567, FIXED_POINT_POSITION);

        c01 = mlal_sat_qs8x8(c01, (char8)a0.s0, b0.s89ABCDEF, FIXED_POINT_POSITION);
        c11 = mlal_sat_qs8x8(c11, (char8)a0.s1, b0.s89ABCDEF, FIXED_POINT_POSITION);
        c21 = mlal_sat_qs8x8(c21, (char8)a0.s2, b0.s89ABCDEF, FIXED_POINT_POSITION);
        c31 = mlal_sat_qs8x8(c31, (char8)a0.s3, b0.s89ABCDEF, FIXED_POINT_POSITION);
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of matrix product */
    char16 c00_qs8 = convert_char16_sat((short16)(c00, c01));
    char16 c10_qs8 = convert_char16_sat((short16)(c10, c11));
    char16 c20_qs8 = convert_char16_sat((short16)(c20, c21));
    char16 c30_qs8 = convert_char16_sat((short16)(c30, c31));

    c00_qs8 = mul_sat_qs8x16(c00_qs8, (char16)ALPHA, FIXED_POINT_POSITION);
    c10_qs8 = mul_sat_qs8x16(c10_qs8, (char16)ALPHA, FIXED_POINT_POSITION);
    c20_qs8 = mul_sat_qs8x16(c20_qs8, (char16)ALPHA, FIXED_POINT_POSITION);
    c30_qs8 = mul_sat_qs8x16(c30_qs8, (char16)ALPHA, FIXED_POINT_POSITION);

    /* Store 16x4 block */
    vstore16(c00_qs8, 0, (__global char *)(offset(&dst, 0, 0)));
    vstore16(c10_qs8, 0, (__global char *)(offset(&dst, 0, 1)));
    vstore16(c20_qs8, 0, (__global char *)(offset(&dst, 0, 2)));
    vstore16(c30_qs8, 0, (__global char *)(offset(&dst, 0, 3)));
}
#endif /* FIXED_POINT_POSITION */

#ifdef WIDTH_VECTOR_A
/** This OpenCL kernel computes the vector by matrix multiplication between the vector A (src0) and matrix B (src1)
 *
 * @attention The width of vector A, the width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_VECTOR_A -DWIDTH_MATRIX_B and -DALPHA
 *
 * @attention The input vector A and matrix B must not be reshaped
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_vm_f32(IMAGE_DECLARATION(src0),
                          IMAGE_DECLARATION(src1),
                          IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * 4;

    /* Compute the address for the vector A and matrix B */
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));
    src_addr.s1 += idx * sizeof(float);

    int end_row_vec_a = src_addr.s0 + (WIDTH_VECTOR_A * sizeof(float));

    float4 acc = 0.0f;

    for(; src_addr.s0 <= (end_row_vec_a - 2 * sizeof(float)); src_addr += (int2)(2 * sizeof(float), 2 * src1_stride_y))
    {
        float2 a0 = vload2(0, (__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        float4 b1 = vload4(0, (__global float *)(src1_ptr + src_addr.s1 + src1_stride_y));

        acc += b0 * (float4)a0.s0;
        acc += b1 * (float4)a0.s1;
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(float), src1_stride_y))
    {
        float  a0 = *((__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));

        acc += b0 * (float4)a0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of vector-matrix product */
    acc = acc * (float4)ALPHA;

    vstore4(acc, 0, (__global float *)(offset(&dst, 0, 0)));
}

/** This OpenCL kernel computes the vector by matrix multiplication between the vector A (src0) and matrix B (src1)
 *
 * @attention The width of vector A, the width of matrix B and the alpha's value need to be passed at compile time using -DWIDTH_VECTOR_A -DWIDTH_MATRIX_B and -DALPHA
 *
 * @attention The input vector A and matrix B must not be reshaped
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_vm_f16(IMAGE_DECLARATION(src0),
                          IMAGE_DECLARATION(src1),
                          IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * 8;

    /* Compute the address for the vector A and matrix B */
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));
    src_addr.s1 += idx * sizeof(half);

    int end_row_vec_a = src_addr.s0 + (WIDTH_VECTOR_A * sizeof(half));

    half8 acc = 0.0f;

    for(; src_addr.s0 <= (end_row_vec_a - 4 * sizeof(half)); src_addr += (int2)(4 * sizeof(half), 4 * src1_stride_y))
    {
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0));
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1 + 0 * src1_stride_y));
        half8 b1 = vload8(0, (__global half *)(src1_ptr + src_addr.s1 + 1 * src1_stride_y));
        half8 b2 = vload8(0, (__global half *)(src1_ptr + src_addr.s1 + 2 * src1_stride_y));
        half8 b3 = vload8(0, (__global half *)(src1_ptr + src_addr.s1 + 3 * src1_stride_y));

        acc += b0 * (half8)a0.s0;
        acc += b1 * (half8)a0.s1;
        acc += b2 * (half8)a0.s2;
        acc += b3 * (half8)a0.s3;
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(half), src1_stride_y))
    {
        half a0  = *((__global half *)(src0_ptr + src_addr.s0));
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));

        acc += b0 * (half8)a0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of vector-matrix product */
    acc = acc * (half8)ALPHA;

    vstore8(acc, 0, (__global half *)(offset(&dst, 0, 0)));
}

#ifdef FIXED_POINT_POSITION
/** This OpenCL kernel computes the vector by matrix multiplication between the vector A (src0) and matrix B (src1) in 8 bit fixed point
 *
 * @attention The width of vector A, the width of matrix B, the alpha's value and the fixed point position need to be passed at compile time using -DWIDTH_VECTOR_A -DWIDTH_MATRIX_B, -DALPHA and -DFIXED_POINT_POSITION
 *
 * @attention The input vector A and matrix B must not be reshaped
 *
 * @note: ALPHA must be passed in 8 bit fixed point format
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: QS8
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
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_vm_qs8(IMAGE_DECLARATION(src0),
                          IMAGE_DECLARATION(src1),
                          IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * 16;

    /* Compute the address for the vector A and matrix B */
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));
    src_addr.s1 += idx;

    int end_row_vec_a = src_addr.s0 + WIDTH_VECTOR_A;

    short8 acc0 = 0;
    short8 acc1 = 0;

    /* This for loop performs 4 accumulations per iteration */
    for(; src_addr.s0 <= (end_row_vec_a - 4); src_addr += (int2)(4, 4 * src1_stride_y))
    {
        char4  a0 = vload4(0, (__global char *)(src0_ptr + src_addr.s0));
        char16 b0 = vload16(0, (__global char *)(src1_ptr + src_addr.s1 + 0 * src1_stride_y));
        char16 b1 = vload16(0, (__global char *)(src1_ptr + src_addr.s1 + 1 * src1_stride_y));
        char16 b2 = vload16(0, (__global char *)(src1_ptr + src_addr.s1 + 2 * src1_stride_y));
        char16 b3 = vload16(0, (__global char *)(src1_ptr + src_addr.s1 + 3 * src1_stride_y));

        acc0 = mlal_sat_qs8x8(acc0, (char8)a0.s0, b0.s01234567, FIXED_POINT_POSITION);
        acc0 = mlal_sat_qs8x8(acc0, (char8)a0.s1, b1.s01234567, FIXED_POINT_POSITION);
        acc0 = mlal_sat_qs8x8(acc0, (char8)a0.s2, b2.s01234567, FIXED_POINT_POSITION);
        acc0 = mlal_sat_qs8x8(acc0, (char8)a0.s3, b3.s01234567, FIXED_POINT_POSITION);

        acc1 = mlal_sat_qs8x8(acc1, (char8)a0.s0, b0.s89ABCDEF, FIXED_POINT_POSITION);
        acc1 = mlal_sat_qs8x8(acc1, (char8)a0.s1, b1.s89ABCDEF, FIXED_POINT_POSITION);
        acc1 = mlal_sat_qs8x8(acc1, (char8)a0.s2, b2.s89ABCDEF, FIXED_POINT_POSITION);
        acc1 = mlal_sat_qs8x8(acc1, (char8)a0.s3, b3.s89ABCDEF, FIXED_POINT_POSITION);
    }

    /* Left-over accumulations */
    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(1, src1_stride_y))
    {
        char   a0 = *((__global char *)(src0_ptr + src_addr.s0));
        char16 b0 = vload16(0, (__global char *)(src1_ptr + src_addr.s1));

        acc0 = mlal_sat_qs8x8(acc0, (char8)a0, b0.s01234567, FIXED_POINT_POSITION);
        acc1 = mlal_sat_qs8x8(acc1, (char8)a0, b0.s89ABCDEF, FIXED_POINT_POSITION);
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Multiply by the weight of matrix product */
    char16 acc_qs8 = convert_char16_sat((short16)(acc0, acc1));

    acc_qs8 = mul_sat_qs8x16(acc_qs8, (char16)ALPHA, FIXED_POINT_POSITION);

    /* Store 16 values */
    vstore16(acc_qs8, 0, (__global char *)(offset(&dst, 0, 0)));
}
#endif /* FIXED_POINT_POSITION */
#endif /* WIDTH_VECTOR_A */
#endif /* WIDTH_MATRIX_B && ALPHA */

#ifdef BETA
/** This OpenCL kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @attention The beta's value need to be passed at compile time using -DBETA
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_f32(IMAGE_DECLARATION(src),
                          IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from A x B */
    float4 alpha_ab = vload4(0, (__global float *)dst.ptr);

    /* Load values from Matrix C */
    float4 c = vload4(0, (__global float *)src.ptr);

    /* Computes alpha * axb + beta * c */
    float4 out = alpha_ab + (float4)BETA * c;

    /* Store final result in axb matrix */
    vstore4(out, 0, (__global float *)dst.ptr);
}

/** This OpenCL kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @attention The beta's value need to be passed at compile time using -DBETA
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_f16(IMAGE_DECLARATION(src),
                          IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from A x B */
    half8 alpha_ab = vload8(0, (__global half *)dst.ptr);

    /* Load values from Matrix C */
    half8 c = vload8(0, (__global half *)src.ptr);

    /* Computes alpha * axb + beta * c */
    half8 out = alpha_ab + (half8)BETA * c;

    /* Store final result in axb matrix */
    vstore8(out, 0, (__global half *)dst.ptr);
}

#ifdef FIXED_POINT_POSITION
/** This OpenCL kernel performs the in-place matrix addition between 2 matrices in 8 bit fixed point taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @attention The beta's value and the fixed point position need to be passed at compile time using -DBETA and -DFIXED_POINT_POSITION
 *
 * @note: BETA must be passed in 8 bit fixed point format
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: QS8
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_qs8(IMAGE_DECLARATION(src),
                          IMAGE_DECLARATION(dst))
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from A x B */
    char16 alpha_ab = vload16(0, (__global char *)dst.ptr);

    /* Load values from Matrix C */
    char16 c = vload16(0, (__global char *)src.ptr);

    /* Computes alpha * axb + beta * c */
    char16 out = mla_sat_qs8x16(alpha_ab, (char16)BETA, c, FIXED_POINT_POSITION);

    /* Store final result in axb matrix */
    vstore16(out, 0, (__global char *)dst.ptr);
}
#endif /* FIXED_POINT_POSITION */
#endif /* BETA */

#ifdef WIDTH_VECTOR_A
/** This OpenCL kernel computes the vector by matrix multiplication between each row of A (src0) and matrix B (src1) used for locally connected layer
 *
 * @attention The width of A need to be passed at compile time using -DWIDTH_VECTOR_A
 *
 * @attention The input A and matrix B must not be reshaped
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
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_lc_vm_f32(IMAGE_DECLARATION(src0),
                             TENSOR3D_DECLARATION(src1),
                             IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * 4;
    int idy = get_global_id(1);

    /* Compute the address for the vector A and matrix B */
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes + src0_stride_y * idy, src1_offset_first_element_in_bytes + src1_stride_z * idy));
    src_addr.s1 += idx * sizeof(float);

    int end_row_vec_a = src_addr.s0 + (WIDTH_VECTOR_A * sizeof(float));

    float4 acc = 0.0f;

    for(; src_addr.s0 <= (end_row_vec_a - 2 * sizeof(float)); src_addr += (int2)(2 * sizeof(float), 2 * src1_stride_y))
    {
        float2 a0 = vload2(0, (__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        float4 b1 = vload4(0, (__global float *)(src1_ptr + src_addr.s1 + src1_stride_y));

        acc += b0 * (float4)a0.s0;
        acc += b1 * (float4)a0.s1;
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(float), src1_stride_y))
    {
        float  a0 = *((__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));

        acc += b0 * (float4)a0;
    }

    /* Compute destination address */
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    vstore4(acc, 0, (__global float *)(offset(&dst, 0, 0)));
}
#endif /* WIDTH_VECTOR_A */
