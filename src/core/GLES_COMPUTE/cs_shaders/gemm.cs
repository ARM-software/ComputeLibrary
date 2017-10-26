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
layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;
#include "helpers.h"

#if defined(DATA_TYPE_FP32)
#define LOAD8(r, name, offset) \
    r.x = LOAD4(name, offset); \
    r.y = LOAD4(name, offset + uint(1))

#define LOAD16(r, name, offset)          \
    r.x = LOAD4(name, offset);           \
    r.y = LOAD4(name, offset + uint(1)); \
    r.z = LOAD4(name, offset + uint(2)); \
    r.w = LOAD4(name, offset + uint(3))

#define STORE16(name, offset, r)         \
    STORE4(name, offset, r.x);           \
    STORE4(name, offset + uint(1), r.y); \
    STORE4(name, offset + uint(2), r.z); \
    STORE4(name, offset + uint(3), r.w)

#ifdef GEMM_TRANSPOSE1xW
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel computes the "vector" 1x4 transposition of input matrix
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
void main(void)
{
    /* Compute address for Matrix B - source */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    uint dst_addr_in_bytes = (gl_GlobalInvocationID.y * uint(16) + gl_GlobalInvocationID.x * dst.stride_y + dst.offset_first_element_in_bytes) >> 2;
    vec4 b0;
    LOAD16(b0, src, offset(src, 0, 0));
    STORE16(dst, dst_addr_in_bytes, b0);
}
#endif /* GEMM_TRANSPOSE1xW */

#ifdef GEMM_INTERLEAVE4x4
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGLES kernel reshapes the input matrix interleaving the values
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
void main(void)
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    int i;
    int j;

    for(i = 0; i < 4; ++i)
    {
        for(j = 0; j < 4; ++j)
        {
            float res    = LOAD4(src, offset(src, i, j));
            uint  ofset0 = CURRENT_OFFSET(dst) + uint(i * 4 + j);
            STORE4(dst, ofset0, res);
        }
    }
}
#endif /* GEMM_INTERLEAVE4x4 */

#ifdef GEMM_ACCUMULATE_BIASES
BUFFER_DECLARATION(accum, 1, float, restrict);
BUFFER_DECLARATION(biases, 2, float, readonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(accum);
    VECTOR_PARAM_DECLARATION(biases);
};

/** This kernel accumulates each row with the biases vector
 *
 * @param[in, out] accum_ptr                            Pointer to the accumulate tensor. Supported data type: F32
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
void main(void)
{
    Image  accum  = CONVERT_TO_IMAGE_STRUCT(accum);
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    for(int i = 0; i < 16; ++i)
    {
        float accum_value  = LOAD4(accum, CURRENT_OFFSET(accum) + uint(i));
        float biases_value = LOAD4(biases, CURRENT_OFFSET(biases) + uint(i));
        accum_value        = biases_value + accum_value;

        // Store result in the accummulate buffer
        STORE4(accum, CURRENT_OFFSET(accum) + uint(i), accum_value);
    }
}
#endif /* GEMM_ACCUMULATE_BIASES */

#ifdef GEMM_MM_INTERLEAVED_TRANSPOSED /* unvalidate */
BUFFER_DECLARATION(src0, 1, float, readonly);
BUFFER_DECLARATION(src1, 2, float, readonly);
BUFFER_DECLARATION(dst, 3, float, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src0);
    IMAGE_PARAM_DECLARATION(src1);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel is optimised for Midgard. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
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
void main()
{
    Image src0 = CONVERT_TO_IMAGE_STRUCT(src0);
    Image src1 = CONVERT_TO_IMAGE_STRUCT(src1);
    Image dst  = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Compute address for matrix A and B */
    src0.current_offset = (src0.offset_first_element_in_bytes + (uint(gl_GlobalInvocationID.y) * uint(src0.stride_y))) >> uint(2);
    src1.current_offset = (src1.offset_first_element_in_bytes + (uint(gl_GlobalInvocationID.x) * uint(src1.stride_y))) >> uint(2);

    /* Compute end row address for matrix B */
    int end_row_mtx_b = int(src1.current_offset) + int(COLS_B);

    /* Reset accumulators */
    vec4 c00 = vec4(0.0f);
    vec4 c10 = vec4(0.0f);
    vec4 c20 = vec4(0.0f);
    vec4 c30 = vec4(0.0f);

    // FIXME: loop unrolling really needed for GLES?
    for(; int(src1.current_offset) <= (end_row_mtx_b - 8); src0.current_offset += uint(8), src1.current_offset += uint(8))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0;
        vec4 b0;
        LOAD16(a0, src0, src0.current_offset);
        LOAD16(b0, src1, src1.current_offset);

        c00 += vec4(a0.x) * b0;
        c10 += vec4(a0.y) * b0;
        c20 += vec4(a0.z) * b0;
        c30 += vec4(a0.w) * b0;

        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        LOAD16(a0, src0, src0.current_offset + uint(4));
        LOAD16(b0, src1, src1.current_offset + uint(4));

        c00 += vec4(a0.x) * b0;
        c10 += vec4(a0.y) * b0;
        c20 += vec4(a0.z) * b0;
        c30 += vec4(a0.w) * b0;
    }

    for(; int(src1.current_offset) < end_row_mtx_b; src0.current_offset += uint(4), src1.current_offset += uint(4))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0;
        vec4 b0;
        LOAD16(a0, src0, src0.current_offset);
        LOAD16(b0, src1, src1.current_offset);

        c00 += vec4(a0.x) * b0;
        c10 += vec4(a0.y) * b0;
        c20 += vec4(a0.z) * b0;
        c30 += vec4(a0.w) * b0;
    }

    /* Multiply by the weight of matrix product */
    c00 = c00 * vec4(ALPHA);
    c10 = c10 * vec4(ALPHA);
    c20 = c20 * vec4(ALPHA);
    c30 = c30 * vec4(ALPHA);

    /* Store 4x4 block */
    STORE16(dst, offset(dst, 0, 0), c00);
    STORE16(dst, offset(dst, 0, 1), c10);
    STORE16(dst, offset(dst, 0, 2), c20);
    STORE16(dst, offset(dst, 0, 3), c30);
}
#endif /* GEMM_MM_INTERLEAVED_TRANSPOSED */

#ifdef GEMM_MM_FLOATING_POINT
BUFFER_DECLARATION(src0, 1, float, readonly);
BUFFER_DECLARATION(src1, 2, float, readonly);
BUFFER_DECLARATION(dst, 3, float, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src0);
    IMAGE_PARAM_DECLARATION(src1);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
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
void main()
{
    Image src0 = CONVERT_TO_IMAGE_STRUCT(src0);
    Image src1 = CONVERT_TO_IMAGE_STRUCT(src1);
    Image dst  = CONVERT_TO_IMAGE_STRUCT(dst);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    src0.current_offset = (src0_offset_first_element_in_bytes + uint(gl_GlobalInvocationID.y) * src0_stride_y * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y)) >> uint(2);
    src1.current_offset = (src1_offset_first_element_in_bytes + uint(idx * 4)) >> uint(2);

    /* Compute end row address for matrix A */
    int end_row_vec_a = int(src0.current_offset) + ((COLS_A * 4) >> 2);

    /* Reset accumulators */
    vec4 acc0 = vec4(0.0f);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vec4 acc1 = vec4(0.0f);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vec4 acc2 = vec4(0.0f);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vec4 acc3 = vec4(0.0f);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    for(; int(src0.current_offset) <= (end_row_vec_a - 2); src0.current_offset += uint(2), src1.current_offset += uint((2 * int(src1_stride_y)) >> 2))
    {
        vec2 a0;
        LOAD8(a0, src0, src0.current_offset);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec2 a1;
        LOAD8(a1, src0, src0.current_offset + (src0_stride_y >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec2 a2;
        LOAD8(a2, src0, src0.current_offset + ((uint(2) * src0_stride_y) >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec2 a3;
        LOAD8(a3, src0, src0.current_offset + ((uint(3) * src0_stride_y) >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0;
        vec4 b1;
        LOAD16(b0, src1, src1.current_offset);
        LOAD16(b1, src1, src1.current_offset + (src1_stride_y >> uint(2)));

        acc0 += b0 * vec4(a0.x);
        acc0 += b1 * vec4(a0.y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += b0 * vec4(a1.x);
        acc1 += b1 * vec4(a1.y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += b0 * vec4(a2.x);
        acc2 += b1 * vec4(a2.y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += b0 * vec4(a3.x);
        acc3 += b1 * vec4(a3.y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    for(; int(src0.current_offset) < end_row_vec_a; src0.current_offset += uint(1), src1.current_offset += uint(int(src1_stride_y) >> 2))
    {
        // Load values from matrix A
        float a0;
        a0 = LOAD4(src0, src0.current_offset);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1;
        a1 = LOAD4(src0, src0.current_offset + ((uint(1) * src0_stride_y) >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2;
        a2 = LOAD4(src0, src0.current_offset + ((uint(2) * src0_stride_y) >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3;
        a3 = LOAD4(src0, src0.current_offset + ((uint(3) * src0_stride_y) >> uint(2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0;
        LOAD16(b0, src1, src1.current_offset);

        acc0 += b0 * vec4(a0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += b0 * vec4(a1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += b0 * vec4(a2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += b0 * vec4(a3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    /* Multiply by the weight of vector-matrix product */
    acc0 = acc0 * vec4(ALPHA);
    STORE16(dst, offset(dst, 0, 0), acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 = acc1 * vec4(ALPHA);
    STORE16(dst, offset(dst, 0, 1), acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 = acc2 * vec4(ALPHA);
    STORE16(dst, offset(dst, 0, 2), acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 = acc3 * vec4(ALPHA);
    STORE16(dst, offset(dst, 0, 3), acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
}
#endif /* GEMM_MM_FLOATING_POINT */

#ifdef GEMM_MATRIXADDITION
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, restrict);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @attention The beta's value need to be passed at compile time using BETA
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
void main(void)
{
    /* Compute source and destination addresses */
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load values from A x B */
    vec4 alpha_ab;
    vec4 c;
    vec4 out1;

    LOAD16(alpha_ab, dst, dst.current_offset);
    LOAD16(c, src, src.current_offset);

    /* Computes alpha * axb + beta * c */
    out1 = alpha_ab + vec4(BETA * c);

    /* Store final result in axb matrix */
    STORE16(dst, dst.current_offset, out1);
}
#endif /* GEMM_MATRIXADDITION */
#elif defined(DATA_TYPE_FP16)
precision mediump float;
#ifdef GEMM_MM_FLOATING_POINT
BUFFER_DECLARATION(src0, 1, uint, readonly);
BUFFER_DECLARATION(src1, 2, uvec2, readonly);
BUFFER_DECLARATION(dst, 3, uvec2, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src0);
    IMAGE_PARAM_DECLARATION(src1);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
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
void main()
{
    Image src0 = GC_CONVERT_TO_IMAGE_STRUCT(src0);
    Image src1 = GC_CONVERT_TO_IMAGE_STRUCT(src1);
    Image dst  = GC_CONVERT_TO_IMAGE_STRUCT(dst);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    src0.current_offset = (src0_offset_first_element_in_bytes + uint(gl_GlobalInvocationID.y) * src0_stride_y * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y));
    src1.current_offset = src1_offset_first_element_in_bytes + uint(idx) * src1_stride_x;

    /* Compute end row address for matrix A */
    uint end_row_vec_a = src0.current_offset + uint(COLS_A << 1);

    /* Reset accumulators */
    vec4 acc0 = vec4(0.0f);

    for(; src0.current_offset < (end_row_vec_a - uint(2)); src0.current_offset += uint(2 * 2), src1.current_offset += uint(2) * src1_stride_y)
    {
        uint packed_a0;
        vec2 a0;

        GC_LOAD1_2D_OFFSET(packed_a0, src0, 0, 0);
        a0 = vec2(unpackHalf2x16(packed_a0));

        uvec2 packed_b0;
        uvec2 packed_b1;
        vec4  b0;
        vec4  b1;

        GC_LOAD1_2D_OFFSET(packed_b0, src1, 0, 0);
        GC_LOAD1_2D_OFFSET(packed_b1, src1, 0, 1);

        b0 = vec4(unpackHalf2x16(packed_b0.x), unpackHalf2x16(packed_b0.y));
        b1 = vec4(unpackHalf2x16(packed_b1.x), unpackHalf2x16(packed_b1.y));

        acc0 += b0 * vec4(a0.x);
        acc0 += b1 * vec4(a0.y);
    }

    for(; src0.current_offset < end_row_vec_a; src0.current_offset += uint(2 * 2), src1.current_offset += src1_stride_y)
    {
        uint packed_a0;
        vec2 a0;

        GC_LOAD1_2D_OFFSET(packed_a0, src0, 0, 0);
        a0 = vec2(unpackHalf2x16(packed_a0));

        uvec2 packed_b0;
        vec4  b0;

        GC_LOAD1_2D_OFFSET(packed_b0, src1, 0, 0);

        b0 = vec4(unpackHalf2x16(packed_b0.x), unpackHalf2x16(packed_b0.y));

        acc0 += b0 * (a0.x);
    }

    /* Multiply by the weight of vector-matrix product */
    acc0 = acc0 * vec4(ALPHA);

    uvec2 packed_d;
    packed_d = uvec2(packHalf2x16(acc0.xy), packHalf2x16(acc0.zw));
    GC_STORE1_2D_OFFSET(packed_d, dst, 0, 0);
}
#endif /* GEMM_MM_FLOATING_POINT */

#ifdef GEMM_ACCUMULATE_BIASES
BUFFER_DECLARATION(accum, 1, uvec2, restrict);
BUFFER_DECLARATION(biases, 2, uvec2, readonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(accum);
    VECTOR_PARAM_DECLARATION(biases);
};

/** This kernel accumulates each row with the biases vector
 *
 * @param[in, out] accum_ptr                            Pointer to the accumulate tensor. Supported data type: F16
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
void main(void)
{
    Image  accum  = GC_CONVERT_TO_IMAGE_STRUCT(accum);
    Vector biases = GC_CONVERT_TO_VECTOR_STRUCT(biases);

    vec4  u[2];
    uvec2 packed_s[2];
    GC_LOAD1_2D_OFFSET(packed_s[0], accum, 0, 0);
    GC_LOAD1_1D_OFFSET(packed_s[1], biases, 0);
    u[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    u[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    vec4 tmp;
    tmp         = u[0] + u[1];
    packed_s[0] = uvec2(packHalf2x16(tmp.xy), packHalf2x16(tmp.zw));
    GC_STORE1_2D_OFFSET(packed_s[0], accum, 0, 0);
}
#endif /* GEMM_ACCUMULATE_BIASES */
#else  /* DATA_TYPE_F32 */
#error Data type not supported
#endif /* DATA_TYPE_F32 */
