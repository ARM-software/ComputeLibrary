/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "helpers_cs.h"

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

#if defined(DATA_TYPE_FP32)
#ifdef GEMM_TRANSPOSE1xW
/** This OpenGL ES kernel computes the "vector" 1x4 transposition of input matrix
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    /* Compute address for Matrix B - source */
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, gl_GlobalInvocationID.y * uint(16) + gl_GlobalInvocationID.x * dst_attrs.stride_y);

    vec4 b0 = VLOAD4_CURRENT_ITEM(vec4, src_ptr, src_iter);
    VSTORE4_CURRENT_ITEM(dst_ptr, dst_iter, b0);
}
#endif /* GEMM_TRANSPOSE1xW */

#ifdef GEMM_INTERLEAVE4x4
/** This OpenGLES kernel reshapes the input matrix interleaving the values
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    /* Compute source and destination addresses */
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    int i;
    int j;

    for(i = 0; i < 4; ++i)
    {
        for(j = 0; j < 4; ++j)
        {
            float res = LOAD(src_ptr, IMAGE_OFFSET(src_iter, i, j));
            STORE(dst_ptr, TENSOR_OFFSET_ADVANCE(dst_iter, (i * 4 + j)), res);
        }
    }
}
#endif /* GEMM_INTERLEAVE4x4 */

#ifdef GEMM_ACCUMULATE_BIASES
/** This kernel accumulates each row with the biases vector
 *
 * @param[in, out] accum_ptr    Pointer to the accumulate tensor. Supported data type: F32
 * @param[in]      accum_attrs  The attributes of the accumulate tensor
 * @param[in]      biases_ptr   Pointer to the biases vector. Same as @p accum_ptr
 * @param[in]      biases_attrs The attributes of the biases tensor
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes  accum_attrs;
    VectorAttributes biases_attrs;
};
TENSOR_DECLARATION(1, accumBuffer, float, accum_ptr, accum_shift, 2, restrict);
TENSOR_DECLARATION(2, biasesBuffer, float, biases_ptr, biases_shift, 2, readonly);

void main(void)
{
    ImageIterator  accum_iter  = CONVERT_TO_IMAGE_ITERATOR(accum_attrs, accum_shift);
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR(biases_attrs, biases_shift);

    for(int i = 0; i < 16; ++i)
    {
        float accum_value  = LOAD(accum_ptr, TENSOR_OFFSET_ADVANCE(accum_iter, i));
        float biases_value = LOAD(biases_ptr, TENSOR_OFFSET_ADVANCE(biases_iter, i));
        accum_value        = biases_value + accum_value;

        // Store result in the accummulate buffer
        STORE(accum_ptr, TENSOR_OFFSET_ADVANCE(accum_iter, i), accum_value);
    }
}
#endif /* GEMM_ACCUMULATE_BIASES */

#ifdef GEMM_MM_INTERLEAVED_TRANSPOSED /* unvalidate */
/** This OpenGL ES kernel is optimised for Midgard. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
 *
 * @param[in]  src0_ptr   Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_attrs The attributes of the source matrix
 * @param[in]  src1_ptr   Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_attrs The attributes of the source matrix
 * @param[out] dst_ptr    Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_attrs  The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src0_attrs;
    ImageAttributes src1_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, src0Buffer, float, src0_ptr, src0_shift, 2, readonly);
TENSOR_DECLARATION(2, src1Buffer, float, src1_ptr, src1_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    /* Compute address for matrix A and B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * (src0_attrs.stride_y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(gl_GlobalInvocationID.x) * (src1_attrs.stride_y));
    /* Compute end row address for matrix B */
    int end_row_mtx_b = int(TENSOR_OFFSET_ADVANCE(src1_iter, COLS_B));

    /* Reset accumulators */
    vec4 c00 = vec4(0.0f);
    vec4 c10 = vec4(0.0f);
    vec4 c20 = vec4(0.0f);
    vec4 c30 = vec4(0.0f);

    for(; int(CURRENT_ITEM_OFFSET(src1_iter)) <= (end_row_mtx_b - 8); TENSOR_ITERATOR_ADVANCE(src0_iter, 8), TENSOR_ITERATOR_ADVANCE(src1_iter, 8))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0 = VLOAD4_CURRENT_ITEM(vec4, src0_ptr, src0_iter);
        vec4 b0 = VLOAD4_CURRENT_ITEM(vec4, src1_ptr, src1_iter);

        c00 += vec4(a0.x) * b0;
        c10 += vec4(a0.y) * b0;
        c20 += vec4(a0.z) * b0;
        c30 += vec4(a0.w) * b0;

        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        a0 = VLOAD4(vec4, src0_ptr, TENSOR_OFFSET_ADVANCE(src0_iter, 4));
        b0 = VLOAD4(vec4, src1_ptr, TENSOR_OFFSET_ADVANCE(src1_iter, 4));

        c00 += vec4(a0.x) * b0;
        c10 += vec4(a0.y) * b0;
        c20 += vec4(a0.z) * b0;
        c30 += vec4(a0.w) * b0;
    }

    for(; int(CURRENT_ITEM_OFFSET(src1_iter)) < end_row_mtx_b; TENSOR_ITERATOR_ADVANCE(src0_iter, 4), TENSOR_ITERATOR_ADVANCE(src1_iter, 4))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0 = VLOAD4_CURRENT_ITEM(vec4, src0_ptr, src0_iter);
        vec4 b0 = VLOAD4_CURRENT_ITEM(vec4, src1_ptr, src1_iter);

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
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 0), c00);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), c10);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), c20);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), c30);
}
#endif /* GEMM_MM_INTERLEAVED_TRANSPOSED */

#ifdef GEMM_MM_FLOATING_POINT
/** This OpenGL ES kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
 *
 * @param[in]  src0_ptr   Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_attrs The attributes of the source matrix
 * @param[in]  src1_ptr   Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_attrs The attributes of the source matrix
 * @param[out] dst_ptr    Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_attrs  The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src0_attrs;
    ImageAttributes src1_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, src0Buffer, float, src0_ptr, src0_shift, 2, readonly);
TENSOR_DECLARATION(2, src1Buffer, float, src1_ptr, src1_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * (src0_attrs.stride_y) * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, idx * 4);

    /* Compute end row address for matrix A */
    int end_row_vec_a = int(TENSOR_OFFSET_ADVANCE_IN_BYTES(src0_iter, COLS_A * 4));

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

    for(; int(CURRENT_ITEM_OFFSET(src0_iter)) <= (end_row_vec_a - 2); TENSOR_ITERATOR_ADVANCE(src0_iter, 2), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(2) * src1_attrs.stride_y))
    {
        vec2 a0 = VLOAD2_CURRENT_ITEM(vec2, src0_ptr, src0_iter);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec2 a1 = VLOAD2(vec2, src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec2 a2 = VLOAD2(vec2, src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec2 a3 = VLOAD2(vec2, src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0 = VLOAD4_CURRENT_ITEM(vec4, src1_ptr, src1_iter);
        vec4 b1 = VLOAD4(vec4, src1_ptr, IMAGE_OFFSET(src1_iter, 0, 1));

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

    for(; int(CURRENT_ITEM_OFFSET(src0_iter)) < end_row_vec_a; TENSOR_ITERATOR_ADVANCE(src0_iter, 1), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, src1_attrs.stride_y))
    {
        // Load values from matrix A
        float a0 = LOAD_CURRENT_ITEM(src0_ptr, src0_iter);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1 = LOAD(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
        //float a1 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2 = LOAD(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3 = LOAD(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0 = VLOAD4_CURRENT_ITEM(vec4, src1_ptr, src1_iter);

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
    VSTORE4_CURRENT_ITEM(dst_ptr, dst_iter, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 = acc1 * vec4(ALPHA);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 = acc2 * vec4(ALPHA);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 = acc3 * vec4(ALPHA);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
}
#endif /* GEMM_MM_FLOATING_POINT */

#ifdef GEMM_MATRIXADDITION
/** This OpenGL ES kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @attention The beta's value need to be passed at compile time using BETA
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, restrict);

void main(void)
{
    /* Compute source and destination addresses */
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    /* Load values from A x B */
    vec4 alpha_ab = VLOAD4_CURRENT_ITEM(vec4, dst_ptr, dst_iter);
    vec4 c        = VLOAD4_CURRENT_ITEM(vec4, src_ptr, src_iter);

    /* Computes alpha * axb + beta * c */
    vec4 out1 = alpha_ab + vec4(float(BETA) * c);

    /* Store final result in axb matrix */
    VSTORE4_CURRENT_ITEM(dst_ptr, dst_iter, out1);
}
#endif /* GEMM_MATRIXADDITION */

#elif defined(DATA_TYPE_FP16)

#ifdef GEMM_TRANSPOSE1xW
/** This OpenGL ES kernel computes the "vector" 1x8 transposition of input matrix
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F16
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main(void)
{
    /* Compute address for Matrix B - source */
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    /* Compute address for Matrix B transposed - destination. X and Y are swapped */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, gl_GlobalInvocationID.y * uint(16) + gl_GlobalInvocationID.x * dst_attrs.stride_y);

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, LOAD_CURRENT_ITEM(src_ptr, src_iter));
}
#endif /* GEMM_TRANSPOSE1xW */

#ifdef GEMM_INTERLEAVE4x4
/** This OpenGLES kernel reshapes the input matrix interleaving the values
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F16
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main(void)
{
    /* Compute source and destination addresses */
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    vec4 s0[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);
    vec4 s1[2] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
    vec4 s2[2] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
    vec4 s3[2] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 3));

    vec4 s[2];
    s[0] = vec4(s0[0].x, s1[0].x, s2[0].x, s3[0].x);
    s[1] = vec4(s0[0].y, s1[0].y, s2[0].y, s3[0].y);
    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, s);

    s[0] = vec4(s0[0].z, s1[0].z, s2[0].z, s3[0].z);
    s[1] = vec4(s0[0].w, s1[0].w, s2[0].w, s3[0].w);
    STORE_PACK8_HALF(dst_ptr, TENSOR_OFFSET_ADVANCE(dst_iter, 1u), s);

    s[0] = vec4(s0[1].x, s1[1].x, s2[1].x, s3[1].x);
    s[1] = vec4(s0[1].y, s1[1].y, s2[1].y, s3[1].y);
    STORE_PACK8_HALF(dst_ptr, TENSOR_OFFSET_ADVANCE(dst_iter, 2u), s);

    s[0] = vec4(s0[1].z, s1[1].z, s2[1].z, s3[1].z);
    s[1] = vec4(s0[1].w, s1[1].w, s2[1].w, s3[1].w);
    STORE_PACK8_HALF(dst_ptr, TENSOR_OFFSET_ADVANCE(dst_iter, 3u), s);
}
#endif /* GEMM_INTERLEAVE4x4 */

#ifdef GEMM_MM_FLOATING_POINT
/** This OpenGL ES kernel computes the matrix multiplication between matrix A(src0) and matrix B(src1)
 * Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_16bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
 *
 * @param[in]  src0_ptr   Pointer to the source matrix.Supported data types: F16
 * @param[in]  src0_attrs The attributes of the source matrix
 * @param[in]  src1_ptr   Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_attrs The attributes of the source matrix
 * @param[out] dst_ptr    Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_attrs  The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src0_attrs;
    ImageAttributes src1_attrs;
    ImageAttributes dst_attrs;
};

#if defined(MM_PROCESS_4X)
TENSOR_DECLARATION(1, src0Buffer, uint, src0_ptr, src0_shift, 2, readonly);
TENSOR_DECLARATION(2, src1Buffer, uvec2, src1_ptr, src1_shift, 3, readonly);
TENSOR_DECLARATION(3, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * src0_attrs.stride_y * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(idx) * src1_attrs.stride_x);

    /* Compute end row address for matrix A */
    uint end_row_vec_a = uint(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) + uint(COLS_A << 1);

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

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a - uint(2));
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, 2 * 2), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(2) * src1_attrs.stride_y))
    {
        vec2 a0 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec2 a1 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec2 a2 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec2 a3 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0 = LOAD_UNPACK4_CURRENT_ITEM_HALF(src1_ptr, src1_iter);
        vec4 b1 = LOAD_UNPACK4_HALF(src1_ptr, IMAGE_OFFSET(src1_iter, 0, 1));

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

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a); TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, 2 * 2), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, src1_attrs.stride_y))
    {
        vec2 a0 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec2 a1 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec  a2 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec2 a3 = LOAD_UNPACK2_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b0 = LOAD_UNPACK4_CURRENT_ITEM_HALF(src1_ptr, src1_iter);

        acc0 += b0 * (a0.x);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += b0 * (a1.x);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += b0 * (a2.x);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += b0 * (a3.x);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    /* Multiply by the weight of vector-matrix product */
    acc0 = acc0 * vec4(ALPHA);

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), acc3);
#endif                                 // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
}
#elif defined(MM_PROCESS_4X_OPTIMIZED) /* PROCESS_4X */
TENSOR_DECLARATION(1, src0Buffer, uvec4, src0_ptr, src0_shift, 4, readonly);
TENSOR_DECLARATION(2, src1Buffer, uvec2, src1_ptr, src1_shift, 3, readonly);
TENSOR_DECLARATION(3, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * src0_attrs.stride_y * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(idx) * src1_attrs.stride_x);

    /* Compute end row address for matrix A */
    uint end_row_vec_a = uint(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) + uint(COLS_A << 1);

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

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a - uint(16));
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(8) * src0_attrs.stride_x), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(8) * src1_attrs.stride_y))
    {
        vec4 a0[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src0_ptr, src0_iter);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec4 a1[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec4 a2[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec4 a3[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b;

        for(int i = 0; i < 8; i++)
        {
            int j = i >> 2;
            int k = i % 4;

            b = LOAD_UNPACK4_HALF(src1_ptr, IMAGE_OFFSET(src1_iter, 0, i));

            acc0 += b * vec4(a0[j][k]);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
            acc1 += b * vec4(a1[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
            acc2 += b * vec4(a2[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
            acc3 += b * vec4(a3[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        }
    }

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a); TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, 2 * 8), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(8) * src1_attrs.stride_y))
    {
        vec4 a0[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src0_ptr, src0_iter);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        vec4 a1[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        vec4 a2[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        vec4 a3[2] = LOAD_UNPACK8_HALF(src0_ptr, IMAGE_OFFSET(src0_iter, 0, 3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        vec4 b;

        int leftover = COLS_A % 8;

        for(int i = 0; i < leftover; i++)
        {
            int j = i >> 2;
            int k = i % 4;

            b = LOAD_UNPACK4_HALF(src1_ptr, IMAGE_OFFSET(src1_iter, 0, i));

            acc0 += b * vec4(a0[j][k]);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
            acc1 += b * vec4(a1[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
            acc2 += b * vec4(a2[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
            acc3 += b * vec4(a3[j][k]);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        }
    }

    /* Multiply by the weight of vector-matrix product */
    acc0 = acc0 * vec4(ALPHA);

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), acc3);
#endif                       // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
}
#elif defined(MM_PROCESS_8X) /* PROCESS_8X */
TENSOR_DECLARATION(1, src0Buffer, uvec4, src0_ptr, src0_shift, 4, readonly);
TENSOR_DECLARATION(2, src1Buffer, uvec4, src1_ptr, src1_shift, 4, readonly);
TENSOR_DECLARATION(3, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    int idx = int(gl_GlobalInvocationID.x) * int(NUM_ELEMS_PROCESSED_PER_THREAD_X);
    /* Compute the address for the vector A and matrix B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * src0_attrs.stride_y * uint(NUM_ELEMS_PROCESSED_PER_THREAD_Y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(idx) * src1_attrs.stride_x);

    /* Compute end row address for matrix A */
    uint end_row_vec_a = uint(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) + uint(COLS_A << 1);

    /* Reset accumulators */
    vec4 acc[2];

    acc[0] = vec4(0.0f);
    acc[1] = vec4(0.0f);

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a - uint(16));
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(8) * src0_attrs.stride_x), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(8) * src1_attrs.stride_y))
    {
        vec4 a[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
        vec4 b[2];

        for(int i = 0; i < 8; i++)
        {
            int j = i >> 2;
            int k = i % 4;

            b = LOAD_UNPACK8_HALF(src1_ptr, IMAGE_OFFSET(src1_iter, 0, i));

            acc[0] += b[0] * vec4(a[j][k]);
            acc[1] += b[1] * vec4(a[j][k]);
        }
    }

    for(; int(CURRENT_ITEM_OFFSET_IN_BYTES(src0_iter)) < int(end_row_vec_a);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(8) * uint(2)), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(8) * src1_attrs.stride_y))
    {
        vec4 a[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
        vec4 b[2];

        int leftover = COLS_A % 8;

        for(int i = 0; i < leftover; i++)
        {
            int j = i >> 2;
            int k = i % 4;

            b = LOAD_UNPACK8_HALF(src1_ptr, IMAGE_OFFSET(src1_iter, 0, i));

            acc[0] += b[0] * vec4(a[j][k]);
            acc[1] += b[1] * vec4(a[j][k]);
        }
    }

    /* Multiply by the weight of vector-matrix product */
    acc[0] = acc[0] * vec4(ALPHA);
    acc[1] = acc[1] * vec4(ALPHA);

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, acc);
}
#endif                       /* PROCESS_8X */
#endif                       /* GEMM_MM_FLOATING_POINT */

#ifdef GEMM_ACCUMULATE_BIASES
#if defined(ACCUM_PROCESS_4X)
/** This kernel accumulates each row with the biases vector
 *
 * @param[in, out] accum_ptr    Pointer to the accumulate tensor. Supported data type: F16
 * @param[in]      accum_attrs  The attributes of the accumulate tensor
 * @param[in]      biases_ptr   Pointer to the biases vector. Same as @p accum_ptr
 * @param[in]      biases_attrs The attributes of the biases tensor
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes  accum_attrs;
    VectorAttributes biases_attrs;
};

TENSOR_DECLARATION(1, accumBuffer, uvec2, accum_ptr, accum_shift, 3, restrict);
TENSOR_DECLARATION(2, biasesBuffer, uvec2, biases_ptr, biases_shift, 3, readonly);

void main(void)
{
    ImageIterator  accum_iter  = CONVERT_TO_IMAGE_ITERATOR(accum_attrs, accum_shift);
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR(biases_attrs, biases_shift);

    vec4 u[2];
    u[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(accum_ptr, accum_iter);
    u[1] = LOAD_UNPACK4_CURRENT_ITEM_HALF(biases_ptr, biases_iter);

    vec4 tmp;
    tmp = u[0] + u[1];
    STORE_PACK4_CURRENT_ITEM_HALF(accum_ptr, accum_iter, tmp);
}
#elif defined(ACCUM_PROCESS_8X) /* ACCUM_PROCESS_8X */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes  accum_attrs;
    VectorAttributes biases_attrs;
};

TENSOR_DECLARATION(1, accumBuffer, uvec4, accum_ptr, accum_shift, 4, restrict);
TENSOR_DECLARATION(2, biasesBuffer, uvec4, biases_ptr, biases_shift, 4, readonly);

void main(void)
{
    ImageIterator  accum_iter  = CONVERT_TO_IMAGE_ITERATOR(accum_attrs, accum_shift);
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR(biases_attrs, biases_shift);

    vec4 u[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(accum_ptr, accum_iter);
    vec4 v[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(biases_ptr, bias_iter);

    vec4 r[2];
    r[0] = u[0] + v[0];
    r[1] = u[1] + v[1];
    STORE_PACK8_CURRENT_ITEM_HALF(accum_ptr, accum_iter, r);
}
#endif                          /* ACCUM_PROCESS_8X */
#endif                          /* GEMM_ACCUMULATE_BIASES */

#ifdef GEMM_MM_INTERLEAVED_TRANSPOSED
/** This OpenGL ES kernel is optimised for Midgard. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * @attention The width of matrix B and the alpha's value need to be passed at compile time using WIDTH_MATRIX_B and ALPHA
 *
 * @param[in]  src0_ptr   Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_attrs The attributes of the source matrix
 * @param[in]  src1_ptr   Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_attrs The attributes of the source matrix
 * @param[out] dst_ptr    Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_attrs  The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src0_attrs;
    ImageAttributes src1_attrs;
    ImageAttributes dst_attrs;
};
TENSOR_DECLARATION(1, src0Buffer, uvec2, src0_ptr, src0_shift, 3, readonly);
TENSOR_DECLARATION(2, src1Buffer, uvec4, src1_ptr, src1_shift, 4, readonly);
TENSOR_DECLARATION(3, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main()
{
    ImageIterator src0_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src0_attrs, src0_shift);
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(src1_attrs, src1_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    /* Compute address for matrix A and B */
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, uint(gl_GlobalInvocationID.y) * (src0_attrs.stride_y));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, uint(gl_GlobalInvocationID.x) * (src1_attrs.stride_y));
    /* Compute end row address for matrix B */
    int end_row_mtx_b = (int(CURRENT_ITEM_OFFSET_IN_BYTES(src1_iter)) >> 1) + int(COLS_B);

    /* Reset accumulators */
    vec4 c00[2];
    vec4 c10[2];
    vec4 c20[2];
    vec4 c30[2];
    c00[0] = vec4(0.0f);
    c00[1] = vec4(0.0f);
    c10[0] = vec4(0.0f);
    c10[1] = vec4(0.0f);
    c20[0] = vec4(0.0f);
    c20[1] = vec4(0.0f);
    c30[0] = vec4(0.0f);
    c30[1] = vec4(0.0f);

    // FIXME: loop unrolling really needed for GLES?
    for(; (int(CURRENT_ITEM_OFFSET_IN_BYTES(src1_iter)) >> 1) <= (end_row_mtx_b - 16); TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, 16), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, 32))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0    = LOAD_UNPACK4_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
        vec4 b0[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src1_ptr, src1_iter);

        c00[0] += vec4(a0.x) * b0[0];
        c00[1] += vec4(a0.x) * b0[1];
        c10[0] += vec4(a0.y) * b0[0];
        c10[1] += vec4(a0.y) * b0[1];
        c20[0] += vec4(a0.z) * b0[0];
        c20[1] += vec4(a0.z) * b0[1];
        c30[0] += vec4(a0.w) * b0[0];
        c30[1] += vec4(a0.w) * b0[1];

        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        a0 = LOAD_UNPACK4_HALF(src0_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src0_iter, 8));
        b0 = LOAD_UNPACK8_HALF(src1_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src1_iter, 16));

        c00[0] += vec4(a0.x) * b0[0];
        c00[1] += vec4(a0.x) * b0[1];
        c10[0] += vec4(a0.y) * b0[0];
        c10[1] += vec4(a0.y) * b0[1];
        c20[0] += vec4(a0.z) * b0[0];
        c20[1] += vec4(a0.z) * b0[1];
        c30[0] += vec4(a0.w) * b0[0];
        c30[1] += vec4(a0.w) * b0[1];
    }

    for(; (int(CURRENT_ITEM_OFFSET_IN_BYTES(src1_iter)) >> 1) < end_row_mtx_b; TENSOR_ITERATOR_ADVANCE_IN_BYTES(src0_iter, 8), TENSOR_ITERATOR_ADVANCE_IN_BYTES(src1_iter, 16))
    {
        /* Load values from matrix A (interleaved) and matrix B (transposed) */
        vec4 a0    = LOAD_UNPACK4_CURRENT_ITEM_HALF(src0_ptr, src0_iter);
        vec4 b0[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src1_ptr, src1_iter);

        c00[0] += vec4(a0.x) * b0[0];
        c00[1] += vec4(a0.x) * b0[1];
        c10[0] += vec4(a0.y) * b0[0];
        c10[1] += vec4(a0.y) * b0[1];
        c20[0] += vec4(a0.z) * b0[0];
        c20[1] += vec4(a0.z) * b0[1];
        c30[0] += vec4(a0.w) * b0[0];
        c30[1] += vec4(a0.w) * b0[1];
    }

    /* Multiply by the weight of matrix product */
    c00[0] = c00[0] * vec4(ALPHA);
    c00[1] = c00[1] * vec4(ALPHA);
    c10[0] = c10[0] * vec4(ALPHA);
    c10[1] = c10[1] * vec4(ALPHA);
    c20[0] = c20[0] * vec4(ALPHA);
    c20[1] = c20[1] * vec4(ALPHA);
    c30[0] = c30[0] * vec4(ALPHA);
    c30[1] = c30[1] * vec4(ALPHA);

    /* Store 4x8 block */
    STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 0), c00);
    STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), c10);
    STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), c20);
    STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), c30);
}
#endif /* GEMM_MM_INTERLEAVED_TRANSPOSED */
#else  /* DATA_TYPE_FP16 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
