/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

#define SWAP_ROW_func(u0, l0) \
    {                         \
        tmp_swap = u0;        \
        u0       = l0;        \
        l0       = tmp_swap;  \
    }

#define SWAP_4x4_func(u0, u1, u2, u3, l0, l1, l2, l3) \
    {                                                 \
        vec4 tmp_swap;                                \
        SWAP_ROW_func(u0, l0);                        \
        SWAP_ROW_func(u1, l1);                        \
        SWAP_ROW_func(u2, l2);                        \
        SWAP_ROW_func(u3, l3);                        \
    }

#define TRANSPOSE_4x4_func(u0, u1, u2, u3) \
    {                                      \
        mat4x4 matin, matout;              \
        matin[0] = u0;                     \
        matin[1] = u1;                     \
        matin[2] = u2;                     \
        matin[3] = u3;                     \
        matout   = transpose(matin);       \
        u0       = matout[0];              \
        u1       = matout[1];              \
        u2       = matout[2];              \
        u3       = matout[3];              \
    }

/** This OpenGL ES kernel computes the matrix transposition of input matrix
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note Optimization name must be passed using "#define OPTIMIZATION_NAME" for F16. e.g. "#define TRANSPOSE_8X8"
 *
 * @param[in]  src_ptr   Pointer to the source matrix. Supported data types: F32/F16
 * @param[in]  src_attrs The attributes of the source matrix
 * @param[out] dst_ptr   Pointer to the destination matrix Supported data type: same as src_ptr
 * @param[in]  dst_attrs The attributes of the destination matrix
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src_attrs;
    ImageAttributes dst_attrs;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    // compute source address
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    // load the NxN block at (x, y)
    vec4 u0 = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, 0, 0));
    vec4 u1 = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
    vec4 u2 = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
    vec4 u3 = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, 0, 3));

    // transpose the block
    TRANSPOSE_4x4_func(u0, u1, u2, u3);

    // store the block at (y, x)
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, uint(16) * uint(gl_GlobalInvocationID.y) + uint(4) * uint(gl_GlobalInvocationID.x) * (dst_attrs.stride_y));

    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 0), u0);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), u1);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), u2);
    VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), u3);
}

#elif defined(DATA_TYPE_FP16) /* DATA_TYPE_FP16 */

#if defined(TRANSPOSE_4X4)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);

void main(void)
{
    // compute source address
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    // load the NxN block at (x, y)
    vec4 u0 = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 0));
    vec4 u1 = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
    vec4 u2 = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
    vec4 u3 = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 3));

    // transpose the block
    TRANSPOSE_4x4_func(u0, u1, u2, u3);

    // store the block at (y, x)
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, uint(8) * uint(gl_GlobalInvocationID.y) + uint(gl_GlobalInvocationID.x) * (dst_attrs.step_y));

    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 0), u0);
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 1), u1);
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 2), u2);
    STORE_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, 3), u3);
}

#elif defined(TRANSPOSE_8X8) /* TRANSPOSE_8X8 */
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main(void)
{
    // compute source address
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    vec4 u[8][2];

    for(int i = 0; i < 8; i++)
    {
        u[i] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, i));
    }

    // transpose the block
    TRANSPOSE_4x4_func(u[0][0], u[1][0], u[2][0], u[3][0]);
    TRANSPOSE_4x4_func(u[0][1], u[1][1], u[2][1], u[3][1]);
    TRANSPOSE_4x4_func(u[4][0], u[5][0], u[6][0], u[7][0]);
    TRANSPOSE_4x4_func(u[4][1], u[5][1], u[6][1], u[7][1]);
    SWAP_4x4_func(u[0][1], u[1][1], u[2][1], u[3][1], u[4][0], u[5][0], u[6][0], u[7][0]);

    // store the block at (y, x)
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, uint(16) * uint(gl_GlobalInvocationID.y) + uint(gl_GlobalInvocationID.x) * (dst_attrs.step_y));

    for(int i = 0; i < 8; i++)
    {
        STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, i), u[i]);
    }
}

#elif defined(TRANSPOSE_8X8_SQUARE) /* TRANSPOSE_8x8_SQUARE */
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main(void)
{
    ImageIterator src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    if(gl_GlobalInvocationID.x <= gl_GlobalInvocationID.y)
    {
        uint blk1_offset_in_bytes = CURRENT_ITEM_OFFSET_IN_BYTES(src_iter);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, uint(16) * uint(gl_GlobalInvocationID.y) + uint(gl_GlobalInvocationID.x) * (dst_attrs.step_y));
        uint blk2_offset_in_bytes = CURRENT_ITEM_OFFSET_IN_BYTES(dst_iter);

        // load block1
        vec4 u1[8][2];

        SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(src_iter, blk1_offset_in_bytes);
        for(int i = 0; i < 8; i++)
        {
            u1[i] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, i));
        }

        // transpose block1
        TRANSPOSE_4x4_func(u1[0][0], u1[1][0], u1[2][0], u1[3][0]);
        TRANSPOSE_4x4_func(u1[0][1], u1[1][1], u1[2][1], u1[3][1]);
        TRANSPOSE_4x4_func(u1[4][0], u1[5][0], u1[6][0], u1[7][0]);
        TRANSPOSE_4x4_func(u1[4][1], u1[5][1], u1[6][1], u1[7][1]);
        SWAP_4x4_func(u1[0][1], u1[1][1], u1[2][1], u1[3][1], u1[4][0], u1[5][0], u1[6][0], u1[7][0]);

        // write to block2
        SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(dst_iter, blk2_offset_in_bytes);
        for(int i = 0; i < 8; i++)
        {
            STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, i), u1[i]);
        }

        // load block2
        vec4 u2[8][2];

        SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(src_iter, blk2_offset_in_bytes);
        for(int i = 0; i < 8; i++)
        {
            u2[i] = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, i));
        }

        // transpose block2
        TRANSPOSE_4x4_func(u2[0][0], u2[1][0], u2[2][0], u2[3][0]);
        TRANSPOSE_4x4_func(u2[0][1], u2[1][1], u2[2][1], u2[3][1]);
        TRANSPOSE_4x4_func(u2[4][0], u2[5][0], u2[6][0], u2[7][0]);
        TRANSPOSE_4x4_func(u2[4][1], u2[5][1], u2[6][1], u2[7][1]);
        SWAP_4x4_func(u2[0][1], u2[1][1], u2[2][1], u2[3][1], u2[4][0], u2[5][0], u2[6][0], u2[7][0]);

        // write to block1
        SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(dst_iter, blk1_offset_in_bytes);
        for(int i = 0; i < 8; i++)
        {
            STORE_PACK8_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, 0, i), u2[i]);
        }
    }
}

#endif /* TRANSPOSE_4X4 */

#endif /* DATA_TYPE_FP32 */
