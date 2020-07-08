/*
 * Copyright (c) 2016-2018 Arm Limited.
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

precision mediump float;
#define ADD(x, y) (x) + (y)

/** This function add two tensors.
 *
 * @param[in]  src1_ptr   Pointer to the first source tensor. Supported data types: F16
 * @param[in]  src1_attrs The attributes of the first source tensor
 * @param[in]  src2_ptr   Pointer to the second source tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  src2_attrs The attributes of the second source tensor
 * @param[out] dst_ptr    Pointer to the destination tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  dst_attrs  The attributes of the destination tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src1_attrs;
    Tensor3DAttributes src2_attrs;
    Tensor3DAttributes dst_attrs;
};

TENSOR_DECLARATION(1, src1Buffer, uvec4, src1_ptr, src1_shift, 4, readonly);
TENSOR_DECLARATION(2, src2Buffer, uvec4, src2_ptr, src2_shift, 4, readonly);
TENSOR_DECLARATION(3, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main(void)
{
    Tensor3DIterator src1_iter = CONVERT_TO_TENSOR3D_ITERATOR(src1_attrs, src1_shift);
    Tensor3DIterator src2_iter = CONVERT_TO_TENSOR3D_ITERATOR(src2_attrs, src2_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4 tmp1[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src1_ptr, src1_iter);
    vec4 tmp2[2] = LOAD_UNPACK8_CURRENT_ITEM_HALF(src2_ptr, src2_iter);
    vec4 addition[2];
    addition[0] = ADD(tmp1[0], tmp2[0]);
    addition[1] = ADD(tmp1[1], tmp2[1]);

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, addition);
}
