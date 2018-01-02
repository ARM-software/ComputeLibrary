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

#include "helpers_cs.h"

/** Calculate the absolute difference of two input images.
 *
 * @param[in]  src1_ptr   Pointer to the first source image. Supported data types: U8
 * @param[in]  src1_attrs The attributes of the first source image
 * @param[in]  src2_ptr   Pointer to the second source image. Supported data types: Same as @p in1_ptr
 * @param[in]  src2_attrs The attributes of the second source image
 * @param[out] dst_ptr    Pointer to the destination image. Supported data types: Same as @p in1_ptr
 * @param[in]  dst_attrs  The attributes of the destination image
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes src1_attrs;
    ImageAttributes src2_attrs;
    ImageAttributes dst_attrs;
};

TENSOR_DECLARATION(1, src1Buffer, uint, src1_ptr, src1_shift, 2, readonly);
TENSOR_DECLARATION(2, src2Buffer, uint, src2_ptr, src2_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    ImageIterator src1_iter = CONVERT_TO_IMAGE_ITERATOR(src1_attrs, src1_shift);
    ImageIterator src2_iter = CONVERT_TO_IMAGE_ITERATOR(src2_attrs, src2_shift);
    ImageIterator dst_iter  = CONVERT_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    lowp uvec4 tmp1 = LOAD_UNPACK4_CURRENT_ITEM_U8(src1_ptr, src1_iter);
    lowp uvec4 tmp2 = LOAD_UNPACK4_CURRENT_ITEM_U8(src2_ptr, src2_iter);
    lowp uvec4 diff = uvec4(abs(ivec4(tmp1 - tmp2)));

    STORE_PACK4_CURRENT_ITEM_U8(dst_ptr, dst_iter, diff);
}
