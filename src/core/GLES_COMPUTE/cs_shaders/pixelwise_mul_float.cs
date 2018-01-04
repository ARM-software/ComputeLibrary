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

/** Performs a pixelwise multiplication with float scale of float inputs.
 *
 * @param[in]  src1_ptr   Pointer to the first source tensor. Supported data types: F32
 * @param[in]  src1_attrs The attributes of the first source tensor
 * @param[in]  src2_ptr   Pointer to the second source tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  src2_attrs The attributes of the second source tensor
 * @param[out] dst_ptr    Pointer to the destination tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  dst_attrs  The attributes of the destination tensor
 * @param[in]  scale      Float scaling factor. Supported data types: F32
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src1_attrs;
    Tensor3DAttributes src2_attrs;
    Tensor3DAttributes dst_attrs;
};
TENSOR_DECLARATION(1, src1Buffer, float, src1_ptr, src1_shift, 2, readonly);
TENSOR_DECLARATION(2, src2Buffer, float, src2_ptr, src2_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main()
{
    // Get pixels pointer
    Tensor3DIterator src1_iter = CONVERT_TO_TENSOR3D_ITERATOR(src1_attrs, src1_shift);
    Tensor3DIterator src2_iter = CONVERT_TO_TENSOR3D_ITERATOR(src2_attrs, src2_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float result = LOAD_CURRENT_ITEM(src1_ptr, src1_iter) * LOAD_CURRENT_ITEM(src2_ptr, src2_iter) * float(SCALE);
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, result);
}
