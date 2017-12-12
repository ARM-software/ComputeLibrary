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

precision mediump float;

/** Apply normalize_planar_yuv layer.
 *
 * @param[in]  src_ptr    Pointer to the first source tensor. Supported data types: F16
 * @param[in]  src_attrs  The attributes of the source tensor
 * @param[out] dst_ptr    Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs  The attributes of the destination tensor
 * @param[in]  mean_ptr   Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_attrs The attributes of the mean tensor
 * @param[in]  sd_ptr     Standard deviation values tensor,pointer to the sd tensor. Supported data types: same as @p src_ptr
 * @param[in]  sd_attrs   The attributes of the sd tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    VectorAttributes   mean_attrs;
    VectorAttributes   sd_attrs;
};

TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, meanBuffer, uvec2, mean_ptr, mean_shift, 3, readonly);
TENSOR_DECLARATION(4, sdBuffer, uvec2, sd_ptr, sd_shift, 3, readonly);

void main(void)
{
    Tensor3DIterator src_iter  = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);
    VectorIterator   mean_iter = CONVERT_TO_VECTOR_ITERATOR(mean_attrs, mean_shift);
    VectorIterator   sd_iter   = CONVERT_TO_VECTOR_ITERATOR(sd_attrs, sd_shift);

    vec4 unpacked_s[3];
    vec4 tmp;
    vec4 result;

    uint current_slice = gl_GlobalInvocationID.z;
    unpacked_s[0]      = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    unpacked_s[1]      = LOAD_UNPACK4_HALF(mean_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(mean_iter, current_slice * mean_attrs.stride_x));
    unpacked_s[2]      = LOAD_UNPACK4_HALF(sd_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(sd_iter, current_slice * sd_attrs.stride_x));

    if((current_slice % uint(4)) == uint(0))
    {
        tmp    = unpacked_s[0] - unpacked_s[1].x;
        result = tmp / unpacked_s[2].x;

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else if((current_slice % uint(4)) == uint(1))
    {
        tmp    = unpacked_s[0] - unpacked_s[1].y;
        result = tmp / unpacked_s[2].y;

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else if((current_slice % uint(4)) == uint(2))
    {
        tmp    = unpacked_s[0] - unpacked_s[1].z;
        result = tmp / unpacked_s[2].z;

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else
    {
        tmp    = unpacked_s[0] - unpacked_s[1].w;
        result = tmp / unpacked_s[2].w;

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
}
