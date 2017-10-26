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

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src1);
    IMAGE_PARAM_DECLARATION(src2);
    IMAGE_PARAM_DECLARATION(dst);
};

BUFFER_DECLARATION(src1, 1, uint, readonly);
BUFFER_DECLARATION(src2, 2, uint, readonly);
BUFFER_DECLARATION(dst, 3, uint, writeonly);

/** Calculate the absolute difference of two input images.
 *
 * @param[in]  src1_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  src1_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  src2_ptr                           Pointer to the second source image. Supported data types: Same as @p in1_ptr
 * @param[in]  src2_stride_x                      Stride of the second source image in X dimension (in bytes)
 * @param[in]  src2_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      Stride of the second source image in Y dimension (in bytes)
 * @param[in]  src2_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes The offset of the first element in the second source image
 * @param[out] dst_ptr                            Pointer to the destination image. Supported data types: Same as @p in1_ptr
 * @param[in]  dst_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination image
 */
void main(void)
{
    Image src1 = CONVERT_TO_IMAGE_STRUCT(src1);
    Image src2 = CONVERT_TO_IMAGE_STRUCT(src2);
    Image dst  = CONVERT_TO_IMAGE_STRUCT(dst);

    uvec4 tmp1 = UNPACK(LOAD4(src1, CURRENT_OFFSET(src1)), uint, uvec4);
    uvec4 tmp2 = UNPACK(LOAD4(src2, CURRENT_OFFSET(src2)), uint, uvec4);
    uvec4 diff = uvec4(abs(ivec4(tmp1 - tmp2)));

    STORE4(dst, CURRENT_OFFSET(dst), PACK(diff, uvec4, uint));
}
