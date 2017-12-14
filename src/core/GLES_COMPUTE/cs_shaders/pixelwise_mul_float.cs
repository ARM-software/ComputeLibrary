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
    TENSOR3D_PARAM_DECLARATION(src1);
    TENSOR3D_PARAM_DECLARATION(src2);
    TENSOR3D_PARAM_DECLARATION(dst);
};

BUFFER_DECLARATION(src1, 1, float, readonly);
BUFFER_DECLARATION(src2, 2, float, readonly);
BUFFER_DECLARATION(dst, 3, float, writeonly);

/** Performs a pixelwise multiplication with float scale of either integer or float inputs.
 *
 * @param[in]  src1_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src1_step_x                        src1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src1_step_z                        src1_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  src2_ptr                           Pointer to the source image. Supported data types: Same as @p src1_ptr
 * @param[in]  src2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src2_step_x                        src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src2_step_y                        src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src2_step_z                        src2_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                            Pointer to the destination image. Supported data types: Same as @p src1_ptr
 * @param[in]  dst_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination image
 * @param[in]  scale                              Float scaling factor. Supported data types: F32
 */
void main()
{
    // Get pixels pointer
    Tensor3D src1 = CONVERT_TO_TENSOR3D_STRUCT(src1);
    Tensor3D src2 = CONVERT_TO_TENSOR3D_STRUCT(src2);
    Tensor3D dst  = CONVERT_TO_TENSOR3D_STRUCT(dst);

    dst_ptr[dst.current_offset] = (src1_ptr[src1.current_offset] * src2_ptr[src2.current_offset] * float(SCALE));
}
