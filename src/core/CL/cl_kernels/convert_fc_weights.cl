/*
 * Copyright (c) 2018 ARM Limited.
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
#include "helpers.h"

#if defined(DATA_TYPE) && defined(FACTOR_1) && defined(FACTOR_2)
/** Perform a NCHW -> NHWC or NHWC -> NCHW conversion for Fully Connected 2D weights.
 *
 * For NCHW -> NHWC, FACTOR_1 will be equal to the product of the first two dimensions of FullyConnectedLayer's input and FACTOR_2 will represent the number of channels of that tensor.
 * For NHWC -> NCHW, FACTOR_1 and FACTOR_2 will hold the same values, but swapped.
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Original input tensor width*height and depth should be given as a preprocessor argument using -DFACTOR_1=size and -DFACTOR_2=size for NCHW and vice versa for NHWC. e.g. -DFACTOR_1=256 and -DFACTOR_2=128
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8, S8, QS8, QASYMM8, U16, S16, QS16, U32, S32, QS32, F16, F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void convert_fc_weights(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(0) * dst_stride_x + (get_global_id(1) % FACTOR_1 * FACTOR_2 + get_global_id(1) / FACTOR_1) * dst_stride_y;

    *((__global DATA_TYPE *)dst_addr) = *((__global DATA_TYPE *)src.ptr);
}
#endif // defined(DATA_TYPE) && defined(FACTOR_1) && defined(FACTOR_2)
