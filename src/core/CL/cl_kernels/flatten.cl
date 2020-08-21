/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#if defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(SRC_DEPTH)

/** This opencl kernel flattens the first 3 dimensions of the input tensor
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The width, height and depth of the input tensor must be passed at compile time using -DSRC_WIDTH, -DSRC_HEIGHT and -DSRC_DEPTH. e.g. -DSRC_WIDTH=24, -DSRC_HEIGHT=24, -DSRC_DEPTH=16
 * @note If the output has 3 dimensions, the 2nd dimension of the output tensor must be passed at compile time using -DDST_DIM1. e.g -DDST_DIM1=3
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void flatten(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DEPTH);

    uint c  = get_global_id(2) % SRC_DEPTH; // input feature map
    uint b0 = get_global_id(2) / SRC_DEPTH; // batch id
    uint b1 = 0;

#if defined(DST_DIM1)
    uint b_tmp = b0;
    b0         = b_tmp % DST_DIM1; // batch id0
    b1         = b_tmp / DST_DIM1; // batch id1
#endif                             // defined(DST_DIM1)

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) + get_global_id(1) * (uint)SRC_WIDTH + c * (uint)(SRC_WIDTH * SRC_HEIGHT)) * sizeof(
                                     DATA_TYPE) + b0 * dst_stride_y + b1 * dst_stride_z;

    *((__global DATA_TYPE *)output_ptr) = *((__global DATA_TYPE *)src.ptr);
}
#endif // defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT)