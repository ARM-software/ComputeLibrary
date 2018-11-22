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

#if defined(DATA_TYPE) && defined(SRC_DEPTH) && defined(STRIDE)

#define CALCULATE_SRC_COORDINATES(xo, yo, zo, xi, yi, zi)     \
    ({                                                        \
        int offset = zo / (int)SRC_DEPTH;                     \
        xi         = xo * (int)STRIDE + offset % (int)STRIDE; \
        yi         = yo * (int)STRIDE + offset / (int)STRIDE; \
        zi         = zo % SRC_DEPTH;                          \
    })

/** Performs a reorganization layer of input tensor to the output tensor when the data layout is NCHW
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The depth of the input tensor must be passed at compile time using -DSRC_DEPTH: e.g. -DSRC_DEPTH=64
 * @note The distance between 2 consecutive pixels along the x and y direction must be passed at compile time using -DSTRIDE: e.g. -DSTRIDE=2
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void reorg_layer_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(dst);

    int xo = get_global_id(0);
    int yo = get_global_id(1);
    int zo = get_global_id(2);
    int xi, yi, zi;

    CALCULATE_SRC_COORDINATES(xo, yo, zo, xi, yi, zi);

    int src_offset                   = xi * sizeof(DATA_TYPE) + yi * src_stride_y + zi * src_stride_z;
    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes + src_offset));
}

/** Performs a reorganization layer of input tensor to the output tensor when the data layout is NHWC
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The depth of the input tensor must be passed at compile time using -DSRC_DEPTH: e.g. -DSRC_DEPTH=64
 * @note The distance between 2 consecutive pixels along the x and y direction must be passed at compile time using -DSTRIDE: e.g. -DSTRIDE=2
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void reorg_layer_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(dst);

    int xo = get_global_id(1);
    int yo = get_global_id(2);
    int zo = get_global_id(0);
    int xi, yi, zi;

    CALCULATE_SRC_COORDINATES(xo, yo, zo, xi, yi, zi);

    int src_offset = zi * sizeof(DATA_TYPE) + xi * src_stride_y + yi * src_stride_z;

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes + src_offset));
}
#endif // // defined(DATA_TYPE) && defined(SRC_DEPTH) && defined(STRIDE)