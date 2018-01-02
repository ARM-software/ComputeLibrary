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
#include "helpers.h"

/** This kernel applies dot product to each plane on the input tensor and the corrispective column of the reshaped weight tensor.
 *
 * @note Datatype and source width and height should be given as a preprocessor argument using -DDATA_TYPE=type, -DSRC_WIDTH=width and -DSRC_HEIGHT=height. e.g. -DDATA_TYPE=short
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 */
__kernel void gemm_mv(TENSOR3D_DECLARATION(src), IMAGE_DECLARATION(weights), VECTOR_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    int y = get_global_id(1) * 4;
    int z = get_global_id(2);

    __global uchar *current_weights = weights_ptr + weights_offset_first_element_in_bytes + z * weights_stride_y;
    __global uchar *input_ptr       = src.ptr;

    DATA_TYPE acc0 = (DATA_TYPE)0;
    DATA_TYPE acc1 = (DATA_TYPE)0;
    DATA_TYPE acc2 = (DATA_TYPE)0;
    DATA_TYPE acc3 = (DATA_TYPE)0;

    // This kernel handle 4 rows in per thread so that it can reuse the weights
    for(int i = 0; i < SRC_WIDTH; i += 4)
    {
        VEC_DATA_TYPE(DATA_TYPE, 4)
        weights = vload4(0, (__global DATA_TYPE *)(current_weights + i * weights_stride_x));

        int4 offset = (int4)i * (int4)src_stride_x + (int4)(0, 1, 2, 3) * (int4)src_stride_y;

        VEC_DATA_TYPE(DATA_TYPE, 4)
        tmp0 = vload4(0, (__global DATA_TYPE *)(input_ptr + offset.s0));
        VEC_DATA_TYPE(DATA_TYPE, 4)
        tmp1 = vload4(0, (__global DATA_TYPE *)(input_ptr + offset.s1));
        VEC_DATA_TYPE(DATA_TYPE, 4)
        tmp2 = vload4(0, (__global DATA_TYPE *)(input_ptr + offset.s2));
        VEC_DATA_TYPE(DATA_TYPE, 4)
        tmp3 = vload4(0, (__global DATA_TYPE *)(input_ptr + offset.s3));

        acc0 += dot(weights, tmp0);
        acc1 += dot(weights, tmp1);
        acc2 += dot(weights, tmp2);
        acc3 += dot(weights, tmp3);
    }

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (y + z * SRC_HEIGHT) * dst_stride_x;

    int rows_left = SRC_HEIGHT - (y + 4);

    // This if check is used to handle the last few rows when it can't be divided by the four
    if(rows_left >= 0)
    {
        VEC_DATA_TYPE(DATA_TYPE, 4)
        out = (VEC_DATA_TYPE(DATA_TYPE, 4))(acc0, acc1, acc2, acc3);
        vstore4(out, 0, (__global DATA_TYPE *)output_ptr);
    }
    else
    {
        switch(rows_left)
        {
            case -1: // three rows left; one is padding
                *((__global DATA_TYPE *)(output_ptr + 2 * dst_stride_x)) = acc2;
            case -2: // two rows left; two are padding
                *((__global DATA_TYPE *)(output_ptr + 1 * dst_stride_x)) = acc1;
            case -3: // one row left; three are padding
                *((__global DATA_TYPE *)(output_ptr + 0 * dst_stride_x)) = acc0;
                break;
        }
    }
}
