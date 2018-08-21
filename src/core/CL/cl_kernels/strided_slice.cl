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

/** Perform a strided slice operation on a given input.
 *
 * @attention Supported tensor rank: up to 4
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Input and output tensor dephts should be given as a preprocessor arguments using -DSRC_DEPTH=size. and -DDST_DEPTH=size
 * @attention Absolute start coordinates for each dimension should be given as preprocessor -DSTART_index=value e.g. -DSTART_0=2
 * @attention Strides for each dimension should be given as preprocessor -DSTRIDE_index=value e.g. -DSTRIDE_1=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void strided_slice(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, SRC_DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DST_DEPTH);

    int offset_0 = 0;
    int offset_1 = 0;
    int offset_2 = 0;
    int offset_3 = 0;

    // Calculate offset
#if defined(START_0) && defined(STRIDE_0)
    offset_0 = (int)START_0 + (int)get_global_id(0) * (int)STRIDE_0;
#endif // defined(START_0) && defined(STRIDE_0)
#if defined(START_1) && defined(STRIDE_1)
    offset_1 = (int)START_1 + (int)get_global_id(1) * (int)STRIDE_1;
#endif // defined(START_1) && defined(STRIDE_1)
#if defined(START_2) && defined(STRIDE_2)
    offset_2 = (int)START_2 + ((int)get_global_id(2) % (int)DST_DEPTH) * (int)STRIDE_2;
#endif // defined(START_2) && defined(STRIDE_2)
#if defined(START_3) && defined(STRIDE_3)
    offset_3 = (int)START_3 + ((int)get_global_id(2) / (int)DST_DEPTH) * (int)STRIDE_3;
#endif // defined(START_2) && defined(STRIDE_2)

    // Store result
    *((__global DATA_TYPE *)(output.ptr)) = *((__global DATA_TYPE *)tensor4D_offset(&input, offset_0, offset_1, offset_2, offset_3));
}
