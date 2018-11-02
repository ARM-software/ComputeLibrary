/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#if defined(DATA_TYPE) && defined(WIDTH_OUTPUT) && defined(ELEMENT_SIZE) && defined(WIDTH_INPUT)

#if ELEMENT_SIZE == 1
#define COND_DATA_TYPE char
#elif ELEMENT_SIZE == 2
#define COND_DATA_TYPE short
#elif ELEMENT_SIZE == 4
#define COND_DATA_TYPE int
#else // ELEMENT_SIZE
#error "Element size not support"
#endif // ELEMENT_SIZE

/** This kernel performs a reshaping of the output of the convolution layer
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width of the input tensor must be passed at compile time using -DWIDTH_INPUT: e.g. -DWIDTH_INPUT=320
 * @note The width of the output tensor must be passed at compile time using -DWIDTH_OUTPUT: e.g. -DWIDTH_OUTPUT=600
 * @note The element size must be passed at compile time using -DELEMENT_SIZE: e.g. -DELEMENT_SIZE=4
 * @note In case of grouping the GROUPING flag must be passed at compile time using -DGROUPING
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
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
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void col2im(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint dst_stride_w)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const uint xd = get_global_id(1) % WIDTH_OUTPUT; // x coordinate of the destination tensor
    const uint yd = get_global_id(1) / WIDTH_OUTPUT; // y coordinate of the destination tensor

    VEC_DATA_TYPE(DATA_TYPE, 8)
    data = vload8(0, (__global DATA_TYPE *)src.ptr);

    uint  x         = get_global_id(0) * 8;
    uint8 x_clamped = x + (uint8)(0, 1, 2, 3, 4, 5, 6, 7);

    VEC_DATA_TYPE(COND_DATA_TYPE, 8)
    cond0 = CONVERT((x_clamped < WIDTH_INPUT), VEC_DATA_TYPE(COND_DATA_TYPE, 8));

    // Clamp x if out-of-bounds
    x_clamped = select((uint8)x, x_clamped, convert_int8(cond0));

    // If out-of-bound, overwrite with the first element
    data = select((VEC_DATA_TYPE(DATA_TYPE, 8))data.s0, data, cond0);

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes;

#if defined(GROUPING)
    // Compute output offset (batches on 4th dimension, no need to compute manually)
    int idx = yd * dst_stride_y + xd * dst_stride_x;

    const uint group = get_global_id(2); // group ID
    x_clamped += group * WIDTH_INPUT;
#else  /* defined(GROUPING) */
    // Compute output offset (batches on 3rd dimension)
    int idx = yd * dst_stride_y + xd * dst_stride_x + get_global_id(2) * dst_stride_w;
#endif /* GROUPING */

    // Store value
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s0 * dst_stride_z)) = data.s0;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s1 * dst_stride_z)) = data.s1;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s2 * dst_stride_z)) = data.s2;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s3 * dst_stride_z)) = data.s3;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s4 * dst_stride_z)) = data.s4;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s5 * dst_stride_z)) = data.s5;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s6 * dst_stride_z)) = data.s6;
    *((__global DATA_TYPE *)(output_ptr + idx + x_clamped.s7 * dst_stride_z)) = data.s7;
}
#endif // defined(DATA_TYPE) && defined(WIDTH_OUTPUT) && defined(ELEMENT_SIZE) && defined(WIDTH_INPUT)
