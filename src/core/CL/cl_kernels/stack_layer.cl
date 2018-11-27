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

#if defined(DATA_TYPE) && defined(AXIS) && defined(SRC_DIM2) && defined(DST_DIM3)

#if AXIS == 0
#define X_DST (idx_input)
#define Y_DST (x_src)
#define Z_DST (y_src)
#define W_DST (z_src)
#define K_DST (w_src)
#elif AXIS == 1 // AXIS == 1
#define X_DST (x_src)
#define Y_DST (idx_input)
#define Z_DST (y_src)
#define W_DST (z_src)
#define K_DST (w_src)
#elif AXIS == 2 // AXIS == 2
#define X_DST (x_src)
#define Y_DST (y_src)
#define Z_DST (idx_input)
#define W_DST (z_src)
#define K_DST (w_src)
#elif AXIS == 3 // AXIS == 3
#define X_DST (x_src)
#define Y_DST (y_src)
#define Z_DST (z_src)
#define W_DST (idx_input)
#define K_DST (w_src)
#elif AXIS == 4 // AXIS == 4
#define X_DST (x_src)
#define Y_DST (y_src)
#define Z_DST (z_src)
#define W_DST (w_src)
#define K_DST (idx_input)
#else // AXIS not supported
#error "Not supported axis"
#endif // AXIS == 0

/** OpenCL kernel to stack a rank-R tensor into one with rank-(R+1) along the axis dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note The dimension to stack the tensors along has to be passed at compile time using -DAXIS. i.e. -DAXIS=1
 * @note Dimension 2 of the input tensor must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM2=112)
 * @note Dimension 3 of the output tensor must be passed at compile time using -DDST_DIM3 (e.g. -DDST_DIM3=112)
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  idx_input                         Index of the input tensor in the list of tensors to stack
 */
__kernel void stack_layer(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    unsigned int idx_input)
{
    uint x_src = get_global_id(0);
    uint y_src = get_global_id(1);
    uint z_src = (get_global_id(2) % SRC_DIM2);
    uint w_src = (get_global_id(2) / SRC_DIM2);

    __global DATA_TYPE *src = (__global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes + x_src * sizeof(DATA_TYPE) + y_src * src_stride_y + z_src * src_stride_z + w_src * src_stride_w);

    __global DATA_TYPE *dst = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + X_DST * sizeof(DATA_TYPE) + Y_DST * dst_stride_y + Z_DST * dst_stride_z + W_DST * dst_stride_w + K_DST *
                                                     dst_stride_w * (uint)DST_DIM3);

    *dst = *src;
}

#undef X_DST
#undef Y_DST
#undef Z_DST
#undef W_DST
#endif // defined(DATA_TYPE) && defined(AXIS) && defined(SRC_DIM2) && defined(DST_DIM3)
