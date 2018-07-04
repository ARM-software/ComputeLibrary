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

#if defined(DATA_TYPE) && defined(ELEMENT_SIZE)

#if ELEMENT_SIZE == 1
#define COND_DATA_TYPE char
#elif ELEMENT_SIZE == 2
#define COND_DATA_TYPE short
#elif ELEMENT_SIZE == 4
#define COND_DATA_TYPE int
#else // ELEMENT_SIZE
#error "Element size not support"
#endif // ELEMENT_SIZE

#if defined(CONVOLVED_WIDTH) && defined(STRIDE_Y) && defined(KERNEL_DEPTH)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 1x1 and the stride_x = 1
 *
 * @note This kernel computes 4 elements
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The stride along the Y direction must be passed at compile time using -DSTRIDE_Y: e.g. -DSTRIDE_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col1x1_stridex1_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const uint xc    = get_global_id(0) * 4;            // x coordinate in the convolved tensor
    const uint yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const uint ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const uint batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Clamp xc
    // The strategy clamps at "xc" as it will be a valid value for sure
    uint4 xc_clamped = xc + (uint4)(0, 1, 2, 3);

    // Check which values are valid
    const VEC_DATA_TYPE(COND_DATA_TYPE, 4) cond0 = CONVERT((xc_clamped < SRC_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, 4));

    xc_clamped = select((uint4)xc, xc_clamped, convert_int4(cond0));

    // Calculate input indices
    const uint xi = xc;
    const uint yi = yc * STRIDE_Y;

    // Calculate output indices
    const uint  xo = ch;
    const uint4 yo = xc_clamped + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + xi * src_stride_x + yi * src_stride_y + ch * src_stride_z + batch * src_stride_w;

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + batch * dst_stride_w;

    VEC_DATA_TYPE(DATA_TYPE, 4)
    data = vload4(0, (__global DATA_TYPE *)input_ptr);

    // If out-of-bound, overwrite with the first element
    data = select((VEC_DATA_TYPE(DATA_TYPE, 4))data.s0, data, cond0);

    *(__global DATA_TYPE *)(output_ptr + yo.s0 * dst_stride_y) = data.s0;
    *(__global DATA_TYPE *)(output_ptr + yo.s1 * dst_stride_y) = data.s1;
    *(__global DATA_TYPE *)(output_ptr + yo.s2 * dst_stride_y) = data.s2;
    *(__global DATA_TYPE *)(output_ptr + yo.s3 * dst_stride_y) = data.s3;

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)(output_ptr + yo.s0 * dst_stride_y) + 1) = 1.0f;
        *((__global DATA_TYPE *)(output_ptr + yo.s1 * dst_stride_y) + 1) = 1.0f;
        *((__global DATA_TYPE *)(output_ptr + yo.s2 * dst_stride_y) + 1) = 1.0f;
        *((__global DATA_TYPE *)(output_ptr + yo.s3 * dst_stride_y) + 1) = 1.0f;
    }
#endif // HAS_BIAS
}
#endif // defined(CONVOLVED_WIDTH) && defined(STRIDE_Y) && defined(KERNEL_DEPTH)

#define PTR_TO_VALUE(PTR, DATA_TYPE) *((__global DATA_TYPE *)(PTR))

#if defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE)

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 5x5
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note The dilation_x and dilation_y must be passed at compile time using -DDILATION_X and -DDILATION_Y: e.g. -DDILATION_X=1, -DDILATION_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col_generic_nhwc(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int src_stride_y_int = (int)src_stride_y;
    const int src_stride_z_int = (int)src_stride_z;
    const int xc               = get_global_id(1);                    // x coordinate in the convolved tensor
    const int yc               = get_global_id(2) % CONVOLVED_HEIGHT; // y coordinate in the convolved tensor
    const int ch               = get_global_id(0);                    // input feature map
    const int batch            = get_global_id(2) / CONVOLVED_HEIGHT; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
    const int xo = ch * KERNEL_HEIGHT * KERNEL_WIDTH;
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr  = src_ptr + src_offset_first_element_in_bytes + xi * src_stride_y_int + yi * src_stride_z_int + ch * src_stride_x + batch * src_stride_w;
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + yo * dst_stride_y + batch * dst_stride_w;

    for(int yk = 0; yk < KERNEL_HEIGHT; ++yk)
    {
        const int dilated_offset_y = yk * DILATION_Y;
        const int y0               = yi + dilated_offset_y;
        if(y0 >= 0 && y0 < SRC_HEIGHT)
        {
            int xk;
            for(xk = 0; xk < KERNEL_WIDTH; xk++)
            {
                const int dilated_offset_x = xk * DILATION_X;
                const int x0               = xi + dilated_offset_x;
                if(x0 >= 0 && x0 < SRC_WIDTH)
                {
                    *((__global DATA_TYPE *)output_ptr) = PTR_TO_VALUE(input_ptr + dilated_offset_x * src_stride_y + dilated_offset_y * src_stride_z, DATA_TYPE);
                }
                else
                {
                    *((__global DATA_TYPE *)output_ptr) = PAD_VALUE;
                }
                output_ptr += 1 * sizeof(DATA_TYPE);
            }
        }
        else
        {
            for(int xk = 0; xk < KERNEL_WIDTH; xk++)
            {
                *((__global DATA_TYPE *)output_ptr) = (DATA_TYPE)PAD_VALUE;
                output_ptr += 1 * dst_stride_x;
            }
        }
    }
#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)output_ptr) = 1.0f;
        output_ptr += 1 * dst_stride_x;
    }
#endif // HAS_BIAS
}

/** This kernel performs a reshaping of the input tensor (with layout NHWC) to a tensor used to perform convolution using GEMM when the kernel size is 3x3
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col3x3_nhwc(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int src_stride_y_int = (int)src_stride_y;
    const int src_stride_z_int = (int)src_stride_z;
    const int xc               = get_global_id(1);                    // x coordinate in the convolved tensor
    const int yc               = get_global_id(2) % CONVOLVED_HEIGHT; // y coordinate in the convolved tensor
    const int ch               = get_global_id(0);                    // input feature map
    const int batch            = get_global_id(2) / CONVOLVED_HEIGHT; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
    const int xo = ch * 9;                    // 3x3
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr  = src_ptr + src_offset_first_element_in_bytes + xi * src_stride_y_int + yi * src_stride_z_int + ch * src_stride_x + batch * src_stride_w;
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + yo * dst_stride_y + batch * dst_stride_w;

    VEC_DATA_TYPE(DATA_TYPE, 3)
    row0 = (VEC_DATA_TYPE(DATA_TYPE, 3))(PAD_VALUE);
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row1 = (VEC_DATA_TYPE(DATA_TYPE, 3))(PAD_VALUE);
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row2 = (VEC_DATA_TYPE(DATA_TYPE, 3))(PAD_VALUE);

    const int3 y = (int3)yi + (int3)(0, 1, 2);
    // Guard against reading outside the input buffer, there is no padding in Z so we check if ry is inside the buffer.
    if(y.s0 >= 0 && y.s0 < SRC_HEIGHT)
    {
        row0 = (VEC_DATA_TYPE(DATA_TYPE, 3))(
                   PTR_TO_VALUE(input_ptr + 0 * src_stride_y, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 1 * src_stride_y, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 2 * src_stride_y, DATA_TYPE));
    }

    if(y.s1 >= 0 && y.s1 < SRC_HEIGHT)
    {
        row1 = (VEC_DATA_TYPE(DATA_TYPE, 3))(
                   PTR_TO_VALUE(input_ptr + 0 * src_stride_y + 1 * src_stride_z, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 1 * src_stride_y + 1 * src_stride_z, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 2 * src_stride_y + 1 * src_stride_z, DATA_TYPE));
    }

    if(y.s2 >= 0 && y.s2 < SRC_HEIGHT)
    {
        row2 = (VEC_DATA_TYPE(DATA_TYPE, 3))(
                   PTR_TO_VALUE(input_ptr + 0 * src_stride_y + 2 * src_stride_z, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 1 * src_stride_y + 2 * src_stride_z, DATA_TYPE),
                   PTR_TO_VALUE(input_ptr + 2 * src_stride_y + 2 * src_stride_z, DATA_TYPE));
    }

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
    // Put 0 if the value is out-of-bound
    const int3 x = (int3)xi + (int3)(0, 1, 2);
    VEC_DATA_TYPE(COND_DATA_TYPE, 3)
    cond0 = CONVERT((x >= (int3)0 && x < (int3)SRC_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, 3));
    row0  = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row0, cond0);
    row1  = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row1, cond0);
    row2  = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row2, cond0);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
    vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row0.s012, row1.s012, row2.s01), 0, (__global DATA_TYPE *)output_ptr);
    *((__global DATA_TYPE *)output_ptr + 8) = row2.s2;

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)output_ptr + 9) = 1.0f;
    }
#endif // HAS_BIAS
}

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 3x3
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col3x3_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);                // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const int batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
    const int xo = ch * 9;                    // 3x3
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + xi * (int)src_stride_x + yi * (int)src_stride_y + ch * src_stride_z + batch * src_stride_w;

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + yo * dst_stride_y + batch * dst_stride_w;

    VEC_DATA_TYPE(DATA_TYPE, 3)
    row0 = vload3(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row1 = vload3(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row2 = vload3(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y));

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
    // Put 0 if the value is out-of-bound
    int3 x = (int3)xi + (int3)(0, 1, 2);
    int3 y = (int3)yi + (int3)(0, 1, 2);

    VEC_DATA_TYPE(COND_DATA_TYPE, 3)
    cond0 = CONVERT((x >= (int3)0 && x < (int3)SRC_WIDTH && (int3)(y.s0 >= 0 && y.s0 < SRC_HEIGHT)), VEC_DATA_TYPE(COND_DATA_TYPE, 3));
    VEC_DATA_TYPE(COND_DATA_TYPE, 3)
    cond1 = CONVERT((x >= (int3)0 && x < (int3)SRC_WIDTH && (int3)(y.s1 >= 0 && y.s1 < SRC_HEIGHT)), VEC_DATA_TYPE(COND_DATA_TYPE, 3));
    VEC_DATA_TYPE(COND_DATA_TYPE, 3)
    cond2 = CONVERT((x >= (int3)0 && x < (int3)SRC_WIDTH && (int3)(y.s2 >= 0 && y.s2 < SRC_HEIGHT)), VEC_DATA_TYPE(COND_DATA_TYPE, 3));

    row0 = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row0, cond0);
    row1 = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row1, cond1);
    row2 = select((VEC_DATA_TYPE(DATA_TYPE, 3))PAD_VALUE, row2, cond2);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0

    vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row0.s012, row1.s012, row2.s01), 0, (__global DATA_TYPE *)output_ptr);
    *((__global DATA_TYPE *)output_ptr + 8) = row2.s2;

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)output_ptr + 9) = 1.0f;
    }
#endif // HAS_BIAS
}

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 5x5
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col5x5_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);                // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const int batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
    const int xo = ch * 25;                   // 5x5
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
    // Put 0 if the value is out-of-bound
    int4 x0 = (int4)xi + (int4)(0, 1, 2, 3);
    int4 y0 = (int4)yi + (int4)(0, 1, 2, 3);
    int  x1 = xi + 4;
    int  y1 = yi + 4;

    // Check if we could have out-of-bounds elements in the x direction
    VEC_DATA_TYPE(COND_DATA_TYPE, 4)
    x0_condition = CONVERT((x0 >= (int4)0 && x0 < (int4)SRC_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, 4));
    VEC_DATA_TYPE(COND_DATA_TYPE, 4)
    y0_condition                = CONVERT((y0 >= (int4)0 && y0 < (int4)SRC_HEIGHT), VEC_DATA_TYPE(COND_DATA_TYPE, 4));
    COND_DATA_TYPE x1_condition = (COND_DATA_TYPE)(x1 >= 0 && x1 < SRC_WIDTH);
    COND_DATA_TYPE y1_condition = (COND_DATA_TYPE)(y1 >= 0 && y1 < SRC_HEIGHT);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0

    // Get input and output address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + xi * (int)src_stride_x + yi * (int)src_stride_y + ch * src_stride_z + batch * src_stride_w;

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + yo * dst_stride_y + batch * dst_stride_w;

    {
        VEC_DATA_TYPE(DATA_TYPE, 4)
        row00 = vload4(0, (__global DATA_TYPE *)input_ptr);
        DATA_TYPE
        row01 = *((__global DATA_TYPE *)input_ptr + 4);

        input_ptr += src_stride_y;

        VEC_DATA_TYPE(DATA_TYPE, 4)
        row10 = vload4(0, (__global DATA_TYPE *)input_ptr);
        DATA_TYPE
        row11 = *((__global DATA_TYPE *)input_ptr + 4);

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
        VEC_DATA_TYPE(COND_DATA_TYPE, 4)
        cond00 = x0_condition && (VEC_DATA_TYPE(COND_DATA_TYPE, 4))y0_condition.s0;
        VEC_DATA_TYPE(COND_DATA_TYPE, 4)
        cond10                = x0_condition && (VEC_DATA_TYPE(COND_DATA_TYPE, 4))y0_condition.s1;
        COND_DATA_TYPE cond01 = (COND_DATA_TYPE)(x1_condition && y0_condition.s0);
        COND_DATA_TYPE cond11 = (COND_DATA_TYPE)(x1_condition && y0_condition.s1);

        // Replace with 0 if the value is not valid
        row00 = select((VEC_DATA_TYPE(DATA_TYPE, 4))PAD_VALUE, row00, cond00);
        row10 = select((VEC_DATA_TYPE(DATA_TYPE, 4))PAD_VALUE, row10, cond10);
        row01 = select((DATA_TYPE)PAD_VALUE, row01, cond01);
        row11 = select((DATA_TYPE)PAD_VALUE, row11, cond11);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s0123, row01,
                                              row10.s012),
                0, (__global DATA_TYPE *)output_ptr);
        vstore2((VEC_DATA_TYPE(DATA_TYPE, 2))(row10.s3, row11), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 10 * dst_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 4)
        row00 = vload4(0, (__global DATA_TYPE *)input_ptr);
        DATA_TYPE
        row01 = *((__global DATA_TYPE *)input_ptr + 4);

        input_ptr += src_stride_y;

        VEC_DATA_TYPE(DATA_TYPE, 4)
        row10 = vload4(0, (__global DATA_TYPE *)input_ptr);
        DATA_TYPE
        row11 = *((__global DATA_TYPE *)input_ptr + 4);

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
        VEC_DATA_TYPE(COND_DATA_TYPE, 4)
        cond00 = x0_condition && (VEC_DATA_TYPE(COND_DATA_TYPE, 4))y0_condition.s2;
        VEC_DATA_TYPE(COND_DATA_TYPE, 4)
        cond10                = x0_condition && (VEC_DATA_TYPE(COND_DATA_TYPE, 4))y0_condition.s3;
        COND_DATA_TYPE cond01 = (COND_DATA_TYPE)(x1_condition && y0_condition.s2);
        COND_DATA_TYPE cond11 = (COND_DATA_TYPE)(x1_condition && y0_condition.s3);

        // Replace with 0 if the value is not valid
        row00 = select((VEC_DATA_TYPE(DATA_TYPE, 4))PAD_VALUE, row00, cond00);
        row10 = select((VEC_DATA_TYPE(DATA_TYPE, 4))PAD_VALUE, row10, cond10);
        row01 = select((DATA_TYPE)PAD_VALUE, row01, cond01);
        row11 = select((DATA_TYPE)PAD_VALUE, row11, cond11);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s0123, row01,
                                              row10.s012),
                0, (__global DATA_TYPE *)output_ptr);
        vstore2((VEC_DATA_TYPE(DATA_TYPE, 2))(row10.s3, row11), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 10 * dst_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 4)
        row00 = vload4(0, (__global DATA_TYPE *)input_ptr);
        DATA_TYPE
        row01 = *((__global DATA_TYPE *)input_ptr + 4);

        input_ptr += src_stride_y;

#if PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0
        VEC_DATA_TYPE(COND_DATA_TYPE, 4)
        cond00                = x0_condition && (VEC_DATA_TYPE(COND_DATA_TYPE, 4))y1_condition;
        COND_DATA_TYPE cond01 = (COND_DATA_TYPE)(x1_condition && y1_condition);

        // Replace with 0 if the value is not valid
        row00 = select((VEC_DATA_TYPE(DATA_TYPE, 4))PAD_VALUE, row00, cond00);
        row01 = select((DATA_TYPE)PAD_VALUE, row01, cond01);
#endif // PAD_LEFT != 0 || PAD_TOP != 0 || PAD_RIGHT != 0 || PAD_BOTTOM != 0

        vstore4(row00, 0, (__global DATA_TYPE *)output_ptr);
        *((__global DATA_TYPE *)output_ptr + 4) = row01;

        output_ptr += 5 * dst_stride_x;
    }

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)output_ptr) = 1.0f;
    }
#endif // HAS_BIAS
}
#endif // defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE)

#if defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_DEPTH)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 11x11
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel depth must be passed at compile time using -DKERNEL_DEPTH: e.g. -DKERNEL_DEPTH=3
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col11x11_padx0_pady0_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);                // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const int batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X;
    const int yi = yc * STRIDE_Y;

    // Calculate output indices
    const int xo = ch * 121;                  // 11x11
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + xi * src_stride_x + yi * src_stride_y + ch * src_stride_z + batch * src_stride_w;

    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + xo * dst_stride_x + yo * dst_stride_y + batch * dst_stride_w;
    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        input_ptr += src_stride_y;
        output_ptr += 11 * src_stride_x;
    }

    {
        VEC_DATA_TYPE(DATA_TYPE, 8)
        row00 = vload8(0, (__global DATA_TYPE *)(input_ptr));
        VEC_DATA_TYPE(DATA_TYPE, 3)
        row01 = vload3(0, (__global DATA_TYPE *)(input_ptr) + 8);

        vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row00.s01234567), 0, (__global DATA_TYPE *)output_ptr);
        vstore3((VEC_DATA_TYPE(DATA_TYPE, 3))(row01.s012), 0, (__global DATA_TYPE *)output_ptr + 8);

        output_ptr += 11 * src_stride_x;
    }

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *((__global DATA_TYPE *)output_ptr) = 1.0f;
    }
#endif // HAS_BIAS
}
#endif // defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_DEPTH)

#if defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(VECTOR_SIZE) && defined(WIDTH_MOD_VECTOR_SIZE)
/** This kernel reshapes the input tensor to a tensor used to perform convolution using GEMM when
 * the kernel width is greater than 1 (except when the kernel size is 3x3) and pad_x == pad_y == 0.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float.
 * @note The vector size must be passed at compile time using -DVECTOR_SIZE e.g. -DVECTOR_SIZE=4.
 * @note The width modulo vector size must be passed at compile time using -DWIDTH_MOD_VECTOR_SIZE e.g. -DWIDTH_MOD_VECTOR_SIZE=3.
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col_generic_padx0_pady0_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);                // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const int batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X;
    const int yi = yc * STRIDE_Y;
    // Calculate output indices
    const int xo                   = ch * KERNEL_WIDTH * KERNEL_HEIGHT;
    const int yo                   = xc + yc * CONVOLVED_WIDTH; // Index of the convolution
    __global uchar *input_ptr      = src_ptr + src_offset_first_element_in_bytes + ch * src_stride_z + batch * src_stride_w;
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + yo * dst_stride_y + batch * dst_stride_w)) + xo;
    // Linearize convolution elements
    for(int y = yi, y_e = yi + KERNEL_HEIGHT; y < y_e; ++y)
    {
        int last_x = 0;
        for(int x = xi, x_e = xi + KERNEL_WIDTH; x + VECTOR_SIZE <= x_e; x += VECTOR_SIZE, output_ptr += VECTOR_SIZE)
        {
            VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
            row = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
            VSTORE(VECTOR_SIZE)
            (row, 0, output_ptr);
            last_x = x;
        }
        // Copy the remainder of the row by doing VLOAD(WIDTH_MOD_VECTOR_SIZE) and VSTORE(WIDTH_MOD_VECTOR_SIZE).
        // Note that x and output_ptr have already been incremented by VECTOR_SIZE by the loop just before exit.
#if WIDTH_MOD_VECTOR_SIZE == 1
        *output_ptr = *((__global DATA_TYPE *)(input_ptr + (last_x + VECTOR_SIZE) * src_stride_x + y * src_stride_y));
#elif WIDTH_MOD_VECTOR_SIZE > 1
        VEC_DATA_TYPE(DATA_TYPE, WIDTH_MOD_VECTOR_SIZE)
        row = VLOAD(WIDTH_MOD_VECTOR_SIZE)(0, (__global DATA_TYPE *)(input_ptr + (last_x + VECTOR_SIZE) * src_stride_x + y * src_stride_y));
        VSTORE(WIDTH_MOD_VECTOR_SIZE)
        (row, 0, output_ptr);
#endif /* WIDTH_MOD_VECTOR_SIZE */
        output_ptr += WIDTH_MOD_VECTOR_SIZE;
    } /* End of loop over KERNEL_HEIGHT */

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *output_ptr = 1.0f;
    }
#endif // HAS_BIAS
}
#endif //defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(VECTOR_SIZE) && defined(WIDTH_MOD_VECTOR_SIZE)

#if defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width and height of the input tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT: e.g. -DSRC_WIDTH=128 and -DSRC_HEIGHT=128
 * @note The width of output tensor after matrix multiplication must be passed at compile time using -DCONVOLVED_WIDTH: e.g. -DCONVOLVED_WIDTH=34
 * @note The kernel width, height and depth must be passed at compile time using -DKERNEL_WIDTH, -DKERNEL_HEIGHT and -DKERNEL_DEPTH: e.g. -DKERNEL_WIDTH=3, -DKERNEL_HEIGHT=3 and -DKERNEL_DEPTH=64
 * @note The pad_left, pad_right, pad_top and pad_bottom must be passed at compile time using -DPAD_LEFT, -DPAD_RIGHT, -DPAD_TOP and -DPAD_BOTTOM: e.g. -DPAD_LEFT=1, -DPAD_RIGHT=2, -DPAD_TOP=3 and -DPAD_BOTTOM=2
 * @note The zero value to store in case we load values out-of-bounds must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0.0
 * @note The stride along the X and Y directions must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y: e.g. -DSTRIDE_X=1 and -DSTRIDE_Y=1
 * @note The dilation_x and dilation_y must be passed at compile time using -DDILATION_X and -DDILATION_Y: e.g. -DDILATION_X=1, -DDILATION_Y=1
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
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
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
__kernel void im2col_generic_dchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int xc    = get_global_id(0);                // x coordinate in the convolved tensor
    const int yc    = get_global_id(1);                // y coordinate in the convolved tensor
    const int ch    = get_global_id(2) % KERNEL_DEPTH; // input feature map
    const int batch = get_global_id(2) / KERNEL_DEPTH; // batch size

    // Calculate input indices
    const int xi = xc * STRIDE_X - PAD_LEFT;
    const int yi = yc * STRIDE_Y - PAD_TOP;

    // Calculate output indices
    const int xo = ch * KERNEL_WIDTH * KERNEL_HEIGHT;
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    __global uchar *input_ptr      = src_ptr + src_offset_first_element_in_bytes + ch * src_stride_z + batch * src_stride_w;
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + yo * dst_stride_y + batch * dst_stride_w)) + xo;

    // Linearize convolution elements
    for(int yk = 0; yk < KERNEL_HEIGHT; ++yk)
    {
        int y = yi + yk * DILATION_Y;
        for(int xk = 0; xk < KERNEL_WIDTH; ++xk, ++output_ptr)
        {
            int x = xi + xk * DILATION_X;
#if PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
            *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
#else  // PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                *output_ptr = PAD_VALUE;
            }
            else
            {
                *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
            }
#endif // PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
        }
    }

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
        *output_ptr = 1.0f;
    }
#endif // HAS_BIAS
}
#endif // defined(CONVOLVED_WIDTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(PAD_LEFT) && defined(PAD_RIGHT) && defined(PAD_TOP) && defined(PAD_BOTTOM) && defined(PAD_VALUE)

/**This kernel reshapes the input tensor to a tensor used to perform convolution using GEMM when
 * the kernel width and height are the same of width and height of the input tensor
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note In case biases will be added in late stage, -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             The width of the input tensor
 * @param[in]  height                            The height of the input tensor
 */
__kernel void im2col_reduced_dchw(
    TENSOR3D_DECLARATION(src),
    VECTOR_DECLARATION(dst),
    uint width, uint height)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const uint image_size = width * height;

    __global uchar *tmp_out_ptr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) + get_global_id(1) * width + get_global_id(2) * image_size) * dst_stride_x;

    *((__global DATA_TYPE *)tmp_out_ptr) = *((__global DATA_TYPE *)src.ptr);

#ifdef HAS_BIAS
    // If it is the last thread in the 3 dimensional workgroup
    if(get_global_id(0) == (get_global_size(0) - 1) && get_global_id(1) == (get_global_size(1) - 1) && get_global_id(2) == (get_global_size(2) - 1))
    {
        tmp_out_ptr += dst_stride_x;
        *((__global DATA_TYPE *)tmp_out_ptr) = (DATA_TYPE)1.0f;
    }
#endif // HAS_BIAS
}
#endif // defined(DATA_TYPE) && defined(ELEMENT_SIZE)
