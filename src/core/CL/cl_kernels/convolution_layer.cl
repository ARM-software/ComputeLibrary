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

#if defined(FIXED_POINT_POSITION)
#include "fixed_point.h"
#endif // FIXED_POINT_POSITION

/** This kernel reshapes the tensor's low three dimensions to single column
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  src_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  bias_ptr                           Pointer to the bias tensor. Same as @p src_ptr
 * @param[in]  bias_stride_x                      Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bias_step_x                        bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  width                              The width of the input tensor
 * @param[in]  height                             The height of the input tensor
 * @param[in]  depth                              The depth of the input tensor
 * @param[in]  total_filters                      Total number of filters. 4th dimension of the weights matrix
 */
__kernel void reshape_to_columns(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(bias),
#endif /* HAS_BIAS */
    uint width, uint height, uint depth, uint total_filters)
{
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT(src);
    bool     is_last_thread = (get_global_id(0) == (get_global_size(0) - 1) && get_global_id(1) == (get_global_size(1) - 1) && get_global_id(2) == (get_global_size(2) - 1));

    __global uchar *tmp_src_ptr = src.ptr;
    __global uchar *tmp_dst_ptr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(0) * dst_stride_y + get_global_id(1) * width * dst_stride_y + get_global_id(
                                      2) * width * height * dst_stride_y;
#ifdef HAS_BIAS
    __global uchar *tmp_bias_ptr = bias_ptr + bias_offset_first_element_in_bytes;
#endif /* HAS_BIAS */

    if(is_last_thread)
    {
        for(uint i = 0; i < total_filters; ++i)
        {
            *((__global DATA_TYPE *)tmp_dst_ptr) = *((__global DATA_TYPE *)tmp_src_ptr);

#ifdef HAS_BIAS
            *((__global DATA_TYPE *)(tmp_dst_ptr + dst_stride_y)) = *((__global DATA_TYPE *)(tmp_bias_ptr));
            tmp_bias_ptr += bias_stride_x;
#endif /* HAS_BIAS */
            tmp_src_ptr += depth * src_stride_z;
            tmp_dst_ptr += dst_stride_x;
        }
    }
    else
    {
        for(uint i = 0; i < total_filters; ++i)
        {
            *((__global DATA_TYPE *)tmp_dst_ptr) = *((__global DATA_TYPE *)tmp_src_ptr);
            tmp_src_ptr += depth * src_stride_z;
            tmp_dst_ptr += dst_stride_x;
        }
    }
}

#if defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(PAD_VALUE)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The value to use for the paddings must be passed at compile time using -DPAD_VALUE: e.g. -DPAD_VALUE=0
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QASYMM8/QS16/F16/F32
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
__kernel void im2col_generic(
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
    for(int y = yi, y_e = yi + KERNEL_HEIGHT; y < y_e; ++y)
    {
        for(int x = xi, x_e = xi + KERNEL_WIDTH; x < x_e; ++x, ++output_ptr)
        {
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
#ifdef FIXED_POINT_POSITION
        *output_ptr = (DATA_TYPE)(1 << FIXED_POINT_POSITION);
#else  // FIXED_POINT_POSITION
        *output_ptr       = 1.0f;
#endif // FIXED_POINT_POSITION
    }
#endif // HAS_BIAS
}

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM when the kernel size is 3x3 and pad_x = pad_y = 0
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QASYMM8/QS16/F16/F32
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
__kernel void im2col_kernel3x3_padx0_pady0(
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
    const int xo = ch * KERNEL_WIDTH * KERNEL_HEIGHT;
    const int yo = xc + yc * CONVOLVED_WIDTH; // Index of the convolution

    // Get input and output address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + xi * src_stride_x + yi * src_stride_y + ch * src_stride_z + batch * src_stride_w;

    __global DATA_TYPE *output_ptr = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + yo * dst_stride_y + batch * dst_stride_w) + xo;

    VEC_DATA_TYPE(DATA_TYPE, 3)
    row0 = vload3(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row1 = vload3(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    row2 = vload3(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y));

    vstore8((VEC_DATA_TYPE(DATA_TYPE, 8))(row0.s012, row1.s012, row2.s01), 0, output_ptr);
    *(output_ptr + 8) = row2.s2;

#ifdef HAS_BIAS
    if(ch == (KERNEL_DEPTH - 1))
    {
#ifdef FIXED_POINT_POSITION
        *(output_ptr + 9) = (DATA_TYPE)(1 << FIXED_POINT_POSITION);
#else  // FIXED_POINT_POSITION
        *(output_ptr + 9) = 1.0f;
#endif // FIXED_POINT_POSITION
    }
#endif // HAS_BIAS
}
#endif //defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT)

#if defined(WIDTH_OUTPUT)
/** This kernel performs a reshaping of the output of the convolution layer.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QASYMM8/QS16/F16/F32
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
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(dst);

    // Compute output offset
    int idx = get_global_id(0) * dst.stride_z + (get_global_id(1) / WIDTH_OUTPUT) * dst_stride_y + (get_global_id(1) % WIDTH_OUTPUT) * dst_stride_x + get_global_id(2) * dst_stride_w;

    // Store value
    *((__global DATA_TYPE *)(dst.ptr + idx)) = *((__global DATA_TYPE *)(src.ptr));
}
#endif // defined(WIDTH_OUTPUT)

/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note In case biases will be added in late stage, -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QASYMM8/QS16/F16/F32
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
__kernel void im2col_reduced(
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
#ifdef FIXED_POINT_POSITION
        *((__global DATA_TYPE *)tmp_out_ptr) = (DATA_TYPE)(1 << FIXED_POINT_POSITION);
#else  // FIXED_POINT_POSITION
        *((__global DATA_TYPE *)tmp_out_ptr) = (DATA_TYPE)1;
#endif // FIXED_POINT_POSITION
    }
#endif // HAS_BIAS
}

#if defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(VECTOR_SIZE) && defined(WIDTH_MOD_VECTOR_SIZE)
/** This kernel reshapes the input tensor to a tensor used to perform convolution using GEMM when
 * the kernel width is greater than 1 (except when the kernel size is 3x3) and pad_x == pad_y == 0.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float.
 * @note The vector size must be passed at compile time using -DVECTOR_SIZE e.g. -DVECTOR_SIZE=4.
 * @note The width modulo vector size must be passed at compile time using -DWIDTH_MOD_VECTOR_SIZE e.g. -DWIDTH_MOD_VECTOR_SIZE=3.
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QS16/F16/F32
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
__kernel void im2col_generic_padx0_pady0(
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
#ifdef FIXED_POINT_POSITION
        *output_ptr = (DATA_TYPE)(1 << FIXED_POINT_POSITION);
#else  // FIXED_POINT_POSITION
        *output_ptr       = 1.0f;
#endif // FIXED_POINT_POSITION
    }
#endif // HAS_BIAS
}
#endif //defined(CONVOLVED_WIDTH) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(KERNEL_DEPTH) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(VECTOR_SIZE) && defined(WIDTH_MOD_VECTOR_SIZE)
