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

/** This kernel reshapes the tensor's low three dimensions to single column
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  src_ptr                            Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  src_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Same as input
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  bias_ptr                           Pointer to the bias tensor. Same as input
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
#if defined HAS_BIAS
    VECTOR_DECLARATION(bias),
#endif
    uint width, uint height, uint depth, uint total_filters)
{
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT(src);
    bool     is_last_thread = (get_global_id(0) == (get_global_size(0) - 1) && get_global_id(1) == (get_global_size(1) - 1) && get_global_id(2) == (get_global_size(2) - 1));

    __global uchar *tmp_src_ptr = src.ptr;
    __global uchar *tmp_dst_ptr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(0) * dst_stride_y + get_global_id(1) * width * dst_stride_y + get_global_id(
                                      2) * width * height * dst_stride_y;
#if defined         HAS_BIAS
    __global uchar *tmp_bias_ptr = bias_ptr + bias_offset_first_element_in_bytes;
#endif

    if(is_last_thread)
    {
        for(uint i = 0; i < total_filters; ++i)
        {
            *((__global DATA_TYPE *)tmp_dst_ptr) = *((__global DATA_TYPE *)tmp_src_ptr);

#if defined HAS_BIAS
            *((__global DATA_TYPE *)(tmp_dst_ptr + dst_stride_y)) = *((__global DATA_TYPE *)(tmp_bias_ptr));
            tmp_bias_ptr += bias_stride_x;
#endif
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

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: F16, F32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  kernel_size                       The convolution kernel size
 * @param[in]  kernel_depth                      The kernel depth
 * @param[in]  width                             The output tensor width
 * @param[in]  input_dims                        The input tensor dimensions
 * @param[in]  strides                           The strides of the im2col operation
 * @param[in]  paddings                          The input tensor paddings
 */
__kernel void im2col_generic(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    int  kernel_size,
    int  kernel_depth,
    int  width,
    int2 input_dims,
    int2 strides,
    int2 paddings)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT_NO_STEP(dst);

    // Determine output index
    uint     idx               = (get_global_id(1) * width + get_global_id(0)) * dst.stride_y;
    __global uchar *output_ptr = dst.ptr + idx;

    // Determine current input index
    const int top_left_x = get_global_id(0) * strides.x - paddings.x;
    const int top_left_y = get_global_id(1) * strides.y - paddings.y;

    // Linearize convolution elements
    for(int d = 0; d < kernel_depth; ++d)
    {
        for(int y = top_left_y, y_e = top_left_y + kernel_size; y < y_e; ++y)
        {
            for(int x = top_left_x, x_e = top_left_x + kernel_size; x < x_e; ++x, output_ptr += dst.stride_x)
            {
                if(x < 0 || x >= input_dims.x || y < 0 || y >= input_dims.y)
                {
                    *((__global DATA_TYPE *)output_ptr) = 0;
                }
                else
                {
                    *((__global DATA_TYPE *)output_ptr) = *((__global DATA_TYPE *)(tensor3D_offset(&src, x, y, d)));
                }
            }
        }
    }

#if defined HAS_BIAS
    *((__global DATA_TYPE *)output_ptr) = 1;
#endif
}

/** This kernel performs a reshaping of the output of the convolution layer.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: F16, F32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             The output tensor width
 */
__kernel void col2im(
    IMAGE_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint width)
{
    Image    src = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(dst);

    int      idx                         = get_global_id(0) * dst.stride_z + (get_global_id(1) / width) * dst.stride_y + (get_global_id(1) % width) * dst.stride_x;
    __global uchar *tmp_out_ptr          = dst.ptr + idx;
    *((__global DATA_TYPE *)tmp_out_ptr) = *((__global DATA_TYPE *)(src.ptr));
}

/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note In case biases will be added in late stage, -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as input.
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

#if defined HAS_BIAS
    // If it is the last thread in the 3 dimensional workgroup
    if(get_global_id(0) == (get_global_size(0) - 1) && get_global_id(1) == (get_global_size(1) - 1) && get_global_id(2) == (get_global_size(2) - 1))
    {
        tmp_out_ptr += dst_stride_x;
        *((__global DATA_TYPE *)tmp_out_ptr) = (DATA_TYPE)1;
    }
#endif
}
