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

#if defined(DATA_TYPE) && defined(NUM_GROUPS)
/** This kernel reshapes the tensor's low three dimensions to single column
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note The number of groups should be given as a preprocessor argument using -DNUM_GROUPS=number. e.g. -DNUM_GROUPS=2
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
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 */
__kernel void reshape_to_columns_nchw(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(bias),
#endif /* HAS_BIAS */
    uint width, uint height, uint depth, uint total_filters, uint dst_stride_z)
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
        for(uint g = 0; g < NUM_GROUPS; ++g)
        {
            __global uchar *curr_group_dst = tmp_dst_ptr;

            for(uint i = 0; i < total_filters / NUM_GROUPS; ++i)
            {
                *((__global DATA_TYPE *)curr_group_dst) = *((__global DATA_TYPE *)tmp_src_ptr);

#ifdef HAS_BIAS
                *((__global DATA_TYPE *)(curr_group_dst + dst_stride_y)) = *((__global DATA_TYPE *)(tmp_bias_ptr));
                tmp_bias_ptr += bias_stride_x;
#endif /* HAS_BIAS */
                tmp_src_ptr += depth * src_stride_z;
                curr_group_dst += dst_stride_x;
            }

            tmp_dst_ptr += dst_stride_z;
        }
    }
    else
    {
        for(uint g = 0; g < NUM_GROUPS; ++g)
        {
            __global uchar *curr_group_dst = tmp_dst_ptr;

            for(uint i = 0; i < total_filters / NUM_GROUPS; ++i)
            {
                *((__global DATA_TYPE *)curr_group_dst) = *((__global DATA_TYPE *)tmp_src_ptr);
                tmp_src_ptr += depth * src_stride_z;
                curr_group_dst += dst_stride_x;
            }

            tmp_dst_ptr += dst_stride_z;
        }
    }
}

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
 * @param[in]  depth                              The depth of the input tensor
 * @param[in]  width                              The width of the input tensor
 * @param[in]  height                             The height of the input tensor
 * @param[in]  total_filters                      Total number of filters. 4th dimension of the weights matrix
 */
__kernel void reshape_to_columns_nhwc(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(bias),
#endif /* HAS_BIAS */
    uint depth, uint width, uint height, uint total_filters, uint dst_stride_z)
{
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT(src);
    bool     is_last_thread = (get_global_id(0) == (get_global_size(0) - 1) && get_global_id(1) == (get_global_size(1) - 1) && get_global_id(2) == (get_global_size(2) - 1));

    __global uchar *tmp_src_ptr = src.ptr;
    __global uchar *tmp_dst_ptr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(1) * dst_stride_y + get_global_id(2) * width * dst_stride_y + get_global_id(
                                      0) * width * height * dst_stride_y;
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
            tmp_src_ptr += height * src_stride_z;
            tmp_dst_ptr += dst_stride_x;
        }
    }
    else
    {
        for(uint i = 0; i < total_filters; ++i)
        {
            *((__global DATA_TYPE *)tmp_dst_ptr) = *((__global DATA_TYPE *)tmp_src_ptr);
            tmp_src_ptr += height * src_stride_z;
            tmp_dst_ptr += dst_stride_x;
        }
    }
}
#endif // defined(DATA_TYPE) && defined(NUM_GROUPS)