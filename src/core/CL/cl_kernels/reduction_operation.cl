/*
 * Copyright (c) 2016-2018 ARM Limited.
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

/** Calculate square sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return square sum of vector.
 */
inline DATA_TYPE square_sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in *= in;

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** Calculate sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return sum of vector.
 */
inline DATA_TYPE sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** This kernel performs parallel reduction given an operation on x-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The operation we want to perform must be passed at compile time using -DOPERATION e.g. -DOPERATION=square_sum
 * @note The mean flag must be passed at compile time using -DMEAN if we want to compute the mean value
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128 if we want to compute the mean value
 *
 * @param[in] src_ptr                                   Pointer to the source tensor. Supported data types: F16/F32
 * @param[in] src_stride_x                              Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                              Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes         The offset of the first element in the source tensor
 * @param[in] partial_sum_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] partial_sum_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] partial_sum_step_x                        partial_sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] partial_sum_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] partial_sum_step_y                        partial_sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] partial_sum_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] local_sums                                Local buffer for storing the partial sum
 */
__kernel void reduction_operation_x(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(partial_sum),
    __local DATA_TYPE *local_sums)
{
    Image src         = CONVERT_TO_IMAGE_STRUCT(src);
    Image partial_sum = CONVERT_TO_IMAGE_STRUCT(partial_sum);

    unsigned int lsize = get_local_size(0);
    unsigned int lid   = get_local_id(0);

    for(unsigned int y = 0; y < get_local_size(1); ++y)
    {
        local_sums[lid] = OPERATION((__global DATA_TYPE *)offset(&src, 0, y));
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform parallel reduction
        for(unsigned int i = lsize >> 1; i > 0; i >>= 1)
        {
            if(lid < i)
            {
                local_sums[lid] += local_sums[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lid == 0)
        {
#if defined(MEAN) && defined(WIDTH)
            if(y == get_local_size(1) - 1)
            {
                local_sums[0] /= WIDTH;
            }
#endif /* defined(MEAN) && defined(WIDTH) */
            ((__global DATA_TYPE *)offset(&partial_sum, get_group_id(0), y))[0] = local_sums[0];
        }
    }
}

#if defined(WIDTH)
/** This kernel performs reduction on x-axis. (QASYMM8)
 *
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128
 *
 * @param[in] src_ptr                              Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_quantized_x(
    VECTOR_DECLARATION(src),
    VECTOR_DECLARATION(output))
{
    Vector src    = CONVERT_TO_VECTOR_STRUCT(src);
    Vector output = CONVERT_TO_VECTOR_STRUCT(output);

    uint res = 0;

    for(unsigned int x = 0; x < WIDTH; ++x)
    {
        res += *((__global uchar *)vector_offset(&src, x));
    }

#if defined(MEAN)
    res /= WIDTH;
#endif /* defined(MEAN) */

    // Store result
    *((__global uchar *)output.ptr) = convert_uchar(res);
}
#endif /* defined(HEIGHT) */

#if defined(HEIGHT)
/** This kernel performs reduction on y-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] src_ptr                              Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_y(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(output))
{
    Image src    = CONVERT_TO_IMAGE_STRUCT(src);
    Image output = CONVERT_TO_IMAGE_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = 0;

    for(unsigned int y = 0; y < HEIGHT; ++y)
    {
        res += CONVERT(vload16(0, (__global DATA_TYPE *)offset(&src, 0, y)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));
    }

#if defined(MEAN)
    res /= HEIGHT;
#endif /* defined(MEAN) */

    // Store result
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
}
#endif /* defined(HEIGHT) */

#if defined(DEPTH)
/** This kernel performs reduction on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The depth size must be passed at compile time using -DDEPTH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_stride_z                      Stride of the output tensor in Z dimension (in bytes)
 * @param[in] output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_z(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = 0;

    for(unsigned int z = 0; z < DEPTH; ++z)
    {
        res += CONVERT(vload16(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, z)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));
    }

#if defined(MEAN)
    res /= DEPTH;
#endif /* defined(MEAN) */

    // Store result
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
}
#endif /* defined(DEPTH) */

#if defined(BATCH) && defined(DEPTH)
/** This kernel performs reduction on w-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The batch size must be passed at compile time using -DBATCH e.g. -DBATCH=128
 * @note The depth size must be passed at compile time using -DBATCH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in] input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_stride_z                      Stride of the output tensor in Z dimension (in bytes)
 * @param[in] output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] output_stride_w                      Stride of the output tensor in W dimension (in bytes)
 * @param[in] output_step_w                        output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_w(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT(input, DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = 0;

    for(unsigned int w = 0; w < BATCH; ++w)
    {
        res += CONVERT(vload16(0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, 0, 0, w)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));
    }

#if defined(MEAN)
    res /= BATCH;
#endif /* defined(MEAN) */

    // Store result
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
}
#endif /* defined(BATCH) && defined(DEPTH) */