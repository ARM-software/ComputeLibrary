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

#ifdef POOL_AVG
#define POOL_OP(x, y) ((x) + (y))
#else /* POOL_AVG */
#define POOL_OP(x, y) (fmax((x), (y)))
#endif /* POOL_AVG */

float calculate_avg_scale(const int pool_size, const int upper_bound_w, const int upper_bound_h,
                          const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = get_global_id(0) * stride_x - pad_x;
    int start_y = get_global_id(1) * stride_y - pad_y;
    int end_x   = min(start_x + pool_size, upper_bound_w);
    int end_y   = min(start_y + pool_size, upper_bound_h);
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to 2.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16, F32;
 * @note In case of average pooling -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: F16, F32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  max_dims                             The maximum index that can be accessed in x and y dimension (width + pad)
 * @param[in]  strides                              The pooling operation strides in each dimension
 * @param[in]  paddings                             The pooling operation paddings in each dimension
 */
__kernel void pooling_layer_2(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output)
#ifdef POOL_AVG
    ,
    int2 max_dims, int2 strides, int2 paddings
#endif /* POOL_AVG */
)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));

    // Perform calculations
    data0         = POOL_OP(data0, data1);
    DATA_TYPE res = POOL_OP(data0.s0, data0.s1);

    // Divide by pool region in case of average pooling
#ifdef POOL_AVG
    res *= calculate_avg_scale(2, max_dims.x, max_dims.y, paddings.x, paddings.y, strides.x, strides.y);
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}

/** Performs a pooling function of pool size equal to 3.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16, F32;
 * @note In case of average pooling -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: F16, F32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  max_dims                             The maximum index that can be accessed in x and y dimension (width + pad)
 * @param[in]  strides                              The pooling operation strides in each dimension
 * @param[in]  paddings                             The pooling operation paddings in each dimension
 */
__kernel void pooling_layer_3(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output)
#ifdef POOL_AVG
    ,
    int2 max_dims, int2 strides, int2 paddings
#endif /* POOL_AVG */
)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 3)
    data0 = vload3(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    data1 = vload3(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 3)
    data2 = vload3(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));

    // Perform calculations
    data0         = POOL_OP(data0, data1);
    data0         = POOL_OP(data0, data2);
    DATA_TYPE res = POOL_OP(POOL_OP(data0.s0, data0.s1), data0.s2);

    // Divide by pool region in case of average pooling
#ifdef POOL_AVG
    res *= calculate_avg_scale(3, max_dims.x, max_dims.y, paddings.x, paddings.y, strides.x, strides.y);
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}

/** Performs a pooling function of pool size equal to 7.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16, F32;
 * @note In case of average pooling -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: F16, F32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  max_dims                             The maximum index that can be accessed in x and y dimension (width + pad)
 * @param[in]  strides                              The pooling operation strides in each dimension
 * @param[in]  paddings                             The pooling operation paddings in each dimension
 */
__kernel void pooling_layer_7(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output)
#ifdef POOL_AVG
    ,
    int2 max_dims, int2 strides, int2 paddings
#endif /* POOL_AVG */
)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data0 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data1 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data2 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data3 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data4 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data5 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 5, 0));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data6 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 6, 0));

    // Pool operation of all rows
    data0 = POOL_OP(data0, data1);
    data2 = POOL_OP(data2, data3);
    data4 = POOL_OP(data4, data5);
    data0 = POOL_OP(data0, data2);
    data4 = POOL_OP(data4, data6);
    data0 = POOL_OP(data0, data4);

    // Set last element
#ifdef POOL_AVG
    data0.s7 = 0;
#else  /* POOL_AVG */
    data0.s7 = data0.s6;
#endif /* POOL_AVG */

    // Reduce result
    VEC_DATA_TYPE(DATA_TYPE, 4)
    reduce4 = POOL_OP(data0.s0123, data0.s4567);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    reduce2       = POOL_OP(reduce4.s01, reduce4.s23);
    DATA_TYPE res = POOL_OP(reduce2.s0, reduce2.s1);

    // Divide by pool region in case of average pooling
#ifdef POOL_AVG
    res *= calculate_avg_scale(7, max_dims.x, max_dims.y, paddings.x, paddings.y, strides.x, strides.y);
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}
