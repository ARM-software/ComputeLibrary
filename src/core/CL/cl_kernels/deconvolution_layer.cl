/*
 * Copyright (c) 2017-2019 ARM Limited.
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

/** This function applies upsample on an input image.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: QASYMM8/F16/F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void deconvolution_upsample(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    // Store result
    *((__global DATA_TYPE *)dst.ptr) = *((__global DATA_TYPE *)src.ptr);
}

#if defined(FILTER_WIDTH) && defined(FILTER_HEIGHT) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DATA_TYPE)
/** This kernel reshapes the deconvolution output tensor before returning the result of the Deconvolution. The decovnolution output tensor
 * is the result of a @ref CLGEMM operation between the deconvolution input and the deconvolution filter
 *
 * @note Data type should be given as a preprocessor argument using -DDATA_TYPE=type, e.g., -DDATA_TYPE=F32
 * @note The width of the filter should be given as a preprocessor argument using -DFILTER_WIDTH=width, e.g., -DFILTER_WIDTH=2
 * @note The height of the filter should be given as a preprocessor argument using -DFILTER_HEIGHT=height, e.g., -DFILTER_HEIGHT=2
 * @note The width of the input should be given as a preprocessor argument using -DSRC_WIDTH=width, e.g., -DSRC_WIDTH=10
 * @note The height of the input should be given as a preprocessor argument using -DSRC_HEIGHT=width, e.g., -DSRC_HEIGHT=10
 * @note The output data layout is NHWC if the preprocessor argument NUM_FILTERS is defined, NCHW if NUM_FILTERS is not defined
 *
 * @param[in]  src_ptr                            Pointer to the source image. Supported data types: QASYMM8/F16/F32
 * @param[in]  src_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] dst_ptr                            Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination image
 * @param[in]  bias_ptr                           (Optional) Pointer to the biases vector. Supported data types: F16/F32/S32
 * @param[in]  bias_stride_x                      (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the biases vector
 */
__kernel void deconvolution_reshape(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst)
#if defined(ADD_BIAS)
    ,
    VECTOR_DECLARATION(bias)
#endif // defined(ADD_BIAS)
)
{
#define FILTER_AREA ((FILTER_WIDTH) * (FILTER_HEIGHT))

    Tensor3D        src  = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D        dst  = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(dst);
    const DATA_TYPE data = *(__global DATA_TYPE *)src.ptr;

    // Store result
    const int x_in = get_global_id(0);
    const int y_in = get_global_id(1);
    const int z_in = get_global_id(2);

#if defined(NUM_FILTERS)
    const int bias_index = x_in / (FILTER_AREA);
    const int z_out      = bias_index + (NUM_FILTERS) * (z_in / (SRC_HEIGHT));
    const int x_out      = x_in % (FILTER_WIDTH) + y_in * (FILTER_WIDTH);
    const int y_out      = (FILTER_HEIGHT) * (z_in % (SRC_HEIGHT)) + ((x_in % (FILTER_AREA)) / (FILTER_WIDTH));
#else  // defined(NUM_FILTERS)
    const int x_out      = x_in / (FILTER_AREA);
    const int y_out      = x_in % (FILTER_WIDTH) + y_in * (FILTER_WIDTH);
    const int z_out      = (FILTER_HEIGHT) * z_in + ((x_in % (FILTER_AREA)) / (FILTER_WIDTH));
    const int bias_index = x_out;
#endif // defined(NUM_FILTERS)

#if defined(ADD_BIAS)
    Vector          bias     = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);
    const DATA_TYPE bias_val = *(__global DATA_TYPE *)vector_offset(&bias, bias_index);
    *((__global DATA_TYPE *)tensor3D_offset(&dst, x_out, y_out, z_out)) = data + bias_val;
#else  // defined(ADD_BIAS)
    *((__global DATA_TYPE *)tensor3D_offset(&dst, x_out, y_out, z_out)) = data;
#endif // defined(ADD_BIAS)

#undef FILTER_AREA
}
#endif // defined(FILTER_WIDTH) && defined(FILTER_HEIGHT) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DATA_TYPE)
