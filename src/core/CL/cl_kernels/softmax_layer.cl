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

#ifdef FIXED_POINT_POSITION

#include "fixed_point.h"
#define MAX_OP(x, y, type, size) MAX_OP_EXPAND(x, y, type, size)
#define ADD_OP(x, y, type, size) ADD_SAT_OP_EXPAND((x), (y), type, size)
#define SUB_OP(x, y, type, size) SUB_SAT_OP_EXPAND((x), (y), type, size)
#define DIV_OP(x, y, type, size) DIV_SAT_OP_VEC_EXPAND((x), (y), type, size, FIXED_POINT_POSITION)
#define EXP_OP(x, type, size) EXP_OP_EXPAND((x), type, size, FIXED_POINT_POSITION)

#define MIN_VAL_EXPAND(type) type##_MIN
#define MIN_VAL(type) MIN_VAL_EXPAND(type)
#define MINVAL MIN_VAL(DATA_TYPE)
#define SELECT_DATA_TYPE EXPAND(DATA_TYPE)

#else /* FIXED_POINT_POSITION */

#define MAX_OP(x, y, type, size) max((x), (y))
#define ADD_OP(x, y, type, size) ((x) + (y))
#define SUB_OP(x, y, type, size) ((x) - (y))
#define DIV_OP(x, y, type, size) ((x) / (y))
#define EXP_OP(x, type, size) exp((x))

#ifdef USE_F16
#define MINVAL -HALF_MAX
#define SELECT_DATA_TYPE short
#else /* USE_F16 */
#define MINVAL -FLT_MAX
#define SELECT_DATA_TYPE int
#endif /* USE_F16 */

#endif /* FIXED_POINT_POSITION */

__constant VEC_DATA_TYPE(DATA_TYPE, 16) type_min = (VEC_DATA_TYPE(DATA_TYPE, 16))(MINVAL);
__constant uint16 idx16 = (uint16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

/** Identifies the maximum value across the 1st dimension.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 * @note In case the input is not multiple of 16 -DNON_MULTIPLE_OF_16 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             Input image width
 */
__kernel void softmax_layer_max(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint width)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);

    // Initialize local maximum
    VEC_DATA_TYPE(DATA_TYPE, 16)
    max_val = (VEC_DATA_TYPE(DATA_TYPE, 16))type_min;

    // Calculate max of row
    const uint width4 = width >> 4;
    for(uint i = 0; i < width4; i++)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        data    = vload16(0, (__global DATA_TYPE *)offset(&src, i << 4, 0));
        max_val = MAX_OP(data, max_val, DATA_TYPE, 16);
    }

#ifdef NON_MULTIPLE_OF_16
    // Handle non multiple of 16
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)offset(&src, width4 << 4, 0));
    VEC_DATA_TYPE(SELECT_DATA_TYPE, 16)
    widx    = CONVERT(((uint16)(width4 << 4) + idx16) < width, VEC_DATA_TYPE(SELECT_DATA_TYPE, 16));
    max_val = MAX_OP(max_val, select(type_min, data, widx), DATA_TYPE, 16);
#endif /* NON_MULTIPLE_OF_16 */

    // Perform max reduction
    max_val.s01234567 = MAX_OP(max_val.s01234567, max_val.s89ABCDEF, DATA_TYPE, 8);
    max_val.s0123     = MAX_OP(max_val.s0123, max_val.s4567, DATA_TYPE, 4);
    max_val.s01       = MAX_OP(max_val.s01, max_val.s23, DATA_TYPE, 2);
    max_val.s0        = MAX_OP(max_val.s0, max_val.s1, DATA_TYPE, 1);

    // Store result
    *((__global DATA_TYPE *)dst.ptr) = max_val.s0;
}

/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 * @note In case the input is not multiple of 16 -DNON_MULTIPLE_OF_16 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  max_ptr                           Pointer to the max values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  max_stride_x                      Stride of the max values tensor in X dimension (in bytes)
 * @param[in]  max_step_x                        max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  max_stride_y                      Stride of the max values tensor in Y dimension (in bytes)
 * @param[in]  max_step_y                        max_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  max_stride_z                      Stride of the max values tensor in Z dimension (in bytes)
 * @param[in]  max_step_z                        max_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  max_offset_first_element_in_bytes The offset of the first element in the max values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[in]  width                             Input image width
 */
__kernel void softmax_layer_shift_exp_sum(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(max),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum),
    uint width)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image max = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(max);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    // Load max value of 1D logits vector (row)
    DATA_TYPE max_val = *((__global DATA_TYPE *)offset(&max, 0, 0));

    // Set sum vector
    VEC_DATA_TYPE(DATA_TYPE, 16)
    sum1D = 0;

    // Shift values, exp and sum
    const uint width4 = width >> 4;
    for(uint i = 0; i < width4; i++)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        data = vload16(0, (__global DATA_TYPE *)offset(&src, i << 4, 0));
        data = SUB_OP(data, max_val, DATA_TYPE, 16);
        data = EXP_OP(data, DATA_TYPE, 16);
        vstore16(data, 0, (__global DATA_TYPE *)offset(&dst, i << 4, 0));
        sum1D = ADD_OP(sum1D, data, DATA_TYPE, 16);
    }

#ifdef NON_MULTIPLE_OF_16
    // Handle non multiple of 16
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)offset(&src, width4 << 4, 0));
    data = SUB_OP(data, max_val, DATA_TYPE, 16);
    data = EXP_OP(data, DATA_TYPE, 16);
    VEC_DATA_TYPE(SELECT_DATA_TYPE, 16)
    widx = CONVERT(((uint16)(width4 << 4) + idx16) < width, VEC_DATA_TYPE(SELECT_DATA_TYPE, 16));
    data = select(0, data, widx);
    vstore16(data, 0, (__global DATA_TYPE *)offset(&dst, width4 << 4, 0));
    sum1D = ADD_OP(sum1D, data, DATA_TYPE, 16);
#endif /* NON_MULTIPLE_OF_16 */

    // Perform min/max reduction
    sum1D.s01234567 = ADD_OP(sum1D.s01234567, sum1D.s89ABCDEF, DATA_TYPE, 8);
    sum1D.s0123     = ADD_OP(sum1D.s0123, sum1D.s4567, DATA_TYPE, 4);
    sum1D.s01       = ADD_OP(sum1D.s01, sum1D.s23, DATA_TYPE, 2);
    sum1D.s0        = ADD_OP(sum1D.s0, sum1D.s1, DATA_TYPE, 1);

    // Calculate and store result
    *((__global DATA_TYPE *)sum.ptr) = sum1D.s0;
}

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void softmax_layer_norm(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(dst))
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    DATA_TYPE sum_val = *((__global DATA_TYPE *)offset(&sum, 0, get_global_id(1)));
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)offset(&src, 0, 0));
    vstore16(DIV_OP(data, sum_val, DATA_TYPE, 16), 0, (__global DATA_TYPE *)offset(&dst, 0, 0));
}
