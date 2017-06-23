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

#if defined(POOL_AVG)
#define POOL_OP(x, y) add_sat(x, y)
#else /* POOL_AVG */
#define POOL_OP(x, y) (max((x), (y)))
#endif /* POOL_AVG */

#define DIV_OP1(x, y) DIV_SAT_OP_EXPAND((x), y, DATA_TYPE, FIXED_POINT_POSITION)
#define DIV_OP(x, y) DIV_OP1(x, y << FIXED_POINT_POSITION)

#else /* FIXED_POINT_POSITION */

#if defined(POOL_AVG)
#define POOL_OP(x, y) ((x) + (y))
#else /* POOL_AVG */
#define POOL_OP(x, y) (fmax((x), (y)))
#endif /* POOL_AVG */

#define DIV_OP(x, y) (x * (1.f / y))

#endif /* FIXED_POINT_POSITION */

#if STRIDE_X == 1
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE1(res, input, output)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE2(res, input, output)
#elif STRIDE_X == 3 /* STRIDE_X not equals 1 or 2 */
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE3(res, input, output)
#endif /* STRIDE_X == 3 */

#define POOLING3x3_STRIDE1(res, input, output)                                                                                               \
    ({                                                                                                                                       \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        data00 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                                                                                          \
        data01 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 4);                                                      \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        data10 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                                                                                          \
        data11 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 4);                                                      \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        data20 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                                                                                          \
        data21 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 4);                                                      \
        \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values00 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data00.s01212323);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values01 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data01.s0, data00.s3, data01.s01);                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values10 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data10.s01212323);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values11 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data11.s0, data10.s3, data11.s01);                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values20 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data20.s01212323);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values21 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data21.s0, data20.s3, data21.s01);                                                          \
        \
        values00 = POOL_OP(values00, values10);                                                                                              \
        values01 = POOL_OP(values01, values11);                                                                                              \
        values00 = POOL_OP(values00, values20);                                                                                              \
        values01 = POOL_OP(values01, values21);                                                                                              \
        \
        res = POOL_OP((VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s036, values01.s1), (VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s147, values01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s25, values01.s03));                                                       \
    })

#define POOLING3x3_STRIDE2(res, input, output)                                                                                               \
    ({                                                                                                                                       \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        data00           = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                                                \
        DATA_TYPE data01 = *((__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 8);                                                    \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        data10           = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                                                \
        DATA_TYPE data11 = *((__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 8);                                                    \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        data20           = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                                                \
        DATA_TYPE data21 = *((__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 8);                                                    \
        \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values00 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data00.s01223445);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values01 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data00.s667, data01);                                                                       \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values10 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data10.s01223445);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values11 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data10.s667, data11);                                                                       \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                          \
        values20 = (VEC_DATA_TYPE(DATA_TYPE, 8))(data20.s01223445);                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                          \
        values21 = (VEC_DATA_TYPE(DATA_TYPE, 4))(data20.s667, data21);                                                                       \
        \
        values00 = POOL_OP(values00, values10);                                                                                              \
        values01 = POOL_OP(values01, values11);                                                                                              \
        values00 = POOL_OP(values00, values20);                                                                                              \
        values01 = POOL_OP(values01, values21);                                                                                              \
        \
        res = POOL_OP((VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s036, values01.s1), (VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s147, values01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(DATA_TYPE, 4))(values00.s25, values01.s03));                                                       \
    })

#define POOLING3x3_STRIDE3(res, input, output)                                                                                       \
    ({                                                                                                                               \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                  \
        data00 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                                                  \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                  \
        data01 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 8);                                              \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                  \
        data10 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                                                  \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                  \
        data11 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 8);                                              \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                  \
        data20 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                                                  \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                                  \
        data21 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 8);                                              \
        \
        data00 = POOL_OP(data00, data10);                                                                                            \
        data01 = POOL_OP(data01, data11);                                                                                            \
        data00 = POOL_OP(data00, data20);                                                                                            \
        data01 = POOL_OP(data01, data21);                                                                                            \
        \
        res = POOL_OP((VEC_DATA_TYPE(DATA_TYPE, 4))(data00.s036, data01.s1), (VEC_DATA_TYPE(DATA_TYPE, 4))(data00.s147, data01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(DATA_TYPE, 4))(data00.s25, data01.s03));                                                   \
    })

DATA_TYPE calculate_avg_scale(const int pool_size, const int upper_bound_w, const int upper_bound_h,
                              const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const int start_x = get_global_id(0) * stride_x - pad_x;
    const int start_y = get_global_id(1) * stride_y - pad_y;
    const int end_x   = min(start_x + pool_size, upper_bound_w);
    const int end_y   = min(start_y + pool_size, upper_bound_h);
    return ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to 2.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QS8/QS16/F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pooling_layer_2(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
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
    res = DIV_OP(res, calculate_avg_scale(2, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}

/** Performs a pooling function of pool size equal to 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QS8/QS16/F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pooling_layer_3(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
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
    res = DIV_OP(res, calculate_avg_scale(3, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}

#if defined(POOLING3x3) && !defined(FIXED_POINT_POSITION)

#define CONVERT_OP(data_type) convert_##data_type##4
#define CONVERT_VECTOR4(data_type) CONVERT_OP(data_type)

VEC_DATA_TYPE(DATA_TYPE, 4)
calculate_avg_scale4(const int pool_size, const int upper_bound_w, const int upper_bound_h,
                     const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const int4 start_x = ((int4)get_global_id(0) * 4 + (int4)(0, 1, 2, 3)) * (int4)stride_x - (int4)pad_x;
    const int  start_y = get_global_id(1) * stride_y - pad_y;
    const int4 end_x   = min(start_x + (int4)pool_size, (int4)upper_bound_w);
    const int  end_y   = min(start_y + pool_size, upper_bound_h);
    return (VEC_DATA_TYPE(DATA_TYPE, 4))(1.f) / CONVERT_VECTOR4(DATA_TYPE)(((int4)(end_y - start_y)) * (end_x - start_x));
}

/** Performs an optimized pooling function of pool size equal to 3 when the stride_x is less equal than 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pooling_layer_3_optimized(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, 4)
    res;

    // Perform pooling 3x3 for 4 output elements
    POOLING3x3(res, input, output);

    // Divide by pool region in case of average pooling
#ifdef POOL_AVG
    res *= calculate_avg_scale4(3, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif // POOL_AVG

    vstore4(res, 0, (__global DATA_TYPE *)output.ptr);
}
#endif // defined(POOLING3x3) && !defined(FIXED_POINT_POSITION)

/** Performs a pooling function of pool size equal to 7.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QS8/QS16/F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pooling_layer_7(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
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
    res = DIV_OP(res, calculate_avg_scale(7, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* POOL_AVG */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}
