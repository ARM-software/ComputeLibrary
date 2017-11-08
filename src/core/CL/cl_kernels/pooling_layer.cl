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

#define DIV_OP1(x, y) DIV_SAT_OP_EXPAND((x), (y), DATA_TYPE, FIXED_POINT_POSITION)
#define DIV_OP(x, y) DIV_OP1(x, y << FIXED_POINT_POSITION)
#define SQRT_OP(x) DIV_OP1((1 << FIXED_POINT_POSITION), (INVSQRT_OP_EXPAND((x), DATA_TYPE, 1, FIXED_POINT_POSITION)))

#if defined(POOL_L2)
#define POW2_OP(x, vec_size) MUL_SAT_OP_EXPAND((x), (x), DATA_TYPE, vec_size, FIXED_POINT_POSITION)
#else /* defined(POOL_L2) */
#define POW2_OP(x, vec_size) (x)
#endif /* defined(POOL_L2) */

#else /* FIXED_POINT_POSITION */

#if defined(POOL_AVG) || defined(POOL_L2)
#define POOL_OP(x, y) ((x) + (y))
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define POOL_OP(x, y) (fmax((x), (y)))
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
#define POW2_OP(x, vec_size) ((x) * (x))
#else /* defined(POOL_L2) */
#define POW2_OP(x, vec_size) (x)
#endif /* defined(POOL_L2) */

#define DIV_OP(x, y) (x * (1.f / y))
#define SQRT_OP(x) sqrt((x))

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
        data00 = POW2_OP(data00, 4);                                                                                                         \
        data01 = POW2_OP(data01, 2);                                                                                                         \
        data10 = POW2_OP(data10, 4);                                                                                                         \
        data11 = POW2_OP(data11, 2);                                                                                                         \
        data20 = POW2_OP(data20, 4);                                                                                                         \
        data21 = POW2_OP(data21, 2);                                                                                                         \
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
        data00           = POW2_OP(data00, 8);                                                                                               \
        data01           = POW2_OP(data01, 1);                                                                                               \
        data10           = POW2_OP(data10, 8);                                                                                               \
        data11           = POW2_OP(data11, 1);                                                                                               \
        data20           = POW2_OP(data20, 8);                                                                                               \
        data21           = POW2_OP(data21, 1);                                                                                               \
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
        data00 = POW2_OP(data00, 8);                                                                                                 \
        data01 = POW2_OP(data01, 4);                                                                                                 \
        data10 = POW2_OP(data10, 8);                                                                                                 \
        data11 = POW2_OP(data11, 4);                                                                                                 \
        data20 = POW2_OP(data20, 8);                                                                                                 \
        data21 = POW2_OP(data21, 4);                                                                                                 \
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
    int       start_x = get_global_id(0) * stride_x - pad_x;
    int       start_y = get_global_id(1) * stride_y - pad_y;
    const int end_x   = min(start_x + pool_size, upper_bound_w);
    const int end_y   = min(start_y + pool_size, upper_bound_h);
#if defined(EXCLUDE_PADDING)
    start_x = max(0, start_x);
    start_y = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    return ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to 2.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
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

#if defined(POOL_L2)
    // Raise to power of 2 for L2 Pooling
    data0 = POW2_OP(data0, 2);
    data1 = POW2_OP(data1, 2);
#endif /* defined(POOL_L2) */

    // Perform calculations
    data0         = POOL_OP(data0, data1);
    DATA_TYPE res = POOL_OP(data0.s0, data0.s1);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average or l2 pooling
    res = DIV_OP(res, calculate_avg_scale(2, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}

/** Performs a pooling function of pool size equal to 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
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

#if defined(POOL_L2)
    // Raise to power of 2 for L2 Pooling
    data0 = POW2_OP(data0, 3);
    data1 = POW2_OP(data1, 3);
    data2 = POW2_OP(data2, 3);
#endif /* defined(POOL_L2) */

    // Perform calculations
    data0         = POOL_OP(data0, data1);
    data0         = POOL_OP(data0, data2);
    DATA_TYPE res = POOL_OP(POOL_OP(data0.s0, data0.s1), data0.s2);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    res = DIV_OP(res, calculate_avg_scale(3, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

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
    int4       start_x = ((int4)get_global_id(0) * 4 + (int4)(0, 1, 2, 3)) * (int4)stride_x - (int4)pad_x;
    int        start_y = get_global_id(1) * stride_y - pad_y;
    const int4 end_x   = min(start_x + (int4)pool_size, (int4)upper_bound_w);
    const int  end_y   = min(start_y + pool_size, upper_bound_h);
#if defined(EXCLUDE_PADDING)
    start_x = max((int4)0, start_x);
    start_y = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    return (VEC_DATA_TYPE(DATA_TYPE, 4))(1.f) / CONVERT_VECTOR4(DATA_TYPE)(((int4)(end_y - start_y)) * (end_x - start_x));
}

/** Performs an optimized pooling function of pool size equal to 3 when the stride_x is less equal than 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
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
__kernel void pooling_layer_optimized_3(
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

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    res *= calculate_avg_scale4(3, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    vstore4(res, 0, (__global DATA_TYPE *)output.ptr);
}
#endif // defined(POOLING3x3) && !defined(FIXED_POINT_POSITION)

#if defined(POOL_SIZE)

// Set the initial value for the pooling operation accordingly with the data type
#if defined(POOL_AVG) || defined(POOL_L2)
#define INITIAL_VALUE 0
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#ifdef FIXED_POINT_POSITION
#define MIN_VAL_EXPAND(type) type##_MIN
#define MIN_VAL(type) MIN_VAL_EXPAND(type)
#define INITIAL_VALUE MIN_VAL(DATA_TYPE)
#else // FIXED_POINT_POSITION
#if FP16
#define INITIAL_VALUE -HALF_MAX
#else // FP16
#define INITIAL_VALUE -FLT_MAX
#endif // FP16
#endif // FIXED_POINT_POSITION

#endif // POOL_AVG

/** Performs a pooling function of pool size equal to N
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are QS8/QS16/F16/F32;
 * @note -DFP16 must be passed at compile time if half float data type is used
 * @note Pool size must be passed using -DPOOL_SIZE e.g. -DPOOL_SIZE=13;
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
__kernel void pooling_layer_N(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, 8)
    vdata           = INITIAL_VALUE;
    DATA_TYPE sdata = INITIAL_VALUE;

    // Load data
    for(int y = 0; y < POOL_SIZE; y++)
    {
        int x = 0;
        for(; x <= ((int)POOL_SIZE - 8); x += 8)
        {
            VEC_DATA_TYPE(DATA_TYPE, 8)
            data0 = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif /* defined(POOL_L2) */
            vdata = POOL_OP(vdata, data0);
        }

        // Leftover
        for(; x < (int)POOL_SIZE; ++x)
        {
            DATA_TYPE data0 = *((__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif /* defined(POOL_L2) */
            sdata = POOL_OP(sdata, data0);
        }
    }

    // Reduce result
    VEC_DATA_TYPE(DATA_TYPE, 4)
    reduce4 = POOL_OP(vdata.s0123, vdata.s4567);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    reduce2       = POOL_OP(reduce4.s01, reduce4.s23);
    DATA_TYPE res = POOL_OP(reduce2.s0, reduce2.s1);
    res           = POOL_OP(res, sdata);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    res = DIV_OP(res, calculate_avg_scale(POOL_SIZE, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)output.ptr = res;
}
#endif // defined(POOL_SIZE)
