/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#define DIV_OP_NHWC(x, y) (x * (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(1.f / y))

#if STRIDE_X == 1
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE1(res, input, output)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE2(res, input, output)
#elif STRIDE_X == 3 /* STRIDE_X not equals 1 or 2 */
#define POOLING3x3(res, input, output) POOLING3x3_STRIDE3(res, input, output)
#endif /* STRIDE_X == 3 */

#if defined(FP_MIXED_PRECISION)
#define CONVERT_TO_ACC_DATA_TYPE(x, n) CONVERT(x, VEC_DATA_TYPE(ACC_DATA_TYPE, n))
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) \
    CONVERT_TO_ACC_DATA_TYPE(vload##n(offset, ptr), n)
#else /* defined(FP_MIXED_PRECISION) */
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) vload##n(offset, ptr)
#endif /* defined(FP_MIXED_PRECISION) */

#define POOLING3x3_STRIDE1(res, input, output)                                                                                                       \
    ({                                                                                                                                               \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        data00 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                                   \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 2)                                                                                                              \
        data01 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(2, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 4);                               \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        data10 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                                   \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 2)                                                                                                              \
        data11 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(2, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 4);                               \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        data20 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                                   \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 2)                                                                                                              \
        data21 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(2, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 4);                               \
        data00 = POW2_OP(data00, 4);                                                                                                                 \
        data01 = POW2_OP(data01, 2);                                                                                                                 \
        data10 = POW2_OP(data10, 4);                                                                                                                 \
        data11 = POW2_OP(data11, 2);                                                                                                                 \
        data20 = POW2_OP(data20, 4);                                                                                                                 \
        data21 = POW2_OP(data21, 2);                                                                                                                 \
        \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values00 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data00.s01212323);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values01 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data01.s0, data00.s3, data01.s01);                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values10 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data10.s01212323);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values11 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data11.s0, data10.s3, data11.s01);                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values20 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data20.s01212323);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values21 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data21.s0, data20.s3, data21.s01);                                                              \
        \
        values00 = POOL_OP(values00, values10);                                                                                                      \
        values01 = POOL_OP(values01, values11);                                                                                                      \
        values00 = POOL_OP(values00, values20);                                                                                                      \
        values01 = POOL_OP(values01, values21);                                                                                                      \
        \
        res = POOL_OP((VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s036, values01.s1), (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s147, values01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s25, values01.s03));                                                           \
    })

#define POOLING3x3_STRIDE2(res, input, output)                                                                                                       \
    ({                                                                                                                                               \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        data00               = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                     \
        ACC_DATA_TYPE data01 = (ACC_DATA_TYPE)(*((__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 8));                                       \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        data10               = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                     \
        ACC_DATA_TYPE data11 = (ACC_DATA_TYPE)(*((__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 8));                                       \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        data20               = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                     \
        ACC_DATA_TYPE data21 = (ACC_DATA_TYPE)(*((__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 8));                                       \
        data00               = POW2_OP(data00, 8);                                                                                                   \
        data01               = POW2_OP(data01, 1);                                                                                                   \
        data10               = POW2_OP(data10, 8);                                                                                                   \
        data11               = POW2_OP(data11, 1);                                                                                                   \
        data20               = POW2_OP(data20, 8);                                                                                                   \
        data21               = POW2_OP(data21, 1);                                                                                                   \
        \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values00 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data00.s01223445);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values01 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data00.s667, data01);                                                                           \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values10 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data10.s01223445);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values11 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data10.s667, data11);                                                                           \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                              \
        values20 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))(data20.s01223445);                                                                              \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                              \
        values21 = (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data20.s667, data21);                                                                           \
        \
        values00 = POOL_OP(values00, values10);                                                                                                      \
        values01 = POOL_OP(values01, values11);                                                                                                      \
        values00 = POOL_OP(values00, values20);                                                                                                      \
        values01 = POOL_OP(values01, values21);                                                                                                      \
        \
        res = POOL_OP((VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s036, values01.s1), (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s147, values01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(values00.s25, values01.s03));                                                           \
    })

#define POOLING3x3_STRIDE3(res, input, output)                                                                                               \
    ({                                                                                                                                       \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                      \
        data00 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));                           \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                      \
        data01 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0) + 8);                       \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                      \
        data10 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));                           \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                      \
        data11 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0) + 8);                       \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 8)                                                                                                      \
        data20 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));                           \
        VEC_DATA_TYPE(ACC_DATA_TYPE, 4)                                                                                                      \
        data21 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(4, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0) + 8);                       \
        data00 = POW2_OP(data00, 8);                                                                                                         \
        data01 = POW2_OP(data01, 4);                                                                                                         \
        data10 = POW2_OP(data10, 8);                                                                                                         \
        data11 = POW2_OP(data11, 4);                                                                                                         \
        data20 = POW2_OP(data20, 8);                                                                                                         \
        data21 = POW2_OP(data21, 4);                                                                                                         \
        \
        data00 = POOL_OP(data00, data10);                                                                                                    \
        data01 = POOL_OP(data01, data11);                                                                                                    \
        data00 = POOL_OP(data00, data20);                                                                                                    \
        data01 = POOL_OP(data01, data21);                                                                                                    \
        \
        res = POOL_OP((VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data00.s036, data01.s1), (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data00.s147, data01.s2)); \
        res = POOL_OP(res, (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(data00.s25, data01.s03));                                                       \
    })

ACC_DATA_TYPE calculate_avg_scale(const int pool_size_x, const int pool_size_y, const int upper_bound_w, const int upper_bound_h,
                                  const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int       start_x = get_global_id(0) * stride_x - pad_x;
    int       start_y = get_global_id(1) * stride_y - pad_y;
    const int end_x   = min(start_x + pool_size_x, upper_bound_w);
    const int end_y   = min(start_y + pool_size_y, upper_bound_h);
#if defined(EXCLUDE_PADDING)
    start_x = max(0, start_x);
    start_y = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    return ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to 2.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_2(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(ACC_DATA_TYPE, 2)
    data0 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(2, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));
    VEC_DATA_TYPE(ACC_DATA_TYPE, 2)
    data1 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(2, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));

#if defined(POOL_L2)
    // Raise to power of 2 for L2 Pooling
    data0 = POW2_OP(data0, 2);
    data1 = POW2_OP(data1, 2);
#endif /* defined(POOL_L2) */

    // Perform calculations
    data0             = POOL_OP(data0, data1);
    ACC_DATA_TYPE res = POOL_OP(data0.s0, data0.s1);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average or l2 pooling
    res = DIV_OP(res, calculate_avg_scale(2, 2, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)output.ptr = (DATA_TYPE)res;
}

/** Performs a pooling function of pool size equal to 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_3(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(ACC_DATA_TYPE, 3)
    data0 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(3, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0));
    VEC_DATA_TYPE(ACC_DATA_TYPE, 3)
    data1 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(3, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(ACC_DATA_TYPE, 3)
    data2 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(3, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));

#if defined(POOL_L2)
    // Raise to power of 2 for L2 Pooling
    data0 = POW2_OP(data0, 3);
    data1 = POW2_OP(data1, 3);
    data2 = POW2_OP(data2, 3);
#endif /* defined(POOL_L2) */

    // Perform calculations
    data0             = POOL_OP(data0, data1);
    data0             = POOL_OP(data0, data2);
    ACC_DATA_TYPE res = POOL_OP(POOL_OP(data0.s0, data0.s1), data0.s2);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    res = DIV_OP(res, calculate_avg_scale(3, 3, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)output.ptr = (DATA_TYPE)res;
}

#if defined(POOLING3x3)

#define CONVERT_OP(data_type) convert_##data_type##4
#define CONVERT_VECTOR4(data_type) CONVERT_OP(data_type)

VEC_DATA_TYPE(ACC_DATA_TYPE, 4)
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
    return (VEC_DATA_TYPE(ACC_DATA_TYPE, 4))(1.f) / CONVERT_VECTOR4(ACC_DATA_TYPE)(((int4)(end_y - start_y)) * (end_x - start_x));
}

/** Performs an optimized pooling function of pool size equal to 3 when the stride_x is less equal than 3
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG or -DPOOL_L2 must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_optimized_3(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(ACC_DATA_TYPE, 4)
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

    vstore4(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 4)), 0, (__global DATA_TYPE *)output.ptr);
}
#endif // defined(POOLING3x3)

#if defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)

/** Performs a pooling function of pool size equal to N  (NCHW)
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32;
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_MxN_nchw(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
    vdata               = INITIAL_VALUE;
    ACC_DATA_TYPE sdata = INITIAL_VALUE;

    // Load data
    for(int y = 0; y < POOL_SIZE_Y; y++)
    {
        int x = 0;
        for(; x <= ((int)POOL_SIZE_X - 8); x += 8)
        {
            VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
            data0 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif /* defined(POOL_L2) */
            vdata = POOL_OP(vdata, data0);
        }

        // Leftover
        for(; x < (int)POOL_SIZE_X; ++x)
        {
            ACC_DATA_TYPE data0 = (ACC_DATA_TYPE)(*((__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0)));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif /* defined(POOL_L2) */
            sdata = POOL_OP(sdata, data0);
        }
    }

    // Reduce result
    VEC_DATA_TYPE(ACC_DATA_TYPE, 4)
    reduce4 = POOL_OP(vdata.s0123, vdata.s4567);
    VEC_DATA_TYPE(ACC_DATA_TYPE, 2)
    reduce2           = POOL_OP(reduce4.s01, reduce4.s23);
    ACC_DATA_TYPE res = POOL_OP(reduce2.s0, reduce2.s1);
    res               = POOL_OP(res, sdata);

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    res = DIV_OP(res, calculate_avg_scale(POOL_SIZE_X, POOL_SIZE_Y, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)output.ptr = (DATA_TYPE)res;
}
#endif // defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)

ACC_DATA_TYPE calculate_avg_scale_nhwc(const int pool_size_x, const int pool_size_y, int upper_bound_w, int upper_bound_h,
                                       const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = get_global_id(1) * stride_x - pad_x;
#if defined(DST_DEPTH)
    int start_y = (get_global_id(2) % DST_DEPTH) * stride_y - pad_y;
#else  /* defined(DST_DEPTH) */
    int       start_y    = get_global_id(2) * stride_y - pad_y;
#endif /* defined(DST_DEPTH) */

#if !defined(EXCLUDE_PADDING)
    upper_bound_w += pad_x;
    upper_bound_h += pad_y;
#endif /* defined(EXCLUDE_PADDING) */
    const int end_x = min(start_x + pool_size_x, upper_bound_w);
    const int end_y = min(start_y + pool_size_y, upper_bound_h);
#if defined(EXCLUDE_PADDING)
    start_x = max(0, start_x);
    start_y = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    return ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to N (NHWC)
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Pad values must be passed at compile time using -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_MxN_nhwc(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    // Get pixels pointer
#if defined(DST_DEPTH)
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT(input, DST_DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DST_DEPTH);
#else  /* defined(DST_DEPTH) */
    Tensor3D  input      = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D  output     = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* defined(DST_DEPTH) */

    VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
    vdata = INITIAL_VALUE;

    const int idx_width = get_global_id(1) * STRIDE_X;
#if defined(DST_DEPTH)
    const int idx_height = (get_global_id(2) % DST_DEPTH) * STRIDE_Y;
#else  /* defined(DST_DEPTH) */
    const int idx_height = get_global_id(2) * STRIDE_Y;
#endif /* defined(DST_DEPTH) */

    for(int y = 0; y < POOL_SIZE_Y; ++y)
    {
        int y1 = select(y, PAD_Y - idx_height, y + idx_height - PAD_Y < 0 || y + idx_height - PAD_Y >= MAX_HEIGHT);
        for(int x = 0; x < POOL_SIZE_X; ++x)
        {
            int x1 = select(x, PAD_X - idx_width - 1, x + idx_width - PAD_X < 0 || x + idx_width - PAD_X >= MAX_WIDTH);
            x1     = select(x1, PAD_X - idx_width - 1, y != y1);

#if defined(DST_DEPTH)
            VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
            data0 = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, x1 - PAD_X, y1 - PAD_Y, 0));
#else  /* defined(DST_DEPTH) */
            VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
            data0    = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, x1 - PAD_X, y1 - PAD_Y));
#endif /* defined(DST_DEPTH) */

#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif /* defined(POOL_L2) */
            vdata = POOL_OP(vdata, CONVERT(data0, VEC_DATA_TYPE(ACC_DATA_TYPE, 8)));
        }
    }

#if defined(POOL_AVG) || defined(POOL_L2)
    // Divide by pool region in case of average pooling
    vdata = DIV_OP_NHWC(vdata, calculate_avg_scale_nhwc(POOL_SIZE_X, POOL_SIZE_Y, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y));
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    vdata = SQRT_OP(vdata);
#endif /* defined(POOL_L2) */

    // Store result
    vstore8(CONVERT(vdata, VEC_DATA_TYPE(DATA_TYPE, 8)), 0, (__global DATA_TYPE *)output.ptr);
}

#if defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

inline void offset_no_padding_nchw(const Tensor3D *input, uint *offset_top, uint *offset_bottom)
{
    const int pad_horiz = PAD_TENSOR_LEFT + PAD_TENSOR_RIGHT;
    const int pad_vert  = PAD_TENSOR_TOP + PAD_TENSOR_BOTTOM;

    const int x = get_global_id(0) * STRIDE_X;
    const int y = get_global_id(1) * STRIDE_Y;
    const int z = get_global_id(2);

    //x axis: width, y axis: height, z axis: component
    const uint padded_offset = input->offset_first_element_in_bytes
                               + x * input->stride_x
                               + y * input->stride_y
                               + z * input->stride_z;

    const uint offset_base = padded_offset
                             - y * pad_horiz * sizeof(DATA_TYPE)                                               /* Horizontal padding for each row */
                             - PAD_TENSOR_TOP * input->stride_y                                                /* top padding */
                             - z * MAX_HEIGHT * pad_horiz * sizeof(DATA_TYPE) - z * pad_vert * input->stride_y /* Z plane padding */
                             - PAD_TENSOR_LEFT * sizeof(DATA_TYPE);

#if defined(TENSOR_CHANNEL) && defined(TENSOR_WIDTH) && defined(TENSOR_HEIGHT)
    *offset_top = (uint)((offset_base / sizeof(DATA_TYPE)) % (TENSOR_CHANNEL * TENSOR_WIDTH * TENSOR_HEIGHT));
#else  /* defined(TENSOR_CHANNEL) && defined(TENSOR_WIDTH) && defined(TENSOR_HEIGHT) */
    *offset_top = (uint)(offset_base / sizeof(DATA_TYPE));
#endif /* defined(TENSOR_CHANNEL) && defined(TENSOR_WIDTH) && defined(TENSOR_HEIGHT) */

    *offset_bottom = *offset_top + input->stride_y / sizeof(DATA_TYPE) - pad_horiz;

    return;
}

inline void offset_no_padding_nhwc_3D(const Tensor3D *input, uint *offset_x0, uint *offset_x1, uint *offset_x2, uint *offset_x3)
{
    const int pad_horiz = PAD_TENSOR_LEFT + PAD_TENSOR_RIGHT;

    const int x = get_global_id(0);
    const int y = get_global_id(1) * STRIDE_X;
    const int z = get_global_id(2) * STRIDE_Y;

    //x axis: component, y axis: width, z axis: height
    const uint padded_offset = input->offset_first_element_in_bytes
                               + x * 8 * input->stride_x
                               + y * input->stride_y
                               + z * input->stride_z;

    const uint offset_base = padded_offset
                             - (z + 1) * PAD_TENSOR_TOP * input->stride_y    /* Top padding for each z plane */
                             - y * pad_horiz * sizeof(DATA_TYPE)             /* Horizontal padding for each row */
                             - z * MAX_WIDTH * pad_horiz * sizeof(DATA_TYPE) /* Horizontal padding for each z plane */
                             - PAD_TENSOR_LEFT * sizeof(DATA_TYPE);

    *offset_x0 = (uint)offset_base / sizeof(DATA_TYPE);
    *offset_x1 = *offset_x0 + input->stride_y / sizeof(DATA_TYPE) - pad_horiz;
    *offset_x2 = *offset_x0 + input->stride_z / sizeof(DATA_TYPE) - pad_horiz * MAX_WIDTH - PAD_TENSOR_TOP * input->stride_y / sizeof(DATA_TYPE);
    *offset_x3 = *offset_x2 + input->stride_y / sizeof(DATA_TYPE) - pad_horiz;

    return;
}

#if defined(DST_DEPTH)
inline void offset_no_padding_nhwc_4D(const Tensor4D *input, uint *offset_x0, uint *offset_x1, uint *offset_x2, uint *offset_x3)
{
    const int pad_horiz = PAD_TENSOR_LEFT + PAD_TENSOR_RIGHT;
    const int z_max     = get_global_size(2) / BATCH_SIZE;

    const int x = get_global_id(0);
    const int y = get_global_id(1) * STRIDE_X;
    const int z = (get_global_id(2) % z_max) * STRIDE_Y;
    const int w = get_global_id(2) / z_max;

    const unsigned int padded_offset = input->offset_first_element_in_bytes
                                       + x * 8 * input->stride_x
                                       + y * input->stride_y
                                       + z * input->stride_z;

    const unsigned int offset_base = padded_offset
                                     - (z + 1) * PAD_TENSOR_TOP * input->stride_y    /* Top padding for each z plane */
                                     - y * pad_horiz * sizeof(DATA_TYPE)             /* Horizontal padding for each row */
                                     - z * MAX_WIDTH * pad_horiz * sizeof(DATA_TYPE) /* Horizontal padding for each z plane */
                                     - PAD_TENSOR_LEFT * sizeof(DATA_TYPE);

    *offset_x0 = (uint)offset_base / sizeof(DATA_TYPE);
    *offset_x1 = *offset_x0 + input->stride_y / sizeof(DATA_TYPE) - pad_horiz;
    *offset_x2 = *offset_x0 + input->stride_z / sizeof(DATA_TYPE) - pad_horiz * MAX_WIDTH - PAD_TENSOR_TOP * input->stride_y / sizeof(DATA_TYPE);
    *offset_x3 = *offset_x2 + input->stride_y / sizeof(DATA_TYPE) - pad_horiz;

    return;
}
#endif //defined(DST_DEPTH)

#endif //defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

/** Performs a MAX pooling of pool size equal to 2, and record max value indices for NCHW.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F32
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Tensor padding values must be passed at compile time using PAD_TENSOR_LEFT, PAD_TENSOR_RIGHT, PAD_TENSOR_TOP and PAD_TENSOR_BOTTOM
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F32
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2_nchw_indices_fp32(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output),
    TENSOR3D_DECLARATION(indices))
{
    // Get pixels pointer
    Tensor3D input   = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output  = CONVERT_TO_TENSOR3D_STRUCT(output);
    Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT(indices);

    // Load data
    float2 data0 = VLOAD(2)(0, (__global float *)tensor3D_offset(&input, 0, 0, 0));
    float2 data1 = VLOAD(2)(0, (__global float *)tensor3D_offset(&input, 0, 1, 0));

    // Perform calculations
    float data0_max = POOL_OP(data0.s0, data0.s1);
    float data1_max = POOL_OP(data1.s0, data1.s1);
    float res       = POOL_OP(data0_max, data1_max);
    // Store result
    *(__global float *)output.ptr = res;

#if defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

    uint offset_top    = 0;
    uint offset_bottom = 0;

    offset_no_padding_nchw(&input, &offset_top, &offset_bottom);

    uint index0 = select(offset_top + 1, offset_top, isgreaterequal(data0.s0, data0.s1));
    uint index1 = select(offset_bottom + 1, offset_bottom, isgreaterequal(data1.s0, data1.s1));
    uint index  = select(index1, index0, isgreaterequal(data0_max, data1_max));

    *(__global uint *)indices.ptr = index;

#endif //defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)
}

/** Performs a MAX pooling of pool size equal to 2, and record max value indices for NCHW.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F16
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Tensor padding values must be passed at compile time using PAD_TENSOR_LEFT, PAD_TENSOR_RIGHT, PAD_TENSOR_TOP and PAD_TENSOR_BOTTOM
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F16
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2_nchw_indices_fp16(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output),
    TENSOR3D_DECLARATION(indices))
{
    // Get pixels pointer
    Tensor3D input   = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output  = CONVERT_TO_TENSOR3D_STRUCT(output);
    Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT(indices);

    // Load data
    half2 data0 = VLOAD(2)(0, (__global half *)tensor3D_offset(&input, 0, 0, 0));
    half2 data1 = VLOAD(2)(0, (__global half *)tensor3D_offset(&input, 0, 1, 0));

    // Perform calculations
    half data0_max = POOL_OP(data0.s0, data0.s1);
    half data1_max = POOL_OP(data1.s0, data1.s1);
    half res       = POOL_OP(data0_max, data1_max);
    // Store result
    *(__global half *)output.ptr = res;

#if defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

    uint offset_top    = 0;
    uint offset_bottom = 0;

    offset_no_padding_nchw(&input, &offset_top, &offset_bottom);

    uint index0 = select(offset_top + 1, offset_top, isgreaterequal(data0.s0, data0.s1));
    uint index1 = select(offset_bottom + 1, offset_bottom, isgreaterequal(data1.s0, data1.s1));
    uint index  = select(index1, index0, isgreaterequal(data0_max, data1_max));

    *(__global uint *)indices.ptr = index;

#endif //defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)
}

/** Performs a MAX pooling of pool size equal to 2, and record max value indices for NHWC.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F32
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Tensor padding values must be passed at compile time using PAD_TENSOR_LEFT, PAD_TENSOR_RIGHT, PAD_TENSOR_TOP and PAD_TENSOR_BOTTOM
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F32
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                        Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                          input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                       Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                         output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_stride_w                      Stride of the indices tensor in W dimension (in bytes)
 * @param[in]  indices_step_w                        indices_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2_nhwc_indices_fp32(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output),
    TENSOR4D_DECLARATION(indices))
{
    // Get pixels pointer
#if defined(DST_DEPTH)
    Tensor4D input   = CONVERT_TO_TENSOR4D_STRUCT(input, DST_DEPTH);
    Tensor4D output  = CONVERT_TO_TENSOR4D_STRUCT(output, DST_DEPTH);
    Tensor4D indices = CONVERT_TO_TENSOR4D_STRUCT(indices, DST_DEPTH);
#else  /* defined(DST_DEPTH) */
    Tensor3D input   = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output  = CONVERT_TO_TENSOR3D_STRUCT(output);
    Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT(indices);
#endif /* defined(DST_DEPTH) */

#if defined(DST_DEPTH)
    // Load data
    float8 data_top0    = VLOAD(8)(0, (__global float *)tensor4D_offset(&input, 0, 0, 0, 0));
    float8 data_top1    = VLOAD(8)(0, (__global float *)tensor4D_offset(&input, 0, 1, 0, 0));
    float8 data_bottom0 = VLOAD(8)(0, (__global float *)tensor4D_offset(&input, 0, 0, 1, 0));
    float8 data_bottom1 = VLOAD(8)(0, (__global float *)tensor4D_offset(&input, 0, 1, 1, 0));
#else  /* defined(DST_DEPTH) */
    // Load data
    float8   data_top0    = VLOAD(8)(0, (__global float *)tensor3D_offset(&input, 0, 0, 0));
    float8   data_top1    = VLOAD(8)(0, (__global float *)tensor3D_offset(&input, 0, 1, 0));
    float8   data_bottom0 = VLOAD(8)(0, (__global float *)tensor3D_offset(&input, 0, 0, 1));
    float8   data_bottom1 = VLOAD(8)(0, (__global float *)tensor3D_offset(&input, 0, 1, 1));
#endif /* defined(DST_DEPTH) */

    float8 data_top_max    = POOL_OP(data_top0, data_top1);
    float8 data_bottom_max = POOL_OP(data_bottom0, data_bottom1);
    float8 data_max        = POOL_OP(data_top_max, data_bottom_max);
    vstore8(data_max, 0, (__global float *)output.ptr);

#if defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

    uint offset_x0 = 0;
    uint offset_x1 = 0;
    uint offset_x2 = 0;
    uint offset_x3 = 0;

#if defined(DST_DEPTH)
    offset_no_padding_nhwc_4D(&input, &offset_x0, &offset_x1, &offset_x2, &offset_x3);
#else  /* defined(DST_DEPTH) */
    offset_no_padding_nhwc_3D(&input, &offset_x0, &offset_x1, &offset_x2, &offset_x3);
#endif /* defined(DST_DEPTH) */

    uint8 voffset_x0 = { offset_x0, offset_x0 + 1, offset_x0 + 2, offset_x0 + 3, offset_x0 + 4, offset_x0 + 5, offset_x0 + 6, offset_x0 + 7 };
    uint8 voffset_x1 = { offset_x1, offset_x1 + 1, offset_x1 + 2, offset_x1 + 3, offset_x1 + 4, offset_x1 + 5, offset_x1 + 6, offset_x1 + 7 };
    uint8 voffset_x2 = { offset_x2, offset_x2 + 1, offset_x2 + 2, offset_x2 + 3, offset_x2 + 4, offset_x2 + 5, offset_x2 + 6, offset_x2 + 7 };
    uint8 voffset_x3 = { offset_x3, offset_x3 + 1, offset_x3 + 2, offset_x3 + 3, offset_x3 + 4, offset_x3 + 5, offset_x3 + 6, offset_x3 + 7 };

    uint8 index0 = select(voffset_x1, voffset_x0, isgreaterequal(data_top0, data_top1));
    uint8 index1 = select(voffset_x3, voffset_x2, isgreaterequal(data_bottom0, data_bottom1));
    uint8 index  = select(index1, index0, isgreaterequal(data_top_max, data_bottom_max));
    vstore8(index, 0, (__global uint *)indices.ptr);

#endif /* defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM */
}

/** Performs a MAX pooling of pool size equal to 2, and record max value indices for NHWC.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F16
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Tensor padding values must be passed at compile time using PAD_TENSOR_LEFT, PAD_TENSOR_RIGHT, PAD_TENSOR_TOP and PAD_TENSOR_BOTTOM
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F16
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                        Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                          input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                       Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                         output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_stride_w                      Stride of the indices tensor in W dimension (in bytes)
 * @param[in]  indices_step_w                        indices_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2_nhwc_indices_fp16(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output),
    TENSOR4D_DECLARATION(indices))
{
    // Get pixels pointer
#if defined(DST_DEPTH)
    Tensor4D input   = CONVERT_TO_TENSOR4D_STRUCT(input, DST_DEPTH);
    Tensor4D output  = CONVERT_TO_TENSOR4D_STRUCT(output, DST_DEPTH);
    Tensor4D indices = CONVERT_TO_TENSOR4D_STRUCT(indices, DST_DEPTH);
#else  /* defined(DST_DEPTH) */
    Tensor3D input        = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output       = CONVERT_TO_TENSOR3D_STRUCT(output);
    Tensor3D indices      = CONVERT_TO_TENSOR3D_STRUCT(indices);
#endif /* defined(DST_DEPTH) */

#if defined(DST_DEPTH)
    // Load data
    half8 data_top0    = VLOAD(8)(0, (__global half *)tensor4D_offset(&input, 0, 0, 0, 0));
    half8 data_top1    = VLOAD(8)(0, (__global half *)tensor4D_offset(&input, 0, 1, 0, 0));
    half8 data_bottom0 = VLOAD(8)(0, (__global half *)tensor4D_offset(&input, 0, 0, 1, 0));
    half8 data_bottom1 = VLOAD(8)(0, (__global half *)tensor4D_offset(&input, 0, 1, 1, 0));
#else  /* defined(DST_DEPTH) */
    // Load data
    half8 data_top0    = VLOAD(8)(0, (__global half *)tensor3D_offset(&input, 0, 0, 0));
    half8 data_top1    = VLOAD(8)(0, (__global half *)tensor3D_offset(&input, 0, 1, 0));
    half8 data_bottom0 = VLOAD(8)(0, (__global half *)tensor3D_offset(&input, 0, 0, 1));
    half8 data_bottom1 = VLOAD(8)(0, (__global half *)tensor3D_offset(&input, 0, 1, 1));
#endif /* defined(DST_DEPTH) */

    half8 data_top_max    = POOL_OP(data_top0, data_top1);
    half8 data_bottom_max = POOL_OP(data_bottom0, data_bottom1);
    half8 data_max        = POOL_OP(data_top_max, data_bottom_max);
    vstore8(data_max, 0, (__global half *)output.ptr);

#if defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM)

    uint offset_x0_int = 0;
    uint offset_x1_int = 0;
    uint offset_x2_int = 0;
    uint offset_x3_int = 0;

#if defined(DST_DEPTH)
    offset_no_padding_nhwc_4D(&input, &offset_x0_int, &offset_x1_int, &offset_x2_int, &offset_x3_int);
#else  /* defined(DST_DEPTH) */
    offset_no_padding_nhwc_3D(&input, &offset_x0_int, &offset_x1_int, &offset_x2_int, &offset_x3_int);
#endif /* defined(DST_DEPTH) */

    ushort offset_x0 = (ushort)offset_x0_int;
    ushort offset_x1 = (ushort)offset_x1_int;
    ushort offset_x2 = (ushort)offset_x2_int;
    ushort offset_x3 = (ushort)offset_x3_int;

    ushort8 voffset_x0 = { offset_x0, offset_x0 + 1, offset_x0 + 2, offset_x0 + 3, offset_x0 + 4, offset_x0 + 5, offset_x0 + 6, offset_x0 + 7 };
    ushort8 voffset_x1 = { offset_x1, offset_x1 + 1, offset_x1 + 2, offset_x1 + 3, offset_x1 + 4, offset_x1 + 5, offset_x1 + 6, offset_x1 + 7 };
    ushort8 voffset_x2 = { offset_x2, offset_x2 + 1, offset_x2 + 2, offset_x2 + 3, offset_x2 + 4, offset_x2 + 5, offset_x2 + 6, offset_x2 + 7 };
    ushort8 voffset_x3 = { offset_x3, offset_x3 + 1, offset_x3 + 2, offset_x3 + 3, offset_x3 + 4, offset_x3 + 5, offset_x3 + 6, offset_x3 + 7 };

    ushort8 index0 = select(voffset_x1, voffset_x0, isgreaterequal(data_top0, data_top1));
    ushort8 index1 = select(voffset_x3, voffset_x2, isgreaterequal(data_bottom0, data_bottom1));
    ushort8 index  = select(index1, index0, isgreaterequal(data_top_max, data_bottom_max));
    vstore8(CONVERT(index, uint8), 0, (__global uint *)indices.ptr);

#endif /* defined(PAD_TENSOR_LEFT) && defined(PAD_TENSOR_RIGHT) && defined(PAD_TENSOR_TOP) && defined(PAD_TENSOR_BOTTOM */
}