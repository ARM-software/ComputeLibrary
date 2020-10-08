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
#include "repeat.h"

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

#if defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)

#if defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)
/** Performs pooling layer of size equal to MxN. This OpenCL kernel can perform the following pooling types:
 * -# max, -DPOOL_MAX must be passed at compile time
 * -# average, -DPOOL_AVG must be passed at compile time. If padding has to be expluded, -DEXCLUDE_PADDING should be passed at compile time
 * -# l2 normalisation, -DPOOL_L2 must be passed at compile time
 *
 * @note Datatype must be passed at compile type using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F32/F16
 * @note Accumulation data type must be passed at compile time using -DACC_DATA_TYPE e.g. -DACC_DATA_TYPE=float
 * @note If -DFP_MIXED_PRECISION is passed at compile time, the kernel will use F32 for the partial result
 * @note Pool size must be passed at compile time using -DPOOL_SIZE_X and -DPOOL_SIZE_Y. e.g. -DPOOL_SIZE_X=4, -DPOOL_SIZE_Y=4
 * @note Input tensor width and height must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT
 * @note Output tensor height, channels and batch size must be passed at compile time using -DDST_HEIGHT, -DDST_CHANNELS and -DDST_BATCH_SIZE
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Pool pads must be passed at compile time using -DPAD_X and -DPAD_Y
 * @note Vector size must be passed at compile time using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size must be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F32/F16
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
    // Note: If C is not multiple of VEC_SIZE, we shift back of VEC_SIZE_LEFTOVER elements to compute the leftover elements for get_global_id(0) == 0
    // Note: If C is less than VEC_SIZE, VEC_SIZE should be SHRINKED to the closest smaller VEC_SIZE. This operation is performed on the host side
    int offset_c = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0) * sizeof(DATA_TYPE);
    int idx_out_w = get_global_id(1);
#if DST_BATCH_SIZE != 1
    // If batch size != 1, the batch size dimension is collapsed over the height dimension
    int idx_out_h = get_global_id(2) % DST_HEIGHT;
    int idx_out_n = get_global_id(2) / DST_HEIGHT;
#else //DST_BATCH_SIZE != 1
    int idx_out_h = get_global_id(2);
    int idx_out_n = 0;
#endif // DST_BATCH_SIZE != 1

    int idx_in_w  = idx_out_w * STRIDE_X - PAD_X;
    int idx_in_h  = idx_out_h * STRIDE_Y - PAD_Y;

    int pool_x_s = max((int)0, -idx_in_w);
    int pool_x_e = min((int)POOL_SIZE_X, (int)SRC_WIDTH - idx_in_w);
    int pool_y_s = max((int)0, -idx_in_h);
    int pool_y_e = min((int)POOL_SIZE_Y, (int)SRC_HEIGHT - idx_in_h);

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes +
                                                      offset_c +
                                                      idx_out_n * input_stride_w;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes +
                                                        offset_c +
                                                        idx_out_w * output_stride_y +
                                                        idx_out_h * output_stride_z +
                                                        idx_out_n * output_stride_w;

#if ((defined(POOL_AVG) || defined(POOL_L2)))
#if defined(EXCLUDE_PADDING)
    int filter_size = 0;
#else // defined(EXCLUDE_PADDING)
    int filter_size = POOL_SIZE_X * POOL_SIZE_Y;
#endif // defined(EXCLUDE_PADDING)
#endif // ((defined(POOL_AVG) || defined(POOL_L2)))

    VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
    res0 = INITIAL_VALUE;

    for(int y = pool_y_s; y < pool_y_e; ++y)
    {
        for(int x = pool_x_s; x < pool_x_e; ++x)
        {
            VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE) data0;
#if defined(FP_MIXED_PRECISION)
            // In case of FP_MIXED_PRECISION, ACC_DATA_TYPE is != DATA_TYPE
            data0 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + (x + idx_in_w) * input_stride_y + (y + idx_in_h) * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
#else // defined(FP_MIXED_PRECISION)
            data0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + (x + idx_in_w) * input_stride_y + (y + idx_in_h) * input_stride_z));
#endif // defined(FP_MIXED_PRECISION)

#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif // defined(POOL_L2)
            res0 = POOL_OP(res0, data0);

#if ((defined(POOL_AVG) || defined(POOL_L2))) && defined(EXCLUDE_PADDING)
            filter_size++;
#endif // ((defined(POOL_AVG) || defined(POOL_L2))) && defined(EXCLUDE_PADDING)
        }
    }

#if defined(POOL_AVG) || defined(POOL_L2)
    res0 /= (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))filter_size;
#endif // defined(POOL_AVG) || defined(POOL_L2)

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res0 = SQRT_OP(res0);
#endif // defined(POOL_L2)

    // Store result
#if defined(FP_MIXED_PRECISION)
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) res_converted0 = CONVERT(res0, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res_converted, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#else // defined(FP_MIXED_PRECISION)
    STORE_VECTOR_SELECT(res, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#endif // defined(FP_MIXED_PRECISION)
}
#endif // defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)

/** Performs pooling layer of size equal to 2. This OpenCL kernel can perform the following pooling types:
 * -# max, -DPOOL_MAX must be passed at compile time
 * -# max extracting the max index, -DPOOL_MAX and -DEXTRACT_MAX_INDEX must be passed at compile time
 * -# average, -DPOOL_AVG must be passed at compile time. If padding has to be expluded, -DEXCLUDE_PADDING should be passed at compile time
 * -# l2 normalisation, -DPOOL_L2 must be passed at compile time
 *
 * @note Datatype must be passed at compile type using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F32/F16
 * @note Accumulation data type must be passed at compile time using -DACC_DATA_TYPE e.g. -DACC_DATA_TYPE=float
 * @note If -DFP_MIXED_PRECISION is passed at compile time, the kernel will use F32 for the partial result
 * @note Input tensor width and height must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT
 * @note Output tensor height, channels and batch size must be passed at compile time using -DDST_HEIGHT, -DDST_CHANNELS and -DDST_BATCH_SIZE
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Pool pads must be passed at compile time using -DPAD_X and -DPAD_Y
 * @note Vector size must be passed at compile time using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size must be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F32/F16
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
 * @param[in]  indices_ptr                           (Optional) Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      (Optional) Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        (Optional) indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      (Optional) Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        (Optional) indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      (Optional) Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        (Optional) indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_stride_w                      (Optional) Stride of the indices tensor in W dimension (in bytes)
 * @param[in]  indices_step_w                        (Optional) indices_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes (Optional) The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2x2_nhwc(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output)
#if defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)
    ,
    TENSOR4D_DECLARATION(indices)
#endif // defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)
)
{
    // Note: If C is not multiple of VEC_SIZE, we shift back of VEC_SIZE_LEFTOVER elements to compute the leftover elements for get_global_id(0) == 0
    // Note: If C is less than VEC_SIZE, VEC_SIZE should be SHRINKED to the closest smaller VEC_SIZE. This operation is performed on the host side
    int idx_out_c = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    int idx_out_w = get_global_id(1);
#if DST_BATCH_SIZE != 1
    // If batch size != 1, the batch size dimension is collapsed over the height dimension
    int idx_out_h = get_global_id(2) % DST_HEIGHT;
    int idx_out_n = get_global_id(2) / DST_HEIGHT;
#else //SRC_BATCH_SIZE != 1
    int idx_out_h = get_global_id(2);
    int idx_out_n = 0;
#endif // SRC_BATCH_SIZE != 1

    int idx_in_w  = idx_out_w * STRIDE_X - PAD_X;
    int idx_in_h  = idx_out_h * STRIDE_Y - PAD_Y;

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes +
                                                      idx_out_c * sizeof(DATA_TYPE) +
                                                      idx_out_n * input_stride_w;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes +
                                                        idx_out_c * sizeof(DATA_TYPE) +
                                                        idx_out_w * output_stride_y +
                                                        idx_out_h * output_stride_z +
                                                        idx_out_n * output_stride_w;

    int pool_x_s = max((int)0, -idx_in_w);
    int pool_x_e = min((int)2, (int)SRC_WIDTH - idx_in_w);
    int pool_y_s = max((int)0, -idx_in_h);
    int pool_y_e = min((int)2, (int)SRC_HEIGHT - idx_in_h);

    int filter_size = (pool_x_e - pool_x_s) * (pool_y_e - pool_y_s);

    int x0 = pool_x_s + idx_in_w;
    int y0 = pool_y_s + idx_in_h;
    int x1 = pool_x_e - 1 + idx_in_w;
    int y1 = pool_y_e - 1 + idx_in_h;

    REPEAT_VAR_INIT_TO_CONST(4, VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE), data, 0);

#if defined(FP_MIXED_PRECISION)
    // In case of FP_MIXED_PRECISION, ACC_DATA_TYPE is != DATA_TYPE
    data0 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y0 * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
    data1 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y0 * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
    data2 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y1 * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
    data3 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y1 * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
#else // defined(FP_MIXED_PRECISION)
    data0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y0 * input_stride_z));
    data1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y0 * input_stride_z));
    data2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y1 * input_stride_z));
    data3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y1 * input_stride_z));
#endif // defined(FP_MIXED_PRECISION)

#if !defined(POOL_MAX)
    if(filter_size != 4)
    {
        // Make invalid the values loaded if the x or y coordinate was clamped (out-of-bound)
        data1 = select(data1, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))(pool_x_e == pool_x_s));
        data2 = select(data2, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))(pool_y_e == pool_y_s));
        data3 = select(data3, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))((pool_x_e == pool_x_s) || (pool_y_e == pool_y_s)));
    }
#endif // !defined(POOL_MAX)

#if defined(POOL_L2)
    // Raise to power of 2 for L2 Pooling
    data0 *= data0;
    data1 *= data1;
    data2 *= data2;
    data3 *= data3;
#endif /* defined(POOL_L2) */

    VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
    res0 = data0;
    res0 = POOL_OP(res0, data1);
    res0 = POOL_OP(res0, data2);
    res0 = POOL_OP(res0, data3);

#if defined(POOL_AVG) || defined(POOL_L2)
#if defined(EXCLUDE_PADDING)
    res0 /= (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))filter_size;
#else // !defined(EXCLUDE_PADDING)
    res0 /= (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))4;
#endif // defined(EXCLUDE_PADDING)
#endif // defined(POOL_AVG) || defined(POOL_L2)

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res0 = SQRT_OP(res0);
#endif // defined(POOL_L2)

    // Store result
#if defined(FP_MIXED_PRECISION)
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) res_converted0 = CONVERT(res0, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res_converted, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#else // defined(FP_MIXED_PRECISION)
    STORE_VECTOR_SELECT(res, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#endif // defined(FP_MIXED_PRECISION)

#if defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)

    // This part is used to return the index of the maximum value
    // Note: DST_CHANNELS and DST_BATCH_SIZE can be used for either the input and output tensor

    // note: Batch dimension does not contribute in the offset contribution
    VEC_DATA_TYPE(uint, VEC_SIZE) base_index = (uint)idx_out_c;

    base_index += VEC_OFFS(VEC_DATA_TYPE(uint, VEC_SIZE), VEC_SIZE);

    VEC_DATA_TYPE(uint, VEC_SIZE) index0 = base_index + (uint)x0 * DST_CHANNELS + (uint)y0 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE) index1 = base_index + (uint)x1 * DST_CHANNELS + (uint)y0 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE) index2 = base_index + (uint)x0 * DST_CHANNELS + (uint)y1 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE) index3 = base_index + (uint)x1 * DST_CHANNELS + (uint)y1 * (DST_CHANNELS * SRC_WIDTH);

    index0 = select(index1, index0, CONVERT(isgreaterequal(data0, data1), VEC_DATA_TYPE(int, VEC_SIZE)));
    index1 = select(index3, index2, CONVERT(isgreaterequal(data2, data3), VEC_DATA_TYPE(int, VEC_SIZE)));
    index0 = select(index1, index0, CONVERT(isgreaterequal(max(data0, data1), max(data2, data3)), VEC_DATA_TYPE(int, VEC_SIZE)));

    __global unsigned char *idx_base_ptr = indices_ptr + indices_offset_first_element_in_bytes +
                                                         idx_out_c * sizeof(uint) +
                                                         idx_out_w * indices_stride_y +
                                                         idx_out_h * indices_stride_z +
                                                         idx_out_n * indices_stride_w;

    // Store result
    STORE_VECTOR_SELECT(index, uint, idx_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, ((VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0));
#endif // defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)
}
#endif // defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_BATCH_SIZE) && defined(SELECT_DATA_TYPE) && defined(ACC_DATA_TYPE)