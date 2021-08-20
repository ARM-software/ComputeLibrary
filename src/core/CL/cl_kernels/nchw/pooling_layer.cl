/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "tile_helpers.h"

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

#if defined(FP_MIXED_PRECISION)
#define CONVERT_TO_ACC_DATA_TYPE(x, n) CONVERT(x, VEC_DATA_TYPE(ACC_DATA_TYPE, n))
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) \
    CONVERT_TO_ACC_DATA_TYPE(vload##n(offset, ptr), n)
#else /* defined(FP_MIXED_PRECISION) */
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) vload##n(offset, ptr)
#endif /* defined(FP_MIXED_PRECISION) */

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