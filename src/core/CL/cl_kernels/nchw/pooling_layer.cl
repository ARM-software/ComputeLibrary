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

#if defined(POOL_AVG) || defined(POOL_L2)
#define POOL_OP(x, y) ((x) + (y))
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#if defined(QUANTIZED)
#define POOL_OP(x, y) (max((x), (y)))
#else // defined(QUANTIZED)
#define POOL_OP(x, y) (fmax((x), (y)))
#endif // defined(QUANTIZED)
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
#define POW2_OP(x, vec_size) ((x) * (x))
#else /* defined(POOL_L2) */
#define POW2_OP(x, vec_size) (x)
#endif /* defined(POOL_L2) */

#define DIV_OP(x, y) (x * (1.f / y))
#define SQRT_OP(x) sqrt((x))

#if defined(FP_MIXED_PRECISION) || defined(QUANTIZED)
#define CONVERT_TO_ACC_DATA_TYPE(x, n) CONVERT(x, VEC_DATA_TYPE(ACC_DATA_TYPE, n))
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) CONVERT_TO_ACC_DATA_TYPE(vload##n(offset, ptr), n)
#else /* defined(FP_MIXED_PRECISION) || defined(QUANTIZED)*/
#define VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(n, offset, ptr) vload##n(offset, ptr)
#endif /* defined(FP_MIXED_PRECISION) || defined(QUANTIZED)*/

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
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32/QASYMM8;
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32/QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_layer_MxN_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id2 = get_global_id(2);

    int x_coords = (id0 * STRIDE_X) - PAD_X;
    int y_coords = (id1 * STRIDE_Y) - PAD_Y;

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + y_coords * (int)src_stride_y + id2 * src_stride_z;

    VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
    vdata               = INITIAL_VALUE;
    ACC_DATA_TYPE sdata = INITIAL_VALUE;

    const int end_x = min((int)POOL_SIZE_X, (int)(SRC_WIDTH - x_coords));
    const int end_y = min((int)POOL_SIZE_Y, (int)(SRC_HEIGHT - y_coords));

    // Load data
    for(int y = 0; y < end_y; ++y)
    {
        if((y_coords + y) >= 0)
        {
            int x = 0;
            for(; x <= (end_x - 8); x += 8)
            {
                int8 src_x = (int8)(x_coords + x) + VEC_OFFS(int, 8);
#if defined(POOL_AVG) || defined(POOL_L2)
                SELECT_VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
                cond_x = CONVERT(src_x < 0, SELECT_VEC_DATA_TYPE(ACC_DATA_TYPE, 8));
                src_x  = clamp(src_x, (int8)0, (int8)(SRC_WIDTH - 1));
                VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
                data0 = select(VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)(src_addr + src_x.s0 * sizeof(DATA_TYPE) + y * src_stride_y)), (VEC_DATA_TYPE(ACC_DATA_TYPE, 8))0, REVERSE(cond_x, 8));
#else  // defined(POOL_AVG) || defined(POOL_L2)
                src_x = clamp(src_x, 0, SRC_WIDTH - 1);
                VEC_DATA_TYPE(ACC_DATA_TYPE, 8)
                data0               = VLOAD_AND_CONVERT_TO_ACC_DATA_TYPE(8, 0, (__global DATA_TYPE *)(src_addr + src_x.s0 * sizeof(DATA_TYPE) + y * src_stride_y));
#endif // defined(POOL_AVG) || defined(POOL_L2

#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data0 *= data0;
#endif /* defined(POOL_L2) */

                vdata = POOL_OP(vdata, data0);
            }

            // Leftover
            for(; x < end_x; ++x)
            {
                int src_x = x_coords + x;
#if defined(POOL_AVG) || defined(POOL_L2)
                SELECT_DATA_TYPE(ACC_DATA_TYPE)
                cond_x              = (src_x < 0);
                src_x               = clamp(src_x, 0, SRC_WIDTH - 1);
                ACC_DATA_TYPE data0 = select((ACC_DATA_TYPE)(*((__global DATA_TYPE *)(src_addr + src_x * sizeof(DATA_TYPE) + y * src_stride_y))), (ACC_DATA_TYPE)0, cond_x);
#else  // defined(POOL_AVG) || defined(POOL_L2)
                src_x               = clamp(src_x, 0, SRC_WIDTH - 1);
                ACC_DATA_TYPE data0 = (ACC_DATA_TYPE)(*((__global DATA_TYPE *)(src_addr + src_x * sizeof(DATA_TYPE) + y * src_stride_y)));
#endif // defined(POOL_AVG) || defined(POOL_L2)

#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data0 *= data0;
#endif /* defined(POOL_L2) */

                sdata = POOL_OP(sdata, data0);
            }
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

#if defined(QUANTIZED)

    DATA_TYPE result_q8 = CONVERT(res, DATA_TYPE);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)

    const float result_f32   = convert_float(result_q8);
    const float input_offset = (float)OFFSET_IN1;
    const float input_scale  = (float)SCALE_IN1;
    const float scale_out    = (float)SCALE_OUT;
    const float offset_out   = (float)OFFSET_OUT;
    const float in_f32       = (result_f32 - input_offset) * input_scale;
    const float out_f32      = in_f32 / scale_out + offset_out;
    result_q8                = CONVERT_SAT(convert_int_rte(out_f32), DATA_TYPE);

#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    *(__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + id0 * sizeof(DATA_TYPE) + id1 * dst_stride_y + id2 * dst_stride_z) = result_q8;

#else // defined(QUANTIZED)

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    *(__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + id0 * sizeof(DATA_TYPE) + id1 * dst_stride_y + id2 * dst_stride_z) = (DATA_TYPE)res;
#endif // defined(QUANTIZED)
}
#endif // defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)

/** Performs a MAX pooling of pool size equal to 2, and record max value indices for NCHW.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are F32
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void pooling_layer_2_nchw_indices(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(indices))
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id2 = get_global_id(2);

    int2 x_coords = clamp((int2)((id0 * STRIDE_X) - PAD_X), (int2)0, (int2)(SRC_WIDTH - 1));
    int2 y_coords = clamp((int2)((id1 * STRIDE_Y) - PAD_Y) + VEC_OFFS(int, 2), (int2)0, (int2)(SRC_HEIGHT - 1));

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + id2 * src_stride_z;

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = VLOAD(2)(0, (__global DATA_TYPE *)(src_addr + x_coords.s0 * sizeof(DATA_TYPE) + y_coords.s0 * (int)src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = VLOAD(2)(0, (__global DATA_TYPE *)(src_addr + x_coords.s1 * sizeof(DATA_TYPE) + y_coords.s1 * (int)src_stride_y));

    // Perform calculations
    DATA_TYPE data0_max = POOL_OP(data0.s0, data0.s1);
    DATA_TYPE data1_max = POOL_OP(data1.s0, data1.s1);
    DATA_TYPE res       = POOL_OP(data0_max, data1_max);
    // Store result
    *(__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + id0 * sizeof(DATA_TYPE) + id1 * dst_stride_y + id2 * dst_stride_z) = res;

#if defined(SRC_BATCH)

    uint offset_top    = (x_coords.s0 + y_coords.s0 * SRC_WIDTH + id2 * (SRC_WIDTH * SRC_HEIGHT)) % SRC_BATCH;
    uint offset_bottom = offset_top + SRC_WIDTH;

    uint index0 = select(offset_top + 1, offset_top, isgreaterequal(data0.s0, data0.s1));
    uint index1 = select(offset_bottom + 1, offset_bottom, isgreaterequal(data1.s0, data1.s1));
    uint index  = select(index1, index0, isgreaterequal(data0_max, data1_max));

    *(__global uint *)(indices_ptr + indices_offset_first_element_in_bytes + id0 * sizeof(uint) + id1 * indices_stride_y + id2 * indices_stride_z) = index;

#endif // defined(SRC_BATCH)
}