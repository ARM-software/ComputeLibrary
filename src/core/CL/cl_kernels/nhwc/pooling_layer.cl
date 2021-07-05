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
    int idx_out_c = GET_SPATIAL_IDX(0, VEC_SIZE, VEC_SIZE_LEFTOVER);
    int idx_out_w = GET_SPATIAL_IDX(1, 1, 0);
#if DST_BATCH_SIZE != 1
    // If batch size != 1, the batch size dimension is collapsed over the height dimension
    int idx_out_h = GET_SPATIAL_IDX(2, 1, 0) % DST_HEIGHT;
    int idx_out_n = GET_SPATIAL_IDX(2, 1, 0) / DST_HEIGHT;
#else  //DST_BATCH_SIZE != 1
    int idx_out_h   = GET_SPATIAL_IDX(2, 1, 0);
    int idx_out_n   = 0;
#endif // DST_BATCH_SIZE != 1

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_n * input_stride_w;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_w * output_stride_y + idx_out_h * output_stride_z + idx_out_n *
                                           output_stride_w;

    VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
    res0 = INITIAL_VALUE;

    int idx_in_w = idx_out_w * STRIDE_X - PAD_X;
    int idx_in_h = idx_out_h * STRIDE_Y - PAD_Y;

    int pool_x_s = max((int)0, -idx_in_w);
    int pool_x_e = min((int)POOL_SIZE_X, (int)SRC_WIDTH - idx_in_w);
    int pool_y_s = max((int)0, -idx_in_h);
    int pool_y_e = min((int)POOL_SIZE_Y, (int)SRC_HEIGHT - idx_in_h);

#if defined(EXCLUDE_PADDING)
    int filter_size = (pool_y_e - pool_y_s) * (pool_x_e - pool_x_s);
#else  // defined(EXCLUDE_PADDING)
    int filter_size = POOL_SIZE_X * POOL_SIZE_Y;
#endif // defined(EXCLUDE_PADDING)

#if POOL_SIZE_X == SRC_WIDTH && POOL_SIZE_Y == SRC_HEIGHT && PAD_X == 0 && PAD_Y == 0
    // Global pooling path
    for(int y = 0; y < POOL_SIZE_Y; ++y)
    {
#pragma unroll 8
        for(int x = 0; x < POOL_SIZE_X; ++x)
        {
#else // POOL_SIZE_X == SRC_WIDTH && POOL_SIZE_Y == SRC_HEIGHT && PAD_X == 0 && PAD_Y == 0
    for(int y = pool_y_s; y < pool_y_e; ++y)
    {
#pragma unroll 8
        for(int x = pool_x_s; x < pool_x_e; ++x)
        {
#endif // POOL_SIZE_X == SRC_WIDTH && POOL_SIZE_Y == SRC_HEIGHT && PAD_X == 0 && PAD_Y == 0
            VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
            data0;
#if defined(FP_MIXED_PRECISION)
            // In case of FP_MIXED_PRECISION, ACC_DATA_TYPE is != DATA_TYPE
            data0 = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + (x + idx_in_w) * input_stride_y + (y + idx_in_h) * input_stride_z)), VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));
#else  // defined(FP_MIXED_PRECISION)
            data0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + (x + idx_in_w) * input_stride_y + (y + idx_in_h) * input_stride_z));
#endif // defined(FP_MIXED_PRECISION)

#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data0 *= data0;
#endif // defined(POOL_L2)
            res0 = POOL_OP(res0, data0);
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
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res_converted0 = CONVERT(res0, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res_converted, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#else  // defined(FP_MIXED_PRECISION)
    STORE_VECTOR_SELECT(res, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#endif // defined(FP_MIXED_PRECISION)
}
#endif // defined(POOL_SIZE_X) && defined(POOL_SIZE_Y)

#define SELECT_TYPE SELECT_VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)

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
#else  //SRC_BATCH_SIZE != 1
    int idx_out_h = get_global_id(2);
    int idx_out_n = 0;
#endif // SRC_BATCH_SIZE != 1

    int idx_in_w = idx_out_w * STRIDE_X - PAD_X;
    int idx_in_h = idx_out_h * STRIDE_Y - PAD_Y;

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_n * input_stride_w;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_w * output_stride_y + idx_out_h * output_stride_z + idx_out_n *
                                           output_stride_w;

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
#else  // defined(FP_MIXED_PRECISION)
    data0         = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y0 * input_stride_z));
    data1         = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y0 * input_stride_z));
    data2         = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x0 * input_stride_y + y1 * input_stride_z));
    data3         = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + x1 * input_stride_y + y1 * input_stride_z));
#endif // defined(FP_MIXED_PRECISION)

#if !defined(POOL_MAX)
    if(filter_size != 4)
    {
        SELECT_TYPE cond_w_s = (SELECT_TYPE)idx_in_w < (SELECT_TYPE)0;
        SELECT_TYPE cond_w_e = (SELECT_TYPE)idx_in_w >= (SELECT_TYPE)(SRC_WIDTH - 1);
        SELECT_TYPE cond_h_s = (SELECT_TYPE)idx_in_h < (SELECT_TYPE)0;
        SELECT_TYPE cond_h_e = (SELECT_TYPE)idx_in_h >= (SELECT_TYPE)(SRC_HEIGHT - 1);

        // Make invalid the values loaded if the x or y coordinate was clamped (out-of-bound)
        data0 = select(data0, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_TYPE)(cond_w_s | cond_h_s));
        data1 = select(data1, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_TYPE)(cond_w_e | cond_h_s));
        data2 = select(data2, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_TYPE)(cond_w_s | cond_h_e));
        data3 = select(data3, (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))INITIAL_VALUE, (SELECT_TYPE)(cond_w_e | cond_h_e));
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
#else  // !defined(EXCLUDE_PADDING)
    res0 /= (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))4;
#endif // defined(EXCLUDE_PADDING)
#endif // defined(POOL_AVG) || defined(POOL_L2)

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res0 = SQRT_OP(res0);
#endif // defined(POOL_L2)

    // Store result
#if defined(FP_MIXED_PRECISION)
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res_converted0 = CONVERT(res0, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res_converted, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#else  // defined(FP_MIXED_PRECISION)
    STORE_VECTOR_SELECT(res, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
#endif // defined(FP_MIXED_PRECISION)

#if defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)

    // This part is used to return the index of the maximum value
    // Note: DST_CHANNELS and DST_BATCH_SIZE can be used for either the input and output tensor

    // note: Batch dimension does not contribute in the offset contribution
    VEC_DATA_TYPE(uint, VEC_SIZE)
    base_index = (uint)idx_out_c;

    base_index += VEC_OFFS(uint, VEC_SIZE);

    VEC_DATA_TYPE(uint, VEC_SIZE)
    index0 = base_index + (uint)x0 * DST_CHANNELS + (uint)y0 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE)
    index1 = base_index + (uint)x1 * DST_CHANNELS + (uint)y0 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE)
    index2 = base_index + (uint)x0 * DST_CHANNELS + (uint)y1 * (DST_CHANNELS * SRC_WIDTH);
    VEC_DATA_TYPE(uint, VEC_SIZE)
    index3 = base_index + (uint)x1 * DST_CHANNELS + (uint)y1 * (DST_CHANNELS * SRC_WIDTH);

    index0 = select(index1, index0, CONVERT(isgreaterequal(data0, data1), VEC_DATA_TYPE(int, VEC_SIZE)));
    index1 = select(index3, index2, CONVERT(isgreaterequal(data2, data3), VEC_DATA_TYPE(int, VEC_SIZE)));
    index0 = select(index1, index0, CONVERT(isgreaterequal(max(data0, data1), max(data2, data3)), VEC_DATA_TYPE(int, VEC_SIZE)));

    __global unsigned char *idx_base_ptr = indices_ptr + indices_offset_first_element_in_bytes + idx_out_c * sizeof(uint) + idx_out_w * indices_stride_y + idx_out_h * indices_stride_z + idx_out_n *
                                           indices_stride_w;

    // Store result
    STORE_VECTOR_SELECT(index, uint, idx_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, ((VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0));
#endif // defined(EXTRACT_MAX_INDEX) && defined(POOL_MAX)
}
#endif // defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)