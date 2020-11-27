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

#if defined(DATA_TYPE) && defined(INITIAL_VALUE)
#define VEC_TYPE(VEC_SIZE) VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
#define VEC_FLOAT(VEC_SIZE) VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT(VEC_SIZE) VEC_DATA_TYPE(int, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
#define REQUANTIZE(VEC_SIZE, input, in_offset, out_offset, in_scale, out_scale, res)                                                                                  \
    {                                                                                                                                                                 \
        const VEC_FLOAT(VEC_SIZE) in_f32  = (CONVERT(input, VEC_FLOAT(VEC_SIZE)) - (VEC_FLOAT(VEC_SIZE))((float)in_offset)) * (VEC_FLOAT(VEC_SIZE))((float)in_scale); \
        const VEC_FLOAT(VEC_SIZE) out_f32 = in_f32 / ((VEC_FLOAT(VEC_SIZE))(float)out_scale) + ((VEC_FLOAT(VEC_SIZE))((float)out_offset));                            \
        res                               = CONVERT_SAT(CONVERT_DOWN(out_f32, VEC_INT(VEC_SIZE)), VEC_TYPE(VEC_SIZE));                                                \
    }
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

#if defined(POOL_AVG)
#define POOL_OP(x, y) ((x) + (y))
#else /* defined(POOL_AVG) */
#define POOL_OP(x, y) (max((x), (y)))
#endif /* defined(POOL_AVG) */

#define DIV_OP(x, y) (x * (1.f / y))

#if defined(POOL_L2)
#error "L2 pooling is not supported"
#endif /* defined(POOL_L2) */

int calculate_avg_scale(const int pool_size_x, const int pool_size_y, const int upper_bound_w, const int upper_bound_h,
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

/** Performs a pooling function of pool size equal to N (NCHW)
 *
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *       -DMAX_WIDTH and -DMAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 *       -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 * @note Input data type must be passed at compile time using -DDAT_TYPE=type, e.g. -DDATA_TYPE=uchar
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED
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
__kernel void pooling_layer_MxN_quantized_nchw(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    int8 vdata = INITIAL_VALUE;
    int  sdata = INITIAL_VALUE;

    // Load data
    for(int y = 0; y < POOL_SIZE_Y; y++)
    {
        int x = 0;
        for(; x <= ((int)POOL_SIZE_X - 8); x += 8)
        {
            VEC_TYPE(8)
            data       = vload8(0, (__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0));
            int8 data0 = convert_int8(data);
            vdata      = POOL_OP(vdata, data0);
        }

        // Leftover
        for(; x < (int)POOL_SIZE_X; ++x)
        {
            DATA_TYPE data = *((__global DATA_TYPE *)tensor3D_offset(&input, x, y, 0));
            int data0      = convert_int(data);
            sdata          = POOL_OP(sdata, data0);
        }
    }

    // Reduce result
    int4 reduce4 = POOL_OP(vdata.s0123, vdata.s4567);
    int2 reduce2 = POOL_OP(reduce4.s01, reduce4.s23);
    int  res     = POOL_OP(reduce2.s0, reduce2.s1);
    res          = POOL_OP(res, sdata);

#if defined(POOL_AVG)
    res = round(DIV_OP(res, calculate_avg_scale(POOL_SIZE_X, POOL_SIZE_Y, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y)));
#endif /* defined(POOL_AVG) */

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

    *(__global DATA_TYPE *)output.ptr = result_q8;
}

#if defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)
/** Performs pooling layer of size equal to MxN. This OpenCL kernel can perform the following pooling types:
 * -# max, -DPOOL_MAX must be passed at compile time
 * -# average, -DPOOL_AVG must be passed at compile time. If padding has to be expluded, -DEXCLUDE_PADDING should be passed at compile time
 *
 * @note Datatype must be passed at compile type using -DDATA_TYPE e.g. -DDATA_TYPE=uchar. Supported data types are QASYMM8/QASYMM8_SIGNED
 * @note Accumulation data type must be passed at compile time using -DACC_DATA_TYPE e.g. -DACC_DATA_TYPE=int
 * @note Pool size must be passed at compile time using -DPOOL_SIZE_X and -DPOOL_SIZE_Y. e.g. -DPOOL_SIZE_X=4, -DPOOL_SIZE_Y=4
 * @note Input tensor width and height must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT
 * @note Output tensor height, channels and batch size must be passed at compile time using -DDST_HEIGHT, -DDST_CHANNELS and -DDST_BATCH_SIZE
 * @note Pool strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Pool pads must be passed at compile time using -DPAD_X and -DPAD_Y
 * @note Vector size must be passed at compile time using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size must be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 * @note If the output has be requantized, -DOFFSET_IN1, -DOFFSET_OUT, -DSCALE_IN1 and -DSCALE_OUT muste be passed at compile time
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pooling_layer_MxN_quantized_nhwc(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    // Note: If C is not multiple of VEC_SIZE, we shift back of VEC_SIZE_LEFTOVER elements to compute the leftover elements for get_global_id(0) == 0
    // Note: If C is less than VEC_SIZE, VEC_SIZE should be SHRINKED to the closest smaller VEC_SIZE. This operation is performed on the host side
    int offset_c  = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0) * sizeof(DATA_TYPE);
    int idx_out_w = get_global_id(1);
#if DST_BATCH_SIZE != 1
    // If batch size != 1, the batch size dimension is collapsed over the height dimension
    int idx_out_h = get_global_id(2) % DST_HEIGHT;
    int idx_out_n = get_global_id(2) / DST_HEIGHT;
#else  //DST_BATCH_SIZE != 1
    int idx_out_h   = get_global_id(2);
    int idx_out_n   = 0;
#endif // DST_BATCH_SIZE != 1

    int idx_in_w = idx_out_w * STRIDE_X - PAD_X;
    int idx_in_h = idx_out_h * STRIDE_Y - PAD_Y;

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes + offset_c + idx_out_n * input_stride_w;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes + offset_c + idx_out_w * output_stride_y + idx_out_h * output_stride_z + idx_out_n * output_stride_w;

    int pool_x_s = max((int)0, -idx_in_w);
    int pool_x_e = min((int)POOL_SIZE_X, (int)SRC_WIDTH - idx_in_w);
    int pool_y_s = max((int)0, -idx_in_h);
    int pool_y_e = min((int)POOL_SIZE_Y, (int)SRC_HEIGHT - idx_in_h);

#if defined(POOL_AVG) && defined(EXCLUDE_PADDING)
    int filter_size = 0;
#elif defined(POOL_AVG) && !defined(EXCLUDE_PADDING) // defined(POOL_AVG) && defined(EXCLUDE_PADDING)
    int filter_size = POOL_SIZE_X * POOL_SIZE_Y;
#endif                                               // defined(POOL_AVG) && !defined(EXCLUDE_PADDING)

    VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
    res0 = INITIAL_VALUE;

    for(int y = pool_y_s; y < pool_y_e; ++y)
    {
        for(int x = pool_x_s; x < pool_x_e; ++x)
        {
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            data;
            VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
            data0;

            data  = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + (x + idx_in_w) * input_stride_y + (y + idx_in_h) * input_stride_z));
            data0 = CONVERT(data, VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));

            res0 = POOL_OP(res0, data0);

#if defined(POOL_AVG) && defined(EXCLUDE_PADDING)
            filter_size++;
#endif // defined(POOL_AVG) && defined(EXCLUDE_PADDING)
        }
    }

#if defined(POOL_AVG)
    res0 = (res0 + (VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE))(filter_size >> 1)) / filter_size;
#endif // defined(POOL_AVG)

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    out_q0 = CONVERT(res0, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    REQUANTIZE(VEC_SIZE, out_q0, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT, out_q0);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    // Store result
    STORE_VECTOR_SELECT(out_q, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, ((VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0));
}
#endif // defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)
#endif // defined(DATA_TYPE) && defined(INITIAL_VALUE)