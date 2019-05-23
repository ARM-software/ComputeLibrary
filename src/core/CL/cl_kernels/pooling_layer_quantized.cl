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

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
#define VEC_FLOAT(VEC_SIZE) \
    VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT(VEC_SIZE) VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_UCHAR(VEC_SIZE) VEC_DATA_TYPE(uchar, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
#define REQUANTIZE(VEC_SIZE, input, in_offset, out_offset, in_scale, out_scale, res)                                                                                  \
    {                                                                                                                                                                 \
        const VEC_FLOAT(VEC_SIZE) in_f32  = (CONVERT(input, VEC_FLOAT(VEC_SIZE)) - (VEC_FLOAT(VEC_SIZE))((float)in_offset)) * (VEC_FLOAT(VEC_SIZE))((float)in_scale); \
        const VEC_FLOAT(VEC_SIZE) out_f32 = in_f32 / ((VEC_FLOAT(VEC_SIZE))(float)out_scale) + ((VEC_FLOAT(VEC_SIZE))((float)out_offset));                            \
        res                               = CONVERT_SAT(CONVERT_DOWN(out_f32, VEC_INT(VEC_SIZE)), VEC_UCHAR(VEC_SIZE));                                               \
    }
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

#if defined(POOL_AVG)
#define POOL_OP(x, y) ((x) + (y))
#else /* defined(POOL_AVG) */
#define POOL_OP(x, y) (max((x), (y)))
#endif /* defined(POOL_AVG) */

#define DIV_OP(x, y) (x * (1.f / y))

#define DIV_OP_NHWC(x, y) (convert_float8(x) * (float8)(1.f / y))

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
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8
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

    int8 vdata = 0;
    int  sdata = 0;

    // Load data
    for(int y = 0; y < POOL_SIZE_Y; y++)
    {
        int x = 0;
        for(; x <= ((int)POOL_SIZE_X - 8); x += 8)
        {
            uchar8 data = vload8(0, (__global uchar *)tensor3D_offset(&input, x, y, 0));
            int8 data0  = convert_int8(data);
            vdata       = POOL_OP(vdata, data0);
        }

        // Leftover
        for(; x < (int)POOL_SIZE_X; ++x)
        {
            uchar data = *((__global uchar *)tensor3D_offset(&input, x, y, 0));
            int data0  = convert_int(data);
            sdata      = POOL_OP(sdata, data0);
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

    uchar result_u8 = convert_uchar(res);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)

    const float result_f32   = convert_float(result_u8);
    const float input_offset = (float)OFFSET_IN1;
    const float input_scale  = (float)SCALE_IN1;
    const float scale_out    = (float)SCALE_OUT;
    const float offset_out   = (float)OFFSET_OUT;
    const float in_f32       = (result_f32 - input_offset) * input_scale;
    const float out_f32      = in_f32 / scale_out + offset_out;
    result_u8                = convert_uchar_sat(convert_int_rte(out_f32));

#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    *(__global uchar *)output.ptr = result_u8;
}

int calculate_avg_scale_nhwc(const int pool_size_x, const int pool_size_y, int upper_bound_w, int upper_bound_h,
                             const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = get_global_id(1) * stride_x - pad_x;
#if defined(DST_DEPTH)
    int start_y = (get_global_id(2) % DST_DEPTH) * stride_y - pad_y;
#else  /* defined(DST_DEPTH) */
    int            start_y    = get_global_id(2) * stride_y - pad_y;
#endif /* defined(DST_DEPTH) */

    const int end_x = min(start_x + pool_size_x, upper_bound_w);
    const int end_y = min(start_y + pool_size_y, upper_bound_h);

    start_x = max(0, start_x);
    start_y = max(0, start_y);

    return ((end_y - start_y) * (end_x - start_x));
}

/** Performs a pooling function of pool size equal to N (NHWC)
 *
 * @note Pool sizes must be passed using -DPOOL_SIZE_X and -DPOOL_SIZE_Y e.g. -DPOOL_SIZE_X=13;
 * @note Tensors width and height must be passed at compile time using -DMAX_WIDTH and -DMAX_HEIGHT
 * @note Strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y which are the steps of the window along the x and y directions
 * @note Pad values must be passed at compile time using -DPAD_X and -DPAD_Y which are the pooling paddings in x and y dimension
 * @note In case of average pooling the following information must be passed at compile time:
 *       -DPOOL_AVG must be provided otherwise max pooling will be performed.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8
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
    // Get pixels pointer
#if defined(DST_DEPTH)
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT(input, DST_DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DST_DEPTH);
#else  /* defined(DST_DEPTH) */
    Tensor3D       input      = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D       output     = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* defined(DST_DEPTH) */

    int8 vdata = 0;

    const int idx_width = get_global_id(1) * STRIDE_X;
#if defined(DST_DEPTH)
    const int idx_height = (get_global_id(2) % DST_DEPTH) * STRIDE_Y;
#else  /* defined(DST_DEPTH) */
    const int      idx_height = get_global_id(2) * STRIDE_Y;
#endif /* defined(DST_DEPTH) */

    for(int y = 0; y < POOL_SIZE_Y; ++y)
    {
        int y1 = select(y, PAD_Y - idx_height, y + idx_height - PAD_Y < 0 || y + idx_height - PAD_Y >= MAX_HEIGHT);
        for(int x = 0; x < POOL_SIZE_X; ++x)
        {
            int x1 = select(x, PAD_X - idx_width - 1, x + idx_width - PAD_X < 0 || x + idx_width - PAD_X >= MAX_WIDTH);
            x1     = select(x1, PAD_X - idx_width - 1, y != y1);

#if defined(DST_DEPTH)
            uchar8 data = vload8(0, (__global uchar *)tensor4D_offset(&input, 0, x1 - PAD_X, y1 - PAD_Y, 0));
#else  /* defined(DST_DEPTH) */
            uchar8 data       = vload8(0, (__global uchar *)tensor3D_offset(&input, 0, x1 - PAD_X, y1 - PAD_Y));
#endif /* defined(DST_DEPTH) */

            int8 data0 = convert_int8(data);
            vdata      = POOL_OP(vdata, data0);
        }
    }

#if defined(POOL_AVG)
    // Divide by pool region in case of average pooling
    vdata = convert_int8(round(DIV_OP_NHWC(vdata, calculate_avg_scale_nhwc(POOL_SIZE_X, POOL_SIZE_Y, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y))));
#endif /* defined(POOL_AVG) */

    uchar8 out_u8 = convert_uchar8(vdata);
#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    REQUANTIZE(8, out_u8, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT, out_u8);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    // Store result
    vstore8(out_u8, 0, (__global uchar *)output.ptr);
}
