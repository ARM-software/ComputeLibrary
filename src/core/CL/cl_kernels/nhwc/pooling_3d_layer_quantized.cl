/*
 * Copyright (c) 2022 Arm Limited.
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
#include "tile_helpers.h" // Needed for GET_SPATIAL_IDX()

#if defined(POOL_AVG)
#define POOL_OP(x, y) ((x) + (y))
#else /* defined(POOL_AVG)  */
#define POOL_OP(x, y) (max((x), (y)))
#endif /* defined(POOL_AVG) */

#define SQRT_OP(x) sqrt((x))

#if defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(SRC_DEPTH) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_DEPTH) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)

#if defined(POOL_SIZE_X) && defined(POOL_SIZE_Y) && defined(POOL_SIZE_Z)

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
#define VEC_FLOAT(VEC_SIZE) VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT(VEC_SIZE) VEC_DATA_TYPE(int, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
#define REQUANTIZE(VEC_SIZE, input, in_offset, out_offset, in_scale, out_scale, res)                                                                                 \
    {                                                                                                                                                                 \
        const VEC_FLOAT(VEC_SIZE) in_f32  = (CONVERT(input, VEC_FLOAT(VEC_SIZE)) - (VEC_FLOAT(VEC_SIZE))((float)in_offset)) * (VEC_FLOAT(VEC_SIZE))((float)in_scale); \
        const VEC_FLOAT(VEC_SIZE) out_f32 = in_f32 / ((VEC_FLOAT(VEC_SIZE))(float)out_scale) + ((VEC_FLOAT(VEC_SIZE))((float)out_offset));                            \
        res                               = CONVERT_SAT(CONVERT_DOWN(out_f32, VEC_INT(VEC_SIZE)), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));                                \
    }
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

#if defined(POOL_L2)
#error "L2 pooling is not supported"
#endif /* defined(POOL_L2) */

/** Performs 3d pooling layer of size equal to MxNXD. This OpenCL kernel can perform the following pooling types:
 * -# max, -DPOOL_MAX must be passed at compile time
 * -# average, -DPOOL_AVG must be passed at compile time. If padding has to be excluded, -DEXCLUDE_PADDING should be passed at compile time
 *
 * @note Datatype must be passed at compile type using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types are QASYMM8_SIGNED, QASYMM8
 * @note Accumulation data type must be passed at compile time using -DACC_DATA_TYPE e.g. -DACC_DATA_TYPE=float
 * @note If -DFP_MIXED_PRECISION is passed at compile time, the kernel will use F32 for the partial result
 * @note Pool size must be passed at compile time using -DPOOL_SIZE_X, -DPOOL_SIZE_Y, and -DPOOL_SIZE_Z. e.g. -DPOOL_SIZE_X=4, -DPOOL_SIZE_Y=4, -DPOOL_SIZE_Z=2
 * @note Input tensor width, height and depth must be passed at compile time using -DSRC_WIDTH, -DSRC_HEIGHT, and -DSRC_DEPTH
 * @note Output tensor height, channels, depth, and batch size must be passed at compile time using -DDST_HEIGHT, -DDST_CHANNELS, -DDST_DEPTH, and -DDST_BATCH_SIZE
 * @note Pool strides must be passed at compile time using -DSTRIDE_X, -DSTRIDE_Y and -DSTRIDE_Z which are the steps of the window along the x, y and z directions
 * @note Pool pads must be passed at compile time using -DPAD_X, -DPAD_Y, -DPAD_Z
 * @note Vector size must be passed at compile time using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size must be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The initial value for the pooling operation must be passed at compile time using -DINITIAL_VALUE e.g. -DINITIAL_VALUE=0
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8_SIGNED, QASYMM8
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in]  input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  input_stride_v                       Stride of the source tensor in V dimension (in bytes)
 * @param[in]  input_step_v                         input_stride_v * number of elements along V processed per workitem(in bytes)
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
 * @param[in]  output_stride_v                      Stride of the destination tensor in V dimension (in bytes)
 * @param[in]  output_step_v                        output_stride_v * number of elements along V processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void pooling_3d_layer_MxN_ndhwc_quantized(
    TENSOR5D_DECLARATION(input),
    TENSOR5D_DECLARATION(output))
{
    // Note: If C is not multiple of VEC_SIZE, we shift back of VEC_SIZE_LEFTOVER elements to compute the leftover elements for get_global_id(0) == 0
    // Note: If C is less than VEC_SIZE, VEC_SIZE should be shrunk to the closest smaller VEC_SIZE. This operation is performed on the host side
    int idx_out_c = GET_SPATIAL_IDX(0, VEC_SIZE, VEC_SIZE_LEFTOVER);
    int idx_out_w = GET_SPATIAL_IDX(1, 1, 0);

    // The depth size dimension and the batch size dimension are collapsed over the height dimension
    int idx_out_h = GET_SPATIAL_IDX(2, 1, 0) % DST_HEIGHT;
    int idx_out_d = (GET_SPATIAL_IDX(2, 1, 0) / DST_HEIGHT) % DST_DEPTH;
    int idx_out_n = (GET_SPATIAL_IDX(2, 1, 0) / DST_HEIGHT) / DST_DEPTH;

    __global unsigned char *in_base_ptr = input_ptr + input_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_n * input_stride_v;

    __global unsigned char *out_base_ptr = output_ptr + output_offset_first_element_in_bytes + idx_out_c * sizeof(DATA_TYPE) + idx_out_w * output_stride_y + idx_out_h * output_stride_z + idx_out_d *
                                           output_stride_w + idx_out_n * output_stride_v;

    VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
    res0 = INITIAL_VALUE;

    int idx_in_w = idx_out_w * STRIDE_X - (int)PAD_X;
    int idx_in_h = idx_out_h * STRIDE_Y - (int)PAD_Y;
    int idx_in_d = idx_out_d * STRIDE_Z - (int)PAD_Z;

    // The start of width to consider in calculation should exclude padding
    int pool_x_s = max((int)0, -idx_in_w);
    // Assumed Symmetric Padding (left padding = right padding = PAD_X), the filter end should be either the pool width or what is remaining from current pos to the (src width + pad right)
    int pool_x_e = min((int)POOL_SIZE_X, (int)SRC_WIDTH + PAD_X - idx_in_w);
    int pool_y_s = max((int)0, -idx_in_h);
    int pool_y_e = min((int)POOL_SIZE_Y, (int)SRC_HEIGHT + PAD_Y - idx_in_h);
    int pool_z_s = max((int)0, -idx_in_d);
    int pool_z_e = min((int)POOL_SIZE_Z, (int)SRC_DEPTH + PAD_Z - idx_in_d);

#if defined(POOL_AVG) && defined(EXCLUDE_PADDING)
    int filter_size = 0;
#elif defined(POOL_AVG) && !defined(EXCLUDE_PADDING) // defined(POOL_AVG) && defined(EXCLUDE_PADDING)
    int filter_size = pool_z_e * pool_y_e * pool_x_e;
#endif                                               // defined(POOL_AVG) && !defined(EXCLUDE_PADDING)

    // The end of width to consider in calculation should exclude PAD_X
    pool_x_e = min(pool_x_e, SRC_WIDTH - idx_in_w);
    pool_y_e = min(pool_y_e, SRC_HEIGHT - idx_in_h);
    pool_z_e = min(pool_z_e, SRC_DEPTH - idx_in_d);

    for(int z = pool_z_s; z < pool_z_e; ++z)
    {
        int depth_offset_src = (z + idx_in_d) * input_stride_w;
        for(int y = pool_y_s; y < pool_y_e; ++y)
        {
            int height_offset_src = (y + idx_in_h) * input_stride_z;
#pragma unroll 8
            for(int x = pool_x_s; x < pool_x_e; ++x)
            {
                int width_offset_src = (x + idx_in_w) * input_stride_y;

                VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
                data;
                VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE)
                data0;

                data  = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(in_base_ptr + width_offset_src + height_offset_src + depth_offset_src));
                data0 = CONVERT(data, VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE));

                res0 = POOL_OP(res0, data0);

#if defined(POOL_AVG) && defined(EXCLUDE_PADDING)
                filter_size++;
#endif // defined(POOL_AVG) && defined(EXCLUDE_PADDING)
            }
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

    STORE_VECTOR_SELECT(out_q, DATA_TYPE, out_base_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, (VEC_SIZE_LEFTOVER != 0) && get_global_id(0) == 0);
}
#endif // defined(POOL_SIZE_X) && defined(POOL_SIZE_Y) && defined(POOL_SIZE_Z)
#endif // defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(SRC_DEPTH) && defined(DST_CHANNELS) && defined(DST_HEIGHT) && defined(DST_DEPTH) && defined(DST_BATCH_SIZE) && defined(ACC_DATA_TYPE)
