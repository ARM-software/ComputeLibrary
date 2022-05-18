/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "activation_float_helpers.h"
#include "helpers.h"
#include "tile_helpers.h"

#if defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)
#if defined(VEC_SIZE) && VEC_SIZE == 2
#if defined(WINOGRAD_OUTPUT_TRANSFORM_2X2_7X7_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_2X1_7X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X2_1X7_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 2x2/2x1 or 1x2, the filter size 7x7/7x1 or 1x7 and the data layout is NHWC
 *
 * @note  must be passed at compile time using -DNUM_TILES_X: e.g. -DNUM_TILES_X=16
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd output transform 7x1, -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd output transform 1x7, -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 * @note The number of output elements processed along the X direction must be passed at compile time using -DN0 e.g. -DN0=1
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  _ISRC_HEIGHT                      The source tensor's height
 * @param[in]  _IDST_WIDTH                       The destination tensor's width
 * @param[in]  _IDST_HEIGHT                      The destination tensor's height
 * @param[in]  _INUM_TILES_X                     The number of tiles along the X direction
 */
__kernel void winograd_output_transform_2x2_7x7_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int _ISRC_HEIGHT,
    const int _IDST_WIDTH,
    const int _IDST_HEIGHT,
    const int _INUM_TILES_X)
{
    const int cout = GET_SPATIAL_IDX(0, N0, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0);  // WINOGRAD OUTPUT TILES
    const int bout = GET_SPATIAL_IDX(2, 1, 0);  // BATCH SIZE IDX

    int x_out = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y_out = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;

#if defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    TILE(DATA_TYPE, 8, N0, in);
    TILE(DATA_TYPE, 2, N0, out);
    TILE(uint, 8, 1, src_indirect_y);

    // Calculate the indirect Y for the source tensor
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        src_indirect_y[i].v = mout + i *_ISRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(_ISRC_HEIGHT * 8);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

    // Load the values across the 8 channels to compose the 8x1 tile
    T_LOAD_INDIRECT(DATA_TYPE, 8, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    // Compute out0 and out01
    out[0].v = in[0].v + in[1].v + in[2].v + in[3].v + in[4].v + in[5].v + in[6].v;
    out[1].v = -in[1].v + in[2].v - 2.f * in[3].v + 2.0f * in[4].v - 3.0f * in[5].v + 3.0f * in[6].v + in[7].v;

#if defined(HAS_BIAS)
    // Add bias
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    T_ADD_BROADCAST_X(DATA_TYPE, 2, N0, out, b, out);
#endif // defined(HAS_BIAS)

    T_ACTIVATION(DATA_TYPE, 2, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 2, 1, dst_indirect_y);

#if defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, yk, 0, 1, 2,
    {
        int y_c              = min(y_out + yk, ((int)_IDST_HEIGHT - 1));
        dst_indirect_y[yk].v = x_out + y_c * (int)(_IDST_WIDTH);
    })
#else  // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, xk, 0, 1, 2,
    {
        int x_c              = min(x_out + xk, ((int)_IDST_WIDTH - 1));
        dst_indirect_y[xk].v = x_c + y_out * (int)(_IDST_WIDTH);
    })
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 2, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 64, N0, in);
    TILE(DATA_TYPE, 4, N0, out);
    TILE(DATA_TYPE, 16, N0, tmp);
    TILE(uint, 64, 1, src_indirect_y);

    // Calculate the indirect Y for the source tensor
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        src_indirect_y[i].v = mout + i *_ISRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(_ISRC_HEIGHT * 64);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // Load the values across the 64 channels to compose the 8x8 tile
    T_LOAD_INDIRECT(DATA_TYPE, 64, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        tmp[i * 2].v     = in[0 + i].v + in[8 + i].v + in[16 + i].v + in[24 + i].v + in[32 + i].v + in[40 + i].v + in[48 + i].v;
        tmp[i * 2 + 1].v = -in[8 + i].v + in[16 + i].v - 2 * in[24 + i].v + 2 * in[32 + i].v + -3 * in[40 + i].v + 3 * in[48 + i].v + in[56 + i].v;
    })

    // Compute the 2x2 output tile
    LOOP_UNROLLING(int, i, 0, 1, 2,
    {
        out[i * 2].v     = tmp[0 + i].v + tmp[2 + i].v + tmp[4 + i].v + tmp[6 + i].v + tmp[8 + i].v + tmp[10 + i].v + tmp[12 + i].v;
        out[i * 2 + 1].v = -tmp[2 + i].v + tmp[4 + i].v - 2 * tmp[6 + i].v + 2 * tmp[8 + i].v - 3 * tmp[10 + i].v + 3 * tmp[12 + i].v + tmp[14 + i].v;
    })

#if defined(HAS_BIAS)
    // Add bias
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    T_ADD_BROADCAST_X(DATA_TYPE, 4, N0, out, b, out);
#endif // defined(HAS_BIAS)

    T_ACTIVATION(DATA_TYPE, 4, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 4, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, yk, 0, 1, 2,
    {
        LOOP_UNROLLING(int, xk, 0, 1, 2,
        {
            int x_c                       = min(x_out + xk, ((int)_IDST_WIDTH - 1));
            int y_c                       = min(y_out + yk, ((int)_IDST_HEIGHT - 1));
            dst_indirect_y[xk + yk * 2].v = x_c + y_c *_IDST_WIDTH;
            dst_indirect_y[xk + yk * 2].v += bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);
        })
    })

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 4, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);
#endif // !defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_2X2_7X7_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_2X1_7X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X2_1X7_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 2

#if defined(VEC_SIZE) && VEC_SIZE == 4
#if defined(WINOGRAD_OUTPUT_TRANSFORM_4X4_3X3_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_3X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X3_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4, 4x1 or 1x4, the filter size 3x3, 3x1 or 1x3 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd output transform 3x1, -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd output transform 1x3, -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 * @note The number of output elements processed along the X direction must be passed at compile time using -DN0 e.g. -DN0=1
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  dst_size                          Size of the destination tensor, minus the last padding
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_4x4_3x3_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    const int cout = GET_SPATIAL_IDX(0, N0, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0);  // WINOGRAD OUTPUT TILES
    const int bout = GET_SPATIAL_IDX(2, 1, 0);  // BATCH SIZE IDX

#if defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 6, N0, in);
    TILE(DATA_TYPE, 4, N0, out);
    TILE(uint, 6, 1, src_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        src_indirect_y[i].v = mout + i *SRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(SRC_HEIGHT * 6);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        in[i].v = 0;
    })

    // Load the values across the 36 channels to compose the 6x6 or 6x1 tile
    T_LOAD_INDIRECT(DATA_TYPE, 6, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    // Compute out00, out01, out02 and out03
    out[0].v = in[0].v + in[1].v + in[2].v + in[3].v + in[4].v;
    out[1].v = in[1].v - in[2].v + 2.0f * in[3].v - 2.0f * in[4].v;
    out[2].v = in[1].v + in[2].v + 4.0f * in[3].v + 4.0f * in[4].v;
    out[3].v = in[1].v - in[2].v + 8.0f * in[3].v - 8.0f * in[4].v + in[5].v;

#if defined(HAS_BIAS)
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, 4, N0, out, b, out);
#endif // HAS_BIAS

    int x_out = (mout % NUM_TILES_X) * OUTPUT_TILE_W;
    int y_out = (mout / NUM_TILES_X) * OUTPUT_TILE_H;

    T_ACTIVATION(DATA_TYPE, 4, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 4, 1, dst_indirect_y);

    // Calculate the destination indirect Y
#if defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, yk, 0, 1, 4,
    {
        int y_c              = min(y_out + yk, ((int)DST_HEIGHT - 1));
        dst_indirect_y[yk].v = x_out + y_c *DST_WIDTH;
        dst_indirect_y[yk].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
    })
#else  // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, xk, 0, 1, 4,
    {
        int x_c              = min(x_out + xk, ((int)DST_WIDTH - 1));
        dst_indirect_y[xk].v = x_c + y_out *DST_WIDTH;
        dst_indirect_y[xk].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
    })
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 4, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    // Calculate the indirect Y for the source tensor
    TILE(DATA_TYPE, 36, N0, in);
    TILE(DATA_TYPE, 4, N0, tmp);
    TILE(uint, 36, 1, src_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        src_indirect_y[i].v = mout + i *SRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(SRC_HEIGHT * 36);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        in[i].v = 0;
    })

    // Load the values across the 36 channels to compose the 6x6 or 6x1 tile
    T_LOAD_INDIRECT(DATA_TYPE, 36, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        tmp[0].v     = in[6 + i].v + in[12 + i].v;
        tmp[1].v     = in[6 + i].v - in[12 + i].v;
        tmp[2].v     = in[18 + i].v + in[24 + i].v;
        tmp[3].v     = in[18 + i].v - in[24 + i].v;
        tmp[3].v     = tmp[3].v + tmp[3].v;
        in[i].v      = in[i].v + tmp[0].v + tmp[2].v;
        in[6 + i].v  = tmp[3].v + tmp[1].v;
        in[12 + i].v = fma(tmp[2].v, (VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[0].v);
        in[18 + i].v = fma(tmp[3].v, (VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[1].v) + in[30 + i].v;
    })

    // Compute the output tile
    TILE(DATA_TYPE, 16, N0, out);

    LOOP_UNROLLING(int, i, 0, 1, 4,
    {
        tmp[0].v         = in[6 * i + 1].v + in[6 * i + 2].v;
        tmp[1].v         = in[6 * i + 1].v - in[6 * i + 2].v;
        tmp[2].v         = in[6 * i + 3].v + in[6 * i + 4].v;
        tmp[3].v         = in[6 * i + 3].v - in[6 * i + 4].v;
        tmp[3].v         = tmp[3].v + tmp[3].v;
        out[4 * i + 0].v = in[6 * i + 0].v + tmp[0].v + tmp[2].v;
        out[4 * i + 1].v = tmp[3].v + tmp[1].v;
        out[4 * i + 2].v = fma(tmp[2].v, (VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[0].v);
        out[4 * i + 3].v = fma(tmp[3].v, (VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[1].v) + in[6 * i + 5].v;
    })

#if defined(HAS_BIAS)
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, 16, N0, out, b, out);
#endif // HAS_BIAS

    int x_out = (mout % NUM_TILES_X) * OUTPUT_TILE_W;
    int y_out = (mout / NUM_TILES_X) * OUTPUT_TILE_H;

    T_ACTIVATION(DATA_TYPE, 16, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 16, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, yk, 0, 1, 4,
    {
        LOOP_UNROLLING(int, xk, 0, 1, 4,
        {
            int x_c                       = min(x_out + xk, ((int)DST_WIDTH - 1));
            int y_c                       = min(y_out + yk, ((int)DST_HEIGHT - 1));
            dst_indirect_y[xk + yk * 4].v = x_c + y_c *DST_WIDTH;
            dst_indirect_y[xk + yk * 4].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
        })
    })

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 16, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_4X4_3X3_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_3X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X3_NHWC)

#if defined(WINOGRAD_OUTPUT_TRANSFORM_4X4_5X5_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_5X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X5_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4/4x1 or 1x4, the filter size 5x5/5x1 or 1x5 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd output transform 5x1, -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd output transform 1x5, -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 * @note The number of output elements processed along the X direction must be passed at compile time using -DN0 e.g. -DN0=1
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_4x4_5x5_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    const int cout = GET_SPATIAL_IDX(0, N0, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0);  // WINOGRAD OUTPUT TILES
    const int bout = GET_SPATIAL_IDX(2, 1, 0);  // BATCH SIZE IDX

#if defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    TILE(DATA_TYPE, 8, N0, in);
    TILE(DATA_TYPE, 4, N0, out);
    TILE(DATA_TYPE, 4, N0, tmp);
    TILE(uint, 8, 1, src_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        src_indirect_y[i].v = mout + i *SRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(SRC_HEIGHT * 8);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

    // "in" contains 1x8 or 8x1 tile here
    T_LOAD_INDIRECT(DATA_TYPE, 8, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    // A^T * in, and in this degenerate case out consists of 1 column/row
    tmp[0].v = in[1].v - in[2].v;
    tmp[1].v = 2.0f * (in[3].v - in[4].v);
    tmp[2].v = 2.0f * (in[5].v + in[6].v);
    tmp[3].v = in[3].v + in[4].v;
    out[0].v = in[0].v + in[1].v + in[2].v + tmp[3].v + 4.0f * tmp[2].v;
    out[1].v = tmp[0].v + tmp[1].v + 4.0f * (in[5].v - in[6].v);
    out[2].v = in[1].v + in[2].v + 4.0f * tmp[3].v + tmp[2].v;
    out[3].v = tmp[0].v + 4.0f * tmp[1].v + in[5].v - in[6].v + in[7].v;

#if defined(HAS_BIAS)
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, 4, N0, out, b, out);
#endif // HAS_BIAS

    int x_out = (mout % NUM_TILES_X) * OUTPUT_TILE_W;
    int y_out = (mout / NUM_TILES_X) * OUTPUT_TILE_H;

    T_ACTIVATION(DATA_TYPE, 4, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 4, 1, dst_indirect_y);

    // Calculate the destination indirect Y
#if defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, yk, 0, 1, 4,
    {
        int y_c              = min(y_out + yk, ((int)DST_HEIGHT - 1));
        dst_indirect_y[yk].v = x_out + y_c *DST_WIDTH;
        dst_indirect_y[yk].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
    })
#else  // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    LOOP_UNROLLING(int, xk, 0, 1, 4,
    {
        int x_c              = min(x_out + xk, ((int)DST_WIDTH - 1));
        dst_indirect_y[xk].v = x_c + y_out *DST_WIDTH;
        dst_indirect_y[xk].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
    })
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 4, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
    // Calculate the indirect Y for the source tensor
    TILE(DATA_TYPE, 64, N0, in);
    TILE(DATA_TYPE, 6, N0, tmp);
    TILE(uint, 64, 1, src_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        src_indirect_y[i].v = mout + i *SRC_HEIGHT;
        src_indirect_y[i].v += bout * (int)(SRC_HEIGHT * 64);
    })

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // "in" here is 8x8 tile
    T_LOAD_INDIRECT(DATA_TYPE, 64, N0, BUFFER, src, cout, src_stride_y, src_indirect_y, in);

    // A^T * in
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        tmp[0].v = in[8 + i].v + in[16 + i].v;
        tmp[1].v = in[8 + i].v - in[16 + i].v;
        tmp[2].v = in[24 + i].v + in[32 + i].v;
        tmp[3].v = in[24 + i].v - in[32 + i].v;
        tmp[3].v = tmp[3].v + tmp[3].v;
        tmp[4].v = in[40 + i].v + in[48 + i].v;
        tmp[4].v = tmp[4].v + tmp[4].v;
        tmp[5].v = in[40 + i].v - in[48 + i].v;

        // 4x8 matrix as a result
        in[i].v      = in[i].v + tmp[0].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[4].v, tmp[2].v);
        in[8 + i].v  = tmp[1].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[5].v, tmp[3].v);
        in[16 + i].v = tmp[0].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[2].v, tmp[4].v);
        in[24 + i].v = tmp[1].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[3].v, tmp[5].v) + in[56 + i].v;
    })

    // Compute the output tile
    TILE(DATA_TYPE, 16, N0, out);

    // in * A, with in = A^T * in as above
    LOOP_UNROLLING(int, i, 0, 1, 4,
    {
        tmp[0].v = in[8 * i + 1].v + in[8 * i + 2].v;
        tmp[1].v = in[8 * i + 1].v - in[8 * i + 2].v;
        tmp[2].v = in[8 * i + 3].v + in[8 * i + 4].v;
        tmp[3].v = in[8 * i + 3].v - in[8 * i + 4].v;
        tmp[3].v = tmp[3].v + tmp[3].v;
        tmp[4].v = in[8 * i + 5].v + in[8 * i + 6].v;
        tmp[4].v = tmp[4].v + tmp[4].v;
        tmp[5].v = in[8 * i + 5].v - in[8 * i + 6].v;

        // 4x4 tile
        out[4 * i].v     = in[8 * i].v + tmp[0].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[4].v, tmp[2].v);
        out[4 * i + 1].v = tmp[1].v + fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[5].v, tmp[3].v);
        out[4 * i + 2].v = fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[2].v, tmp[0].v) + tmp[4].v;
        out[4 * i + 3].v = fma((VEC_DATA_TYPE(DATA_TYPE, N0))4.0f, tmp[3].v, tmp[1].v) + tmp[5].v + in[8 * i + 7].v;
    })

#if defined(HAS_BIAS)
    TILE(DATA_TYPE, 1, N0, b);

    T_LOAD(DATA_TYPE, 1, N0, BUFFER, bias, cout, 0, 1, 0, b);

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, 16, N0, out, b, out);
#endif // HAS_BIAS

    int x_out = (mout % NUM_TILES_X) * OUTPUT_TILE_W;
    int y_out = (mout / NUM_TILES_X) * OUTPUT_TILE_H;

    T_ACTIVATION(DATA_TYPE, 16, N0, ACTIVATION_TYPE, A_VAL, B_VAL, out, out);

    TILE(uint, 16, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, yk, 0, 1, 4,
    {
        LOOP_UNROLLING(int, xk, 0, 1, 4,
        {
            int x_c                       = min(x_out + xk, ((int)DST_WIDTH - 1));
            int y_c                       = min(y_out + yk, ((int)DST_HEIGHT - 1));
            dst_indirect_y[xk + yk * 4].v = x_c + y_c *DST_WIDTH;
            dst_indirect_y[xk + yk * 4].v += bout * (int)(DST_WIDTH * DST_HEIGHT);
        })
    })

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 16, N0, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_4X4_5X5_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_5X1_NHWC) || defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X5_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 4

#if defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL)
#if defined(VEC_SIZE) && VEC_SIZE == 2
#if defined(WINOGRAD_OUTPUT_TRANSFORM_2X1_7X1_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 2x1, the filter size 7x1 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_2x1_7x1_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_2x2_7x7_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_2X1_7X1_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 2

#if defined(VEC_SIZE) && VEC_SIZE == 4
#if defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_3X1_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 4x1, the filter size 3x1 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_4x1_3x1_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_4x4_3x3_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_3X1_NHWC)

#if defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_5X1_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 4x1, the filter size 5x1 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_4x1_5x1_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_4x4_5x5_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_4X1_5X1_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 4
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL)

#if defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
#if defined(VEC_SIZE) && VEC_SIZE == 2
#if defined(WINOGRAD_OUTPUT_TRANSFORM_1X2_1X7_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 1x2, the filter size 1x7 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_1x2_1x7_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_2x2_7x7_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_1X2_1X7_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 2

#if defined(VEC_SIZE) && VEC_SIZE == 4
#if defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X3_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 1x4, the filter size 1x3 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_1x4_1x3_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_4x4_3x3_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X3_NHWC)

#if defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X5_NHWC)
/** This OpenCL kernel performs Winograd output transform when the output tile is 1x4, the filter size 1x5 and the data layout is NHWC
 *
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note The width of the output tensor must be passed at compile time using -DDST_WIDTH: e.g. -DDST_WIDTH=24
 * @note The height of the output tensor must be passed at compile time using -DDST_HEIGHT: e.g. -DDST_HEIGHT=32
 * @note -DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32/F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  SRC_HEIGHT                        The source tensor's height
 * @param[in]  DST_WIDTH                         The destination tensor's width
 * @param[in]  DST_HEIGHT                        The destination tensor's height
 * @param[in]  NUM_TILES_X                       The number of tiles along the X direction
 */
__kernel void winograd_output_transform_1x4_1x5_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int       dst_size,
    const int SRC_HEIGHT,
    const int DST_WIDTH,
    const int DST_HEIGHT,
    const int NUM_TILES_X)
{
    winograd_output_transform_4x4_5x5_nhwc(src_ptr,
                                           src_stride_x,
                                           src_step_x,
                                           src_stride_y,
                                           src_step_y,
                                           src_stride_z,
                                           src_step_z,
                                           src_stride_w,
                                           src_step_w,
                                           src_offset_first_element_in_bytes,
                                           dst_ptr,
                                           dst_stride_x,
                                           dst_step_x,
                                           dst_stride_y,
                                           dst_step_y,
                                           dst_stride_z,
                                           dst_step_z,
                                           dst_stride_w,
                                           dst_step_w,
                                           dst_offset_first_element_in_bytes,
#if defined(HAS_BIAS)
                                           bias_ptr,
                                           bias_stride_x,
                                           bias_step_x,
                                           bias_offset_first_element_in_bytes,
#endif // defined(HAS_BIAS)
                                           dst_size,
                                           SRC_HEIGHT,
                                           DST_WIDTH,
                                           DST_HEIGHT,
                                           NUM_TILES_X);
}
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_1X4_1X5_NHWC)
#endif // defined(VEC_SIZE) && VEC_SIZE == 4
#endif // defined(WINOGRAD_OUTPUT_TRANSFORM_VERTICAL)
#endif // defined(NUM_TILES_X) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)