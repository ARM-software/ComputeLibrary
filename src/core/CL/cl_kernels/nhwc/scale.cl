/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "tile_helpers.h"

//! @cond Doxygen_Suppress
/** Performs scale on a tensor by interpolating with the NEAREAST NEIGHBOUR method. (NHWC)
 *
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The tensor type ("BUFFER" only is supported) of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" only is supported) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The border value value must be passed at compile time using -DCONSTANT_VALUE (e.g. -DCONSTANT_VALUE=0)
 * @note In case of F32/F16, -DIS_FLOATING_POINT must be passed at compile time
 * @note The scale value to apply on the source width must be passed at compile using -DSCALE_X (e.g., -DSCALE_X=0.5)
 * @note The scale value to apply on the source height must be passed at compile using -DSCALE_Y (e.g., -DSCALE_Y=0.5)
 * @note If the source tensor has more than 3 dimensions, -DBATCHED_EXECUTION must be passed at compile time
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32.
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
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
 //! @endcond
__kernel void scale_nearest_neighbour_nhwc(
    TENSOR4D(src, SRC_TENSOR_TYPE),
    TENSOR4D(dst, DST_TENSOR_TYPE))
{
    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _ISCALE_X SCALE_X
#define _ISCALE_Y SCALE_Y

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, 1, 0);           // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % _IDST_HEIGHT; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / _IDST_HEIGHT; // BATCH SIZE IDX
#else                                                         // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0; // BATCH SIZE IDX
#endif                                                        // defined(BATCHED_EXECUTION)

#ifdef SAMPLING_POLICY_TOP_LEFT
    float xi_f = (xo * (float)SCALE_X);
    float yi_f = (yo * (float)SCALE_Y);
#elif SAMPLING_POLICY_CENTER
    float     xi_f = ((xo + 0.5f) * (float)SCALE_X);
    float     yi_f = ((yo + 0.5f) * (float)SCALE_Y);
#else // SAMPLING_POLICY
#error("Unsupported sampling policy");
#endif // SAMPLING_POLICY

#ifdef ALIGN_CORNERS
    xi_f = round(xi_f);
    yi_f = round(yi_f);
#endif // ALIGN_CORNERS

    const int xi0 = clamp((int)xi_f, 0, _ISRC_WIDTH - 1);
    const int yi0 = clamp((int)yi_f, 0, _ISRC_HEIGHT - 1);

    TILE(SRC_DATA_TYPE, 1, N0, in00);

    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi0, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, false, in00);

    TILE(uint, 1, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    dst_indirect_y[0].v = xo + (yo * (int)(_IDST_WIDTH)) + bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, 1, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, in00, dst_indirect_y);
}

//! @cond Doxygen_Suppress
/** Performs scale on a tensor by interpolating with the BILINEAR method. (NHWC)
 *
 * @note If border mode replicate is used, is should be passed as -DBORDER_MODE_REPLICATE
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The tensor type ("BUFFER" only is supported) of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" only is supported) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The border value value must be passed at compile time using -DCONSTANT_VALUE (e.g. -DCONSTANT_VALUE=0)
 * @note In case of F32/F16, -DIS_FLOATING_POINT must be passed at compile time
 * @note The scale value to apply on the source width must be passed at compile using -DSCALE_X (e.g., -DSCALE_X=0.5)
 * @note The scale value to apply on the source height must be passed at compile using -DSCALE_Y (e.g., -DSCALE_Y=0.5)
 * @note If the source tensor has more than 3 dimensions, -DBATCHED_EXECUTION must be passed at compile time
 *
 * @note In case of QASYMM8, the following extra information must be passed at compile time:
 * - The source offset e.g. -DOFFSET=4
 * - The source scale e.g. -DSCALE=4
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32.
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
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
 //! @endcond
__kernel void scale_bilinear_nhwc(
    TENSOR4D(src, SRC_TENSOR_TYPE),
    TENSOR4D(dst, DST_TENSOR_TYPE))
{
    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _ISCALE_X SCALE_X
#define _ISCALE_Y SCALE_Y

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, 1, 0);           // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % _IDST_HEIGHT; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / _IDST_HEIGHT; // BATCH SIZE IDX
#else                                                         // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0;                        // BATCH SIZE IDX
#endif                                                        // defined(BATCHED_EXECUTION)

#ifdef SAMPLING_POLICY_TOP_LEFT
    float xi_f = (xo * (float)SCALE_X);
    float yi_f = (yo * (float)SCALE_Y);
#elif SAMPLING_POLICY_CENTER
    float     xi_f = ((xo + 0.5f) * (float)SCALE_X - 0.5f);
    float     yi_f = ((yo + 0.5f) * (float)SCALE_Y - 0.5f);
#else // SAMPLING_POLICY
#error("Unsupported sampling policy");
#endif // SAMPLING_POLICY

    const int xi = (int)floor(xi_f);
    const int yi = (int)floor(yi_f);

    TILE(SRC_DATA_TYPE, 1, N0, in00);
    TILE(SRC_DATA_TYPE, 1, N0, in01);
    TILE(SRC_DATA_TYPE, 1, N0, in10);
    TILE(SRC_DATA_TYPE, 1, N0, in11);

    // Initialize the tiles to CONSTANT_VALUE
    in00[0].v = CONSTANT_VALUE;
    in01[0].v = CONSTANT_VALUE;
    in10[0].v = CONSTANT_VALUE;
    in11[0].v = CONSTANT_VALUE;

#ifndef BORDER_MODE_REPLICATE
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi, xi, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, true, in00);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi, xi + 1, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, true, in01);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi + 1, xi, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, true, in10);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi + 1, xi + 1, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, true, in11);
#else  // BORDER_MODE_REPLICATE
    const int xi0  = clamp(xi, 0, _ISRC_WIDTH - 1);
    const int yi0  = clamp(yi, 0, _ISRC_HEIGHT - 1);
    const int xi1  = clamp(xi + 1, 0, _ISRC_WIDTH - 1);
    const int yi1  = clamp(yi + 1, 0, _ISRC_HEIGHT - 1);

    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi0, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, false, in00);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi1, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, false, in01);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi1, xi0, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, false, in10);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi1, xi1, cout, _ISRC_WIDTH, _ISRC_HEIGHT, 1, 1, false, in11);
#endif // BORDER_MODE_REPLICATE

    TILE(DST_DATA_TYPE, 1, N0, out);

#if defined(IS_FLOATING_POINT)
    const SRC_DATA_TYPE a  = (SRC_DATA_TYPE)(xi_f - (float)xi);
    const SRC_DATA_TYPE b  = (SRC_DATA_TYPE)(1.f - a);
    const SRC_DATA_TYPE a1 = (SRC_DATA_TYPE)(yi_f - (float)yi);
    const SRC_DATA_TYPE b1 = (SRC_DATA_TYPE)(1.f - a1);

    // Calculate the output
    out[0].v = ((in00[0].v * b * b1) + (in01[0].v * a * b1) + (in10[0].v * b * a1) + (in11[0].v * a * a1));
#else  // defined(IS_FLOATING_POINT)
    TILE(float, 1, N0, out_f);
    TILE(float, 1, N0, in00_f);
    TILE(float, 1, N0, in01_f);
    TILE(float, 1, N0, in10_f);
    TILE(float, 1, N0, in11_f);

    const float a  = (xi_f - (float)xi);
    const float b  = (1.f - a);
    const float a1 = (yi_f - (float)yi);
    const float b1 = (1.f - a1);

    // Dequantize
    LOOP_UNROLLING(int, n0, 0, 1, N0,
    {
        in00_f[0].s[n0] = ((float)in00[0].s[n0] - (float)OFFSET) * (float)SCALE;
        in01_f[0].s[n0] = ((float)in01[0].s[n0] - (float)OFFSET) * (float)SCALE;
        in10_f[0].s[n0] = ((float)in10[0].s[n0] - (float)OFFSET) * (float)SCALE;
        in11_f[0].s[n0] = ((float)in11[0].s[n0] - (float)OFFSET) * (float)SCALE;
    })

    // Calculate the output in the floating-point domain
    out_f[0].v = ((in00_f[0].v * b * b1) + (in01_f[0].v * a * b1) + (in10_f[0].v * b * a1) + (in11_f[0].v * a * a1));

    // Quantize
    LOOP_UNROLLING(int, n0, 0, 1, N0,
    {
        out[0].s[n0] = CONVERT_SAT(out_f[0].s[n0] / (float)SCALE + (float)OFFSET, DST_DATA_TYPE);
    })
#endif // defined(IS_FLOATING_POINT)

    TILE(uint, 1, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    dst_indirect_y[0].v = xo + (yo * (int)(_IDST_WIDTH)) + bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, 1, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, out, dst_indirect_y);
}