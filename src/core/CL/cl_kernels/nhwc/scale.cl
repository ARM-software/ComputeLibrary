/*
 * Copyright (c) 2016-2022 Arm Limited.
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

#if defined(SCALE_NEAREST_NEIGHBOUR)
//! @cond Doxygen_Suppress
/** Performs scale on a tensor by interpolating with the NEAREAST NEIGHBOUR method. (NHWC)
 *
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note The tensor type ("BUFFER" only is supported) of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" only is supported) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The border value value must be passed at compile time using -DCONSTANT_VALUE (e.g. -DCONSTANT_VALUE=0)
 * @note In case of F32/F16, -DIS_FLOATING_POINT must be passed at compile time
 * @note If the source tensor has more than 3 dimensions, -DBATCHED_EXECUTION must be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32.
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_c                             The size of the channels dimension of the source tensor
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the batches dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: U8/S16/F16/F32.
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_c                             The size of the channels dimension of the destination tensor
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the batches dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] scale_x                           The scale value to apply on the source width
 * @param[in] scale_y                           The scale value to apply on the source height
 */
//! @endcond
__kernel void scale_nearest_neighbour_nhwc(
    TENSOR4D_T(src, SRC_TENSOR_TYPE),
    TENSOR4D_T(dst, DST_TENSOR_TYPE),
    const float scale_x,
    const float scale_y)
{
    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, 1, 0);           // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % dst_h; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / dst_h; // BATCH SIZE IDX
#else                                                  // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0;                        // BATCH SIZE IDX
#endif                                                 // defined(BATCHED_EXECUTION)

#ifdef SAMPLING_POLICY_TOP_LEFT
    float xi_f = (xo * scale_x);
    float yi_f = (yo * scale_y);
#elif SAMPLING_POLICY_CENTER
    float     xi_f = ((xo + 0.5f) * scale_x);
    float     yi_f = ((yo + 0.5f) * scale_y);
#else // SAMPLING_POLICY
#error("Unsupported sampling policy");
#endif // SAMPLING_POLICY

#ifdef ALIGN_CORNERS
    xi_f = round(xi_f);
    yi_f = round(yi_f);
#endif // ALIGN_CORNERS

    const int xi0 = clamp((int)xi_f, 0, (int)src_w - 1);
    const int yi0 = clamp((int)yi_f, 0, (int)src_h - 1);

    TILE(SRC_DATA_TYPE, 1, N0, in00);

    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi0, cout, src_w, src_h, 1, 1, false, in00);

    TILE(uint, 1, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    dst_indirect_y[0].v = xo + (yo * (int)(dst_w)) + bout * (int)(dst_w * dst_h);

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, 1, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, in00, dst_indirect_y);
}
#endif /* SCALE_NEAREST_NEIGHBOUR */

#if defined(SCALE_BILINEAR)
//! @cond Doxygen_Suppress
/** Performs scale on a tensor by interpolating with the BILINEAR method. (NHWC)
 *
 * @note If border mode replicate is used, is should be passed as -DBORDER_MODE_REPLICATE
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note The tensor type ("BUFFER" only is supported) of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" only is supported) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The border value value must be passed at compile time using -DCONSTANT_VALUE (e.g. -DCONSTANT_VALUE=0)
 * @note In case of F32/F16, -DIS_FLOATING_POINT must be passed at compile time
 * @note If the source tensor has more than 3 dimensions, -DBATCHED_EXECUTION must be passed at compile time
 *
 * @note In case of QASYMM8, the following extra information must be passed at compile time:
 * - The source offset e.g. -DOFFSET=4
 * - The source scale e.g. -DSCALE=4
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32.
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_c                             The size of the channels dimension of the source tensor
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the batches dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: U8/S16/F16/F32.
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_c                             The size of the channels dimension of the destination tensor
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the batches dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] scale_x                           The scale value to apply on the source width
 * @param[in] scale_y                           The scale value to apply on the source height
 */
//! @endcond
__kernel void scale_bilinear_nhwc(
    TENSOR4D_T(src, SRC_TENSOR_TYPE),
    TENSOR4D_T(dst, DST_TENSOR_TYPE),
    const float scale_x,
    const float scale_y)
{
    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, 1, 0);           // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % dst_h; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / dst_h; // BATCH SIZE IDX
#else                                                  // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0;                        // BATCH SIZE IDX
#endif                                                 // defined(BATCHED_EXECUTION)

#ifdef SAMPLING_POLICY_TOP_LEFT
    float xi_f = (xo * scale_x);
    float yi_f = (yo * scale_y);
#elif SAMPLING_POLICY_CENTER
    float     xi_f = ((xo + 0.5f) * scale_x - 0.5f);
    float     yi_f = ((yo + 0.5f) * scale_y - 0.5f);
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
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi, xi, cout, src_w, src_h, 1, 1, true, in00);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi, xi + 1, cout, src_w, src_h, 1, 1, true, in01);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi + 1, xi, cout, src_w, src_h, 1, 1, true, in10);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi + 1, xi + 1, cout, src_w, src_h, 1, 1, true, in11);
#else  // BORDER_MODE_REPLICATE
    const int xi0  = clamp(xi, 0, (int)src_w - 1);
    const int yi0  = clamp(yi, 0, (int)src_h - 1);
    const int xi1  = clamp(xi + 1, 0, (int)src_w - 1);
    const int yi1  = clamp(yi + 1, 0, (int)src_h - 1);

    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi0, cout, src_w, src_h, 1, 1, false, in00);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi0, xi1, cout, src_w, src_h, 1, 1, false, in01);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi1, xi0, cout, src_w, src_h, 1, 1, false, in10);
    T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, 1, N0, SRC_TENSOR_TYPE, src, bout, yi1, xi1, cout, src_w, src_h, 1, 1, false, in11);
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

    const float a  = (xi_f - (float)xi);
    const float b  = (1.f - a);
    const float a1 = (yi_f - (float)yi);
    const float b1 = (1.f - a1);

    out[0].v = CONVERT_SAT((CONVERT(in00[0].v, VEC_DATA_TYPE(float, N0)) * b * b1) + 
                   (CONVERT(in01[0].v, VEC_DATA_TYPE(float, N0)) * a * b1) + 
                   (CONVERT(in10[0].v, VEC_DATA_TYPE(float, N0)) * b * a1) + 
                   (CONVERT(in11[0].v, VEC_DATA_TYPE(float, N0)) * a * a1), 
                VEC_DATA_TYPE(DST_DATA_TYPE, N0));
#endif // defined(IS_FLOATING_POINT)

    TILE(uint, 1, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    dst_indirect_y[0].v = xo + (yo * (int)(dst_w)) + bout * (int)(dst_w * dst_h);

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, 1, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, out, dst_indirect_y);
}
#endif /* SCALE_BILINEAR */