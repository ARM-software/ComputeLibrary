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
#include "tile_helpers.h"

//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the transposed convolution.
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The transposed convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The transposed convolution strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The channels of the destination tensor must be passed at compile time using -DDST_CHANNELS (e.g. -DDST_CHANNELS=64)
 * @note The tensor type (currently only "BUFFER" is supported) of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type (currently only "BUFFER" is supported) of the weights tensor must be passed at compile time using -DWEI_TENSOR_TYPE (e.g. -DWEI_TENSOR_TYPE=BUFFER)
 * @note The tensor type (currently only "BUFFER" is supported) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The number of K0 inner accumulations must be passed at compile time using -DK0 (e.g. -DK0=2)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1
 *  - N0 = 1
 *  - K0 = 2, 3, 4, 8
 *
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_c                             The size of the channels (IFM) dimension of the source tensor
 * @param[in]  src_w                             The size of the width dimension of the source tensor
 * @param[in]  src_h                             The size of the height dimension of the source tensor
 * @param[in]  src_n                             The size of the batches dimension of the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_c                             The size of the channels (OFM) dimension of the destination tensor
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the batches dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  wei_ptr                           Pointer to the weights tensor. Supported data type: same as @p src_ptr
 * @param[in]  wei_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  wei_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  wei_stride_w                      Stride of the weights tensor in W dimension (in bytes)
 * @param[in]  wei_c                             The size of the channels (IFM) dimension of the weights tensor
 * @param[in]  wei_w                             The size of the width dimension of the weights tensor
 * @param[in]  wei_h                             The size of the height dimension of the weights tensor
 * @param[in]  wei_n                             The size of the batches (OFM) dimension of the weights tensor
 * @param[in]  wei_offset_first_element_in_bytes The offset of the first element in the bias matrix
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr (if F32/F16)
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 */
//! @endcond
__kernel void transposed_convolution_nhwc(
    TENSOR4D_T(src, SRC_TENSOR_TYPE),
    TENSOR4D_T(dst, DST_TENSOR_TYPE),
    TENSOR4D_T(wei, WEI_TENSOR_TYPE)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bia)
#endif // defined(HAS_BIAS)
)
{
    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _IWEI_WIDTH WEI_WIDTH
#define _IWEI_HEIGHT WEI_HEIGHT
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _ISRC_CHANNELS SRC_CHANNELS
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _IDST_CHANNELS DST_CHANNELS
#define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT)

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int mout = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH x HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0);           // BATCH SIZE IDX

    // .v    = access the whole vector (OpenCL vector)
    // .s[x] = access the vector element at position x (scalar access)
    TILE(int, 1, M0, xi);
    TILE(int, 1, M0, yi);
    TILE(int, 1, M0, xu);
    TILE(int, 1, M0, yu);

    // Convert the linear index to coordinate
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        xu[0].s[i] = ((mout + i) % _IDST_WIDTH) - PAD_LEFT;
        yu[0].s[i] = ((mout + i) / _IDST_WIDTH) - PAD_TOP;
        xi[0].s[i] = ceil(xu[0].s[i] / (float)STRIDE_X);
        yi[0].s[i] = ceil(yu[0].s[i] / (float)STRIDE_Y);
    })

    // Initialize the accumulators
    TILE(ACC_DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    // Flipped indices
    const int x_start = _IWEI_WIDTH - (xi[0].s[0] * STRIDE_X - xu[0].s[0]) - 1;
    const int y_start = _IWEI_HEIGHT - (yi[0].s[0] * STRIDE_Y - yu[0].s[0]) - 1;

    for(int yk = y_start, yi_step = 0; yk >= 0; yk -= STRIDE_Y, ++yi_step)
    {
        for(int xk = x_start, xi_step = 0; xk >= 0; xk -= STRIDE_X, ++xi_step)
        {
            int weights_y = cout * _IY_MULTIPLIER + yk * _IWEI_WIDTH + xk;

            TILE(int, 1, M0, my);

            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                int x_s    = xi[0].s[i] + xi_step;
                int y_s    = yi[0].s[i] + yi_step;
                my[0].s[i] = x_s + y_s *_ISRC_WIDTH;
                my[0].s[i] = my[0].s[i] + bout * (int)(_ISRC_WIDTH * _ISRC_HEIGHT);
                my[0].s[i] = select(-1, my[0].s[i], x_s >= 0);
                my[0].s[i] = select(-1, my[0].s[i], x_s < _ISRC_WIDTH);
                my[0].s[i] = select(-1, my[0].s[i], y_s >= 0);
                my[0].s[i] = select(-1, my[0].s[i], y_s < _ISRC_HEIGHT);
            })

            int ck = 0;
            for(; ck <= (_ISRC_CHANNELS - K0); ck += K0)
            {
                TILE(SRC_DATA_TYPE, M0, K0, a);
                TILE(WEI_DATA_TYPE, N0, K0, b);

                // Initialize tiles
                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = 0.f;
                })

                LOOP_UNROLLING(int, i, 0, 1, N0,
                {
                    b[i].v = 0.f;
                })

                // Load tile from the src tensor
                T_LOAD2D_INDIRECT(SRC_DATA_TYPE, M0, K0, SRC_TENSOR_TYPE, src, ck, src_stride_y, my, a);

                // Load tile from the weights tensor
                T_LOAD(WEI_DATA_TYPE, N0, K0, WEI_TENSOR_TYPE, wei, ck, weights_y, _IY_MULTIPLIER, wei_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, M0, N0, K0, NT, T, a, b, c);
            }

            // This #if directive should be removed in case of dynamic tensor support
#if defined(LEFTOVER_LOOP)
            // Left-over accumulations
            for(; ck < _ISRC_CHANNELS; ++ck)
            {
                TILE(SRC_DATA_TYPE, M0, 1, a);
                TILE(WEI_DATA_TYPE, N0, 1, b);

                // Initialize tiles
                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = 0.f;
                })

                // Load tile from the src tensor
                // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
                T_LOAD2D_INDIRECT(SRC_DATA_TYPE, M0, 1, BUFFER, src, ck, src_stride_y, my, a);

                // Load tile from the weights tensor
                // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
                T_LOAD(WEI_DATA_TYPE, N0, 1, BUFFER, wei, ck, weights_y, _IY_MULTIPLIER, wei_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, M0, N0, 1, NT, T, a, b, c);
            }
#endif // defined(LEFTOVER_LOOP)
        }
    }

#if defined(HAS_BIAS)
    TILE(BIA_DATA_TYPE, 1, N0, bias0);

    T_LOAD(BIA_DATA_TYPE, 1, N0, BUFFER, bia, cout, 0, 1, 0, bias0);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(ACC_DATA_TYPE, M0, N0, c, bias0, c);

#endif // HAS_BIAS

    TILE(uint, M0, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        dst_indirect_y[i].v = (uint)min(mout + i, (int)(_IDST_WIDTH * _IDST_HEIGHT) - 1);
        dst_indirect_y[i].v += bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);
    })

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, M0, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, c, dst_indirect_y);

#undef _IWEI_WIDTH
#undef _IWEI_HEIGHT
#undef _ISRC_WIDTH
#undef _ISRC_HEIGHT
#undef _ISRC_CHANNELS
#undef _IDST_WIDTH
#undef _IDST_HEIGHT
#undef _IDST_CHANNELS
#undef _IY_MULTIPLIER
}