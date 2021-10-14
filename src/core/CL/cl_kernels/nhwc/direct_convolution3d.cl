/*
 * Copyright (c) 2021 Arm Limited.
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
/** OpenCL kernel to compute the direct convolution 3d.
 *
 * @note Data layout supported: NDHWC
 * @note Data type supported: F32/F16
 * @note The accumulation data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE_PROMOTED=half)
 * @note The convolution padding (left, top and front) must be passed at compile time using -DPAD_LEFT, -DPAD_TOP and -DPAD_FRONT (e.g. -DPAD_LEFT=2, -DPAD_TOP=2, -DPAD_FRONT=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE_X, -DSTRIDE_Y and -DSTRIDE_Z (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2, -DSTRIDE_Z=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH, -DWEI_HEIGHT and -DWEI_DEPTH (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9, -DWEI_DEPTH=9)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH, -DSRC_HEIGHT and -DSRC_DEPTH (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64, -DSRC_DEPTH=32)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH, -DDST_HEIGHT and -DDST_DEPTH (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64, -DDST_DEPTH=32)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The channels of the destination tensor must be passed at compile time using -DDST_CHANNELS (e.g. -DDST_CHANNELS=64)
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The number of K0 inner accumulations must be passed at compile time using -DK0 (e.g. -DK0=2)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, .... n
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  wei_ptr                           Pointer to the weights tensor. Supported data type: same as @p src_ptr
 * @param[in]  wei_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  wei_step_x                        wei_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  wei_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  wei_step_y                        wei_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  wei_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  wei_step_z                        wei_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  wei_stride_w                      Stride of the weights tensor in W dimension (in bytes)
 * @param[in]  wei_step_w                        wei_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  wei_offset_first_element_in_bytes The offset of the first element in the weights matrix
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 */
//! @endcond
__kernel void direct_convolution3d_ndhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER),
    TENSOR4D(wei, BUFFER)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bia)
#endif // defined(HAS_BIAS)
)
{
#define _IWEI_WIDTH WEI_WIDTH
#define _IWEI_HEIGHT WEI_HEIGHT
#define _IWEI_DEPTH WEI_DEPTH
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _ISRC_DEPTH SRC_DEPTH
#define _ISRC_CHANNELS SRC_CHANNELS
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _IDST_DEPTH DST_DEPTH
#define _IDST_CHANNELS DST_CHANNELS
#define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT * _IWEI_DEPTH)

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int mout = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH x HEIGHT x DEPTH
    const int bout = GET_SPATIAL_IDX(2, 1, 0);           // BATCH SIZE IDX

    TILE(int, M0, 1, xi);
    TILE(int, M0, 1, yi);
    TILE(int, M0, 1, zi);

    // Convert the linear index to coordinate
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        xi[i].v = ((mout + i) % _IDST_WIDTH) * STRIDE_X;
        yi[i].v = (((mout + i) / _IDST_WIDTH) % _IDST_HEIGHT) * STRIDE_Y;
        zi[i].v = (((mout + i) / (_IDST_WIDTH * _IDST_HEIGHT)) % _IDST_DEPTH) * STRIDE_Z;

        xi[i].v -= PAD_LEFT;
        yi[i].v -= PAD_TOP;
        zi[i].v -= PAD_FRONT;
    })

    // Initialize the accumulators
    TILE(ACC_DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = (ACC_DATA_TYPE)0;
    })

    for(int i = 0; i < _IY_MULTIPLIER; ++i)
    {
        int ck = 0;
        int xk = i % _IWEI_WIDTH;
        int yk = (i / _IWEI_WIDTH) % _IWEI_HEIGHT;
        int zk = i / (_IWEI_WIDTH * _IWEI_HEIGHT);

        int k = 0;
        for(; k <= (_ISRC_CHANNELS - K0); k += K0)
        {
            TILE(DATA_TYPE, M0, K0, a);
            TILE(DATA_TYPE, N0, K0, b);

            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = (DATA_TYPE)0;
            })

            // Load tile from the src tensor
            T_LOAD_NDHWC_INDIRECT(DATA_TYPE, M0, K0, BUFFER, src, bout, zk, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, _ISRC_DEPTH, src_stride_y, xi, yi, zi, a);

            // Load tile from the weights tensor
            const int b_offs = k + (xk * _ISRC_CHANNELS) + (yk * _ISRC_CHANNELS * _IWEI_WIDTH) + (zk * _ISRC_CHANNELS * _IWEI_WIDTH * _IWEI_HEIGHT);
            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                if((cout + i) < _IDST_CHANNELS)
                {
                    LOOP_UNROLLING(int, j, 0, 1, K0,
                    {
                        b[i].s[j] = *(__global DATA_TYPE *)(wei_ptr + wei_offset_first_element_in_bytes + (cout + i) * sizeof(DATA_TYPE) + j * wei_stride_y + b_offs * wei_stride_y);
                    })
                }
            })

            // Compute the matrix multiplication between two tiles
            T_MMUL(DATA_TYPE, DATA_TYPE, ACC_DATA_TYPE, M0, N0, K0, NT, T, a, b, c);

            ck += K0;
        }

#if((_ISRC_CHANNELS % K0) != 0)
        // Left-over accumulations
        for(; k < _ISRC_CHANNELS; ++k)
        {
            TILE(DATA_TYPE, M0, 1, a);
            TILE(DATA_TYPE, N0, 1, b);

            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = (DATA_TYPE)0;
            })

            // Load tile from the src tensor
            T_LOAD_NDHWC_INDIRECT(DATA_TYPE, M0, 1, BUFFER, src, bout, zk, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, _ISRC_DEPTH, src_stride_y, xi, yi, zi, a);

            // Load tile from the weights tensor
            const int b_offs = k + (xk * _ISRC_CHANNELS) + (yk * _ISRC_CHANNELS * _IWEI_WIDTH) + (zk * _ISRC_CHANNELS * _IWEI_WIDTH * _IWEI_HEIGHT);
            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                if((cout + i) < _IDST_CHANNELS)
                {
                    b[i].v = *(__global DATA_TYPE *)(wei_ptr + wei_offset_first_element_in_bytes + (cout + i) * sizeof(DATA_TYPE) + b_offs * wei_stride_y);
                }
            })

            // // Compute the matrix multiplication between two tiles
            T_MMUL(DATA_TYPE, DATA_TYPE, ACC_DATA_TYPE, M0, N0, 1, NT, T, a, b, c);

            ++ck;
        }
#endif // ((_ISRC_CHANNELS % K0) != 0)
    }

#if defined(HAS_BIAS)
    TILE(DATA_TYPE, 1, N0, bias0);

    if((cout + N0) <= _IDST_CHANNELS)
    {
        bias0[0].v = VLOAD(N0)(0, (__global DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + cout * sizeof(DATA_TYPE)));
    }
    else
    {
        VLOAD_PARTIAL(N0, PARTIAL_N0)
        (bias0[0].v, 0, (__global DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + cout * sizeof(DATA_TYPE)));
    }

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(ACC_DATA_TYPE, M0, N0, c, bias0, c);

#endif // HAS_BIAS

    TILE(uint, M0, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        dst_indirect_y[i].v = (uint)min(mout + i, (int)(_IDST_WIDTH *_IDST_HEIGHT * _IDST_DEPTH) - 1);
        dst_indirect_y[i].v += bout * (int)(_IDST_WIDTH *_IDST_HEIGHT * _IDST_DEPTH);
    })

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_N0, BUFFER, dst, cout, dst_stride_y, x_cond, c, dst_indirect_y);
}