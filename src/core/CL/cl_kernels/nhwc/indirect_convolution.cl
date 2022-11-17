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
/** OpenCL kernel to compute the indirect convolution 2d indirect buffer.
 *
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2)
 * @note The kernel width must be passed at compile time using -DWEI_CONV_WIDTH (e.g. -DWEI_CONV_WIDTH=9)
 * @note The spatial dimensions of the source tensor used by conv2d must be passed at compile time using -DSRC_CONV_WIDTH and -DSRC_CONV_HEIGHT (e.g. -DSRC_CONV_WIDTH=96, -DSRC_CONV_HEIGHT=64)
 * @note The width dimension of the destination tensor produced by conv2d must be passed at compile time using -DDST_CONV_WIDTH (e.g. -DDST_CONV_WIDTH=96)
 * @note The tensor type ("BUFFER" only) of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, and 8
 *
 * @param[out] dst_img                           CLImage object to the destination tensor (DST_TENSOR_TYPE=IMAGE only)
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: INT32
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_c                             The size of the channels dimension of the destination tensor
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the batches dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void indirect_convolution_address_precalculation(
    TENSOR4D_T(dst, DST_TENSOR_TYPE))
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Note: WIDTH = M0 x KernelWidth x KernelHeight

    // m index
    const int mi = x % M0;
    // Kernel index
    const int ki = x / M0;
    // Kernel width coordinate
    const int xk = ki % WEI_CONV_WIDTH;
    // kernel height coordinate
    const int yk = ki / WEI_CONV_WIDTH;

    TILE(DST_DATA_TYPE, 1, 1, xi);
    TILE(DST_DATA_TYPE, 1, 1, yi);
    TILE(DST_DATA_TYPE, 1, 1, my);

    const int mout = y * M0;

    xi[0].s[0] = ((mout + mi) % DST_CONV_WIDTH) * STRIDE_X;
    yi[0].s[0] = ((mout + mi) / DST_CONV_WIDTH) * STRIDE_Y;
    xi[0].s[0] -= PAD_LEFT;
    yi[0].s[0] -= PAD_TOP;

    const int x_s = xi[0].s[0] + xk;
    const int y_s = yi[0].s[0] + yk;
    my[0].s[0]    = x_s + y_s * SRC_CONV_WIDTH;
    my[0].s[0]    = my[0].s[0] + z * (int)(SRC_CONV_WIDTH * SRC_CONV_HEIGHT);
    my[0].s[0]    = select(-1, my[0].s[0], x_s >= 0);
    my[0].s[0]    = select(-1, my[0].s[0], x_s < SRC_CONV_WIDTH);
    my[0].s[0]    = select(-1, my[0].s[0], y_s >= 0);
    my[0].s[0]    = select(-1, my[0].s[0], y_s < SRC_CONV_HEIGHT);

    VSTORE(1)
    (my[0].s[0], 0, (__global DST_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + x * sizeof(DST_DATA_TYPE) + y * dst_stride_y + z * dst_stride_z));
}