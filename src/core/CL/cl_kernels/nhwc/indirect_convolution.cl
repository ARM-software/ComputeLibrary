/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#if defined(INDIRECT_CONVOLUTION_ADDRESS_PRECALCULATION)
//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the indirect convolution 2d indirect buffer.
 *
 * @note This kernel only works for unit batch_size
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
 * @param[out] dst_img                           (Not supported) Write only cl_image object for the destination tensor. Included when DST_TENSOR_TYPE=IMAGE
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
    TENSOR4D_WO_T(dst, DST_TENSOR_TYPE))
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
#endif // defined(INDIRECT_CONVOLUTION_ADDRESS_PRECALCULATION)

#if defined(INDIRECT_CONVOLUTION_NHWC)
//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the indirect convolution.
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The tensor type ("BUFFER" or "IMAGE") of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the weights tensor must be passed at compile time using -DWEI_TENSOR_TYPE (e.g. -DWEI_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The number of K0 inner accumulations must be passed at compile time using -DK0 (e.g. -DK0=2)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note The vector length used for loading the values from the indirect buffer should be passed at compile time using -DIND_BUFF_VEC_SIZE (e.g. -DIND_BUFF_VEC_SIZE=4)
 * @note The activation function to fuse and corresponding A and B values should be passed at compile time using -DACTIVATION_TYPE, -DA_VAL, and -DB_VAL
 *        (e.g. -DFUNCTION_TYPE=lu_brelu_op, -DA_VAL=3.0, and -DB_VAL=1.0)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, and 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16 (only 4, 8 and 16 if WEI_TENSOR_TYPE=IMAGE)
 *
 * @param[in]  src_img                           (Not supported) Read only cl_image object for the source tensor. Included when SRC_TENSOR_TYPE=IMAGE
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_c                             The size of the channels dimension of the source tensor
 * @param[in]  src_w                             The size of the width dimension of the source tensor
 * @param[in]  src_h                             The size of the height dimension of the source tensor
 * @param[in]  src_n                             The size of the batches dimension of the source tensor
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  off_img                           (Not supported) Read only cl_image object for the indirect buffer tensor. Included when OFF_TENSOR_TYPE=IMAGE
 * @param[in]  off_ptr                           Pointer to the indirect buffer tensor. Supported data type: INT32
 * @param[in]  off_stride_y                      Stride of the indirect buffer tensor in Y dimension (in bytes)
 * @param[in]  off_stride_z                      Stride of the indirect buffer tensor in Z dimension (in bytes)
 * @param[in]  off_stride_w                      Stride of the indirect buffer tensor in W dimension (in bytes)
 * @param[in]  off_c                             The size of the channels dimension of the indirect buffer tensor
 * @param[in]  off_w                             The size of the width dimension of the indirect buffer tensor
 * @param[in]  off_h                             The size of the height dimension of the indirect buffer tensor
 * @param[in]  off_n                             The size of the batches dimension of the indirect buffer tensor
 * @param[in]  off_offset_first_element_in_bytes The offset of the first element in the indirect buffer tensor
 * @param[out] dst_img                           (Not supported) Write only cl_image object for the destination tensor. Included when DST_TENSOR_TYPE=IMAGE
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_c                             The size of the channels dimension of the destination tensor
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the batches dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] wei_img                           (Optional) Read only cl_image object for the weights tensor. Included when WEI_TENSOR_TYPE=IMAGE
 * @param[out] wei_ptr                           Pointer to the weights tensor. Supported data type: same as @p src_ptr
 * @param[in]  wei_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  wei_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  wei_stride_w                      Stride of the weights tensor in W dimension (in bytes)
 * @param[in]  wei_c                             The size of the channels dimension of the weights tensor
 * @param[in]  wei_w                             The size of the width dimension of the weights tensor
 * @param[in]  wei_h                             The size of the height dimension of the weights tensor
 * @param[in]  wei_n                             The size of the batches dimension of the weights tensor
 * @param[in]  wei_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 */
//! @endcond
__kernel void indirect_convolution_nhwc(
    TENSOR4D_RO_T(src, SRC_TENSOR_TYPE),
    TENSOR4D_RO_T(off, OFF_TENSOR_TYPE),
    TENSOR4D_WO_T(dst, DST_TENSOR_TYPE),
    TENSOR4D_RO_T(wei, WEI_TENSOR_TYPE)
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
#define _ISRC_CHANNELS SRC_CHANNELS
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT)

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int mout = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH x HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0);           // BATCH SIZE IDX

    off_offset_first_element_in_bytes += get_global_id(1) * off_stride_y;
    off_offset_first_element_in_bytes += bout * off_stride_z;

    // Initialize the accumulators
    TILE(DST_DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    for(int i = 0; i < (_IWEI_WIDTH * _IWEI_HEIGHT); ++i)
    {
        TILE(int, 1, IND_BUFF_VEC_SIZE, my);
        T_LOAD(int, 1, IND_BUFF_VEC_SIZE, OFF_TENSOR_TYPE, off, i * M0, 0, 1, 0, my);

        int ck = 0;
        for(; ck <= (_ISRC_CHANNELS - K0); ck += K0)
        {
            TILE(SRC_DATA_TYPE, M0, K0, a);
            TILE(WEI_DATA_TYPE, N0, K0, b);

            // Initialize tiles
            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = 0.0;
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = 0.0;
            })

            // Load tile from the src tensor
            T_LOAD2D_INDIRECT(SRC_DATA_TYPE, M0, K0, SRC_TENSOR_TYPE, src, ck, src_stride_y, my, a);

            // Load tile from the weights tensor
            T_LOAD(WEI_DATA_TYPE, N0, K0, WEI_TENSOR_TYPE, wei, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, wei_stride_y, b);

            // Compute the matrix multiplication between two tiles
            T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, NT, T, a, b, c);
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
                a[i].v = 0.0;
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = 0.0;
            })

            // Load tile from the src tensor
            T_LOAD2D_INDIRECT(SRC_DATA_TYPE, M0, 1, SRC_TENSOR_TYPE, src, ck, src_stride_y, my, a);

            // Load tile from the weights tensor
            // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
            T_LOAD(WEI_DATA_TYPE, N0, 1, BUFFER, wei, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, wei_stride_y, b);

            // Compute the matrix multiplication between two tiles
            T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, DST_DATA_TYPE, M0, N0, 1, NT, T, a, b, c);
        }
#endif // defined(LEFTOVER_LOOP)
    }

#if defined(HAS_BIAS)
    TILE(BIA_DATA_TYPE, 1, N0, bias0);

    T_LOAD(BIA_DATA_TYPE, 1, N0, BUFFER, bia, cout, 0, 1, 0, bias0);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(DST_DATA_TYPE, M0, N0, c, bias0, c);

#endif // HAS_BIAS

    // Apply activation
    T_ACTIVATION(DST_DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, c, c);

    TILE(uint, M0, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        dst_indirect_y[i].v = (uint)min(mout + i, (int)(_IDST_WIDTH * _IDST_HEIGHT) - 1);
        dst_indirect_y[i].v += bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);
    })

    const bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, M0, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, c, dst_indirect_y);

#undef _IWEI_WIDTH
#undef _IWEI_HEIGHT
#undef _ISRC_CHANNELS
#undef _IDST_WIDTH
#undef _IDST_HEIGHT
#undef _IY_MULTIPLIER
}
#endif // defined(INDIRECT_CONVOLUTION_NHWC)
