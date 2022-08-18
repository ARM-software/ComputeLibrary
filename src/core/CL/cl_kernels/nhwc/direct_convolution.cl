/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "helpers_asymm.h"
#include "tile_helpers.h"

//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the direct convolution.
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16/QASYMM8/QASYMM8_SIGNED
 * @note The accumulation data type must be passed at compile time using -DACC_DATA_TYPE (e.g. -DDATA_TYPE_PROMOTED=half)
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The channels of the destination tensor must be passed at compile time using -DDST_CHANNELS (e.g. -DDDST_CHANNELS=64)
 * @note The tensor type ("BUFFER" or "IMAGE") of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the weights tensor must be passed at compile time using -DWEI_TENSOR_TYPE (e.g. -DWEI_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The number of K0 inner accumulations must be passed at compile time using -DK0 (e.g. -DK0=2)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note The zero value must be passed at compile time using -DZERO_VALUE (e.g. -DZERO_VALUE=0)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, .... n
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16 (only 4, 8 and 16 if WEI_TENSOR_TYPE=IMAGE)
 *
 *@note In case of QASYMM8/QASYMM8_SIGNED, the following extra information must be passed at compile time:
 * - -DIS_QUANTIZED
 * - The destination quantization multiplier e.g. -DDST_MULTIPLIER=1234
 * - The destination quantization shift e.g. -DDST_SHIFT=4
 * - The destination offset e.g. -DDST_OFFSET=4
 * - The source offset e.g. -DSRC_OFFSET=4
 * - The weights offset e.g. -DWEI_OFFSET=4
 * - The quantized zero value e.g. -DZERO_VALUE=4
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32/QASYMM8
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
 * @param[in]  wei_offset_first_element_in_bytes The offset of the first element in the bias matrix
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr (if F32/F16) or S32 (if QASYMM8/QASYMM8_SIGNED)
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 */
//! @endcond
__kernel void direct_convolution_nhwc(
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
#define _ISRC_WIDTH src_w
#define _ISRC_HEIGHT src_h
#define _ISRC_CHANNELS src_c
#define _IDST_WIDTH dst_w
#define _IDST_HEIGHT dst_h
#define _IDST_CHANNELS dst_c
#define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT)

    // If quantized, the output tile has to be quantized first before being stored to global memory
#if defined(IS_QUANTIZED)
#define _IOUTPUT_TILE cq
#else // defined(IS_QUANTIZED)
#define _IOUTPUT_TILE c
#endif // defined(IS_QUANTIZED)

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int mout = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH x HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0);           // BATCH SIZE IDX

    // .v    = access the whole vector (OpenCL vector)
    // .s[x] = access the vector element at position x (scalar access)
    TILE(int, M0, 1, xi);
    TILE(int, M0, 1, yi);

    // Convert the linear index to coordinate
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        xi[i].v = ((mout + i) % _IDST_WIDTH) * STRIDE_X;
        yi[i].v = ((mout + i) / _IDST_WIDTH) * STRIDE_Y;
        xi[i].v -= PAD_LEFT;
        yi[i].v -= PAD_TOP;
    })

    // Initialize the accumulators
    TILE(ACC_DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    for(int i = 0; i < (_IWEI_WIDTH * _IWEI_HEIGHT); ++i)
    {
        int ck = 0;
        int xk = i % _IWEI_WIDTH;
        int yk = i / _IWEI_WIDTH;

        int k = 0;
        for(; k <= (_ISRC_CHANNELS - K0); k += K0)
        {
            TILE(SRC_DATA_TYPE, M0, K0, a);
            TILE(WEI_DATA_TYPE, N0, K0, b);

            // Initialize tiles
            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = ZERO_VALUE;
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = ZERO_VALUE;
            })

            // Load tile from the src tensor
            T_LOAD_NHWC_INDIRECT(SRC_DATA_TYPE, M0, K0, SRC_TENSOR_TYPE, src, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, xi, yi, a);

            // Load tile from the weights tensor
            T_LOAD(WEI_DATA_TYPE, N0, K0, WEI_TENSOR_TYPE, wei, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, wei_stride_y, b);

            // Compute the matrix multiplication between two tiles
            T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, M0, N0, K0, NT, T, a, b, c);

            // Apply the offset correction (correction usually needed for asymmetric quantized computation)
            // The computation is not performed if both SRC_OFFSET and WEI_OFFSET are zero
            T_OFFSET_CORRECTION(ACC_DATA_TYPE, M0, N0, K0, SRC_OFFSET, WEI_OFFSET, a, b, c);

            ck += K0;
        }

        // We voluntarily use SRC_CHANNELS rather than _DSRC_CHANNELS
        // This #if directive should be removed in case of dynamic tensor support
#if defined(LEFTOVER_LOOP)
        // Left-over accumulations
        for(; k < _ISRC_CHANNELS; ++k)
        {
            TILE(SRC_DATA_TYPE, M0, 1, a);
            TILE(WEI_DATA_TYPE, N0, 1, b);

            // Initialize tiles
            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = ZERO_VALUE;
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = ZERO_VALUE;
            })

            // Load tile from the src tensor
            T_LOAD_NHWC_INDIRECT(SRC_DATA_TYPE, M0, 1, SRC_TENSOR_TYPE, src, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, xi, yi, a);

            // Load tile from the weights tensor
            // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
            T_LOAD(WEI_DATA_TYPE, N0, 1, BUFFER, wei, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, wei_stride_y, b);

            // Compute the matrix multiplication between two tiles
            T_MMUL(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, M0, N0, 1, NT, T, a, b, c);

            // Apply the offset correction (operation usually needed for asymmetric quantized computation)
            // The computation is not performed if both SRC_OFFSET and WEI_OFFSET are zero
            T_OFFSET_CORRECTION(ACC_DATA_TYPE, M0, N0, 1, SRC_OFFSET, WEI_OFFSET, a, b, c);

            ++ck;
        }
#endif // defined(LEFTOVER_LOOP)
    }

    // Offset correction required for the quantized asymmetric computation
    // The computation is not performed if both SRC_OFFSET and WEI_OFFSET are zero
    T_ADD_CONSTANT(ACC_DATA_TYPE, M0, N0, c, (_IWEI_WIDTH * _IWEI_HEIGHT * _ISRC_CHANNELS * SRC_OFFSET * WEI_OFFSET), c);

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

#if defined(IS_QUANTIZED)

    TILE(DST_DATA_TYPE, M0, N0, cq);

    // Quantize the tile
    T_QUANTIZE8_ASYMMETRIC(ACC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, c, cq);
#endif // defined(IS_QUANTIZED)

    // Apply activation
    T_ACTIVATION(DST_DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, _IOUTPUT_TILE, _IOUTPUT_TILE);

    // _IOUTPUT_TILE: c = fp32/fp16, cq=qasymm8
    // Store the tile in reverse order so the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, M0, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout, dst_stride_y, x_cond, _IOUTPUT_TILE, dst_indirect_y);

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