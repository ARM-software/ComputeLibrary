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
#include "tile_helpers.h"
// *INDENT-OFF*
// clang-format off
#if defined(WEI_WIDTH) && defined(WEI_HEIGHT) && defined(N0) && defined(M0) && defined(DILATION_X) && defined(DILATION_Y) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP)
//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the depthwise convolution for floating-point data types (F32/F16)
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The accumulation data type must be passed at compile time using -DACC_DATA_TYPE (e.g. -DDATA_TYPE_PROMOTED=half)
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2)
 * @note The convolution dilations must be passed at compile time using -DDILATION_X and -DDILATION_Y (e.g. -DDILATION_X=2, -DDILATION_Y=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The tensor type ("BUFFER" or "IMAGE") of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the weights tensor must be passed at compile time using -DWEI_TENSOR_TYPE (e.g. -DWEI_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=float)
 * @note The number of M0 rows (width) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The size of the partial store block in the first dimension must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note Only the following configurations of M0 and N0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, .... n (M0 != 1 with STRIDE_X == 1 && DILATION_X == 1 only)
 *  - N0 = 2, 3, 4, 8, 16 (only 4, 8 and 16 if WEI_TENSOR_TYPE=IMAGE)
 * @note The number of rows to read from the src tensor must be passed at compile time using -DM0_A (e.g., -DM0_A=3). M0_A must be equal to WEI_WIDTH + (M0 - 1)
 * @note The number of columns to read from the src tensor must be passed at compile time using -DN0_A. It can either be 1 (for DEPTH_MULTIPLIER > 1) or N0 (for DEPTH_MULTIPLIER == 1)
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_c                             The size of the channels dimension of the source tensor
 * @param[in]  src_w                             The size of the width dimension of the source tensor
 * @param[in]  src_h                             The size of the height dimension of the source tensor
 * @param[in]  src_n                             The size of the batches dimension of the source tensor
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_c                             The size of the channels dimension of the destination tensor
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the batches dimension of the destination tensor
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
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 */
//! @endcond
__kernel void dwc_native_fp_nhwc(
    TENSOR4D_T(src, SRC_TENSOR_TYPE),
    TENSOR4D_T(dst, DST_TENSOR_TYPE),
    TENSOR4D(wei, WEI_TENSOR_TYPE)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bia)
#endif // defined(HAS_BIAS)
)
{
    // Only the weight tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _IWEI_WIDTH WEI_WIDTH
#define _IWEI_HEIGHT WEI_HEIGHT
#define _IM0_A M0_A        // _IWEI_WIDTH + (M0 - 1) Rows tile A (If M0 != 1, the tiles overlap of 1 element on the X dimension)
#define _IN0_A N0_A        // Cols tile A. It can be either 1 (for DEPTH_MULTIPLIER > 1) or N0 (for DEPTH_MULTIPLIER == 1)
#define _IM0_B _IWEI_WIDTH // Rows tile B
#define _IN0_B N0          // Cols tile B
#define _IBOUNDARY_CHECK (!((WEI_WIDTH == 1 && WEI_HEIGHT == 1 && PAD_LEFT == 0 && PAD_TOP == 0 && M0 == 1)))

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % dst_h; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / dst_h; // BATCH SIZE IDX
#else                                                  // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0;                        // BATCH SIZE IDX
#endif                                                 // defined(BATCHED_EXECUTION)

    int xi = xo * STRIDE_X;
    int yi = yo * STRIDE_Y;
    xi -= PAD_LEFT;
    yi -= PAD_TOP;

    TILE(ACC_DATA_TYPE, M0, N0, c);

    // Reset accumulators
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

#if _IWEI_HEIGHT < 5
    LOOP_UNROLLING(int, yk, 0, 1, _IWEI_HEIGHT,
#else  // _IWEI_HEIGHT <= 5
    for(int yk = 0; yk < _IWEI_HEIGHT; ++yk)
#endif // _IWEI_HEIGHT <= 5
    {
        TILE(SRC_DATA_TYPE, _IM0_A, _IN0_A, a);

        LOOP_UNROLLING(int, i, 0, 1, _IM0_A,
        {
            a[i].v = 0;
        })

        // Load tile from the src tensor (TILE A)
        T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, _IM0_A, _IN0_A, SRC_TENSOR_TYPE, src, bout, yi + yk * DILATION_Y, xi, (cout / DEPTH_MULTIPLIER), SRC_WIDTH, SRC_HEIGHT, DILATION_X, 1, _IBOUNDARY_CHECK, a);

        TILE(WEI_DATA_TYPE, _IM0_B, _IN0_B, b);

        // Load tile from the weights tensor (TILE B)
        T_LOAD(WEI_DATA_TYPE, _IM0_B, _IN0_B, WEI_TENSOR_TYPE, wei, cout, yk * _IM0_B, 1, wei_stride_y, b);

        // Optimized path for STRIDE_X == 1
        // If M0 != 1, we can skip the common loads between the two applied kernels on the X (WIDTH) dimension
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, xk, 0, 1, _IWEI_WIDTH,
            {
                c[m0].v = fma(a[xk + m0].v, b[xk].v, c[m0].v);
            })
        })
    }
#if _IWEI_HEIGHT < 5
                      )
#endif // _IWEI_HEIGHT <= 5

#if defined(HAS_BIAS)
    TILE(BIA_DATA_TYPE, 1, N0, bias0);

    T_LOAD(BIA_DATA_TYPE, 1, N0, BUFFER, bia, cout, 0, 0, 0, bias0);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(ACC_DATA_TYPE, M0, N0, c, bias0, c);
#endif // HAS_BIAS

    T_ACTIVATION(ACC_DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, c, c);

    TILE(uint, M0, 1, dst_indirect_y);

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    if(x_cond)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            int xi_out = min(xo + M0 - 1 - m0, (int)(DST_WIDTH) - 1);
            VSTORE_PARTIAL(N0, PARTIAL_N0)
            (c[M0 - 1 - m0].v, 0, (__global DST_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + cout * sizeof(DST_DATA_TYPE) + (uint)xi_out * dst_stride_y + (uint)yo * dst_stride_z + (uint)bout * dst_stride_w));
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            int xi_out = min(xo + M0 - 1 - m0, (int)(DST_WIDTH) - 1);
            VSTORE(N0)
            (c[M0 - 1 - m0].v, 0, (__global DST_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + cout * sizeof(DST_DATA_TYPE) + (uint)xi_out * dst_stride_y + (uint)yo * dst_stride_z + (uint)bout * dst_stride_w));
        })
    }
}
#endif // defined(WEI_WIDTH) && defined(WEI_HEIGHT) && defined(N0) && defined(M0) && defined(DILATION_X) && defined(DILATION_Y) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP)
// *INDENT-ON*
// clang-format on