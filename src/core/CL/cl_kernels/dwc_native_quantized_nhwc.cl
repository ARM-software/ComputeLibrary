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
#include "helpers_asymm.h"
#include "tile_helpers.h"

#if defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_WIDTH) && defined(DST_HEIGHT) && defined(WEI_WIDTH) && defined(WEI_HEIGHT) && defined(N0) && defined(M0) && defined(DILATION_X) && defined(DILATION_Y) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP)
//! @cond Doxygen_Suppress
/** OpenCL kernel to compute the depthwise convolution for quantized data types
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: QSYMM8/QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE_X and -DSTRIDE_Y (e.g. -DSTRIDE_X=2, -DSTRIDE_Y=2)
 * @note The convolution dilations must be passed at compile time using -DDILATION_X and -DDILATION_Y (e.g. -DDILATION_X=2, -DDILATION_Y=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The channels of the destination tensor must be passed at compile time using -DDST_CHANNELS (e.g. -DDDST_CHANNELS=64)
 * @note The tensor type ("BUFFER" or "IMAGE") of the source tensor must be passed at compile time using -DSRC_TENSOR_TYPE (e.g. -DSRC_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the weights tensor must be passed at compile time using -DWEI_TENSOR_TYPE (e.g. -DWEI_TENSOR_TYPE=BUFFER)
 * @note The tensor type ("BUFFER" or "IMAGE") of the destination tensor must be passed at compile time using -DDST_TENSOR_TYPE (e.g. -DDST_TENSOR_TYPE=BUFFER)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=int8)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=int8)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=int8)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=int)
 * @note The number of M0 rows (width) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The size of the partial store block in the first dimension must be passed at compile time using -DPARTIAL_N0 (e.g. -DPARTIAL_N0=1)
 * @note The activation type must be passed at compile using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note The A and B variables required by some activation functions must be passed at compile time using -DA_VAL= and -DB_VAL= respectively
 * @note The quantization offset used for both the per-tensor and per-channel quantization must be passed at compile using -DDST_OFFSET (e.g., -DDST_OFFSET=3)
 * @note The quantization shift for the per-tensor quantization must be passed at compile time using -DDST_SHIFT (e.g., -DDST_SHIFT=1)
 * @note The quantization multiplier for the per-tensor quantization must be passed at compile using -DDST_MULTIPLIER (e.g., -DDST_MULTIPLER=121432)
 * @note Only the following configurations of M0 and N0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, .... n (M0 != 1 with STRIDE_X == 1 && DILATION_X == 1 only)
 *  - N0 = 2, 3, 4, 8, 16
 * @note The number of rows to read from the src tensor must be passed at compile time using -DM0_A (e.g., -DM0_A=3). M0_A must be equal to WEI_WIDTH + (M0 - 1)
 *
 * @param[in]  src_ptr                                       Pointer to the source tensor. Supported data type: QSYMM8/QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL
 * @param[in]  src_stride_x                                  Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                                    src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                                  Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                                    src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                                  Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                                    src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                                  Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                                    src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes             The offset of the first element in the source tensor
 * @param[out] dst_ptr                                       Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_x                                  Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                                    dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                                  Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                                    dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                                  Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                                    dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                                  Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  dst_step_w                                    dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes             The offset of the first element in the destination tensor
 * @param[in]  wei_ptr                                       Pointer to the weights tensor. Supported data type: same as @p src_ptr
 * @param[in]  wei_stride_x                                  Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  wei_step_x                                    wei_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  wei_stride_y                                  Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  wei_step_y                                    wei_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  wei_stride_z                                  Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  wei_step_z                                    wei_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  wei_stride_w                                  Stride of the weights tensor in W dimension (in bytes)
 * @param[in]  wei_step_w                                    wei_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  wei_offset_first_element_in_bytes             The offset of the first element in the weights tensor
 * @param[in]  dst_multipliers_ptr                           Pointer to the destination multipliers tensor for the per-channel quantization. Supported data type: S32
 * @param[in]  dst_multipliers_stride_x                      Stride of the destination multipliers tensor in X dimension (in bytes)
 * @param[in]  dst_multipliers_step_x                        dst_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_multipliers_offset_first_element_in_bytes The offset of the first element in the destination multipliers tensor
 * @param[in]  dst_shifts_ptr                                Pointer to the destination shifts tensor for the per-channel quantization. Supported data type: S32
 * @param[in]  dst_shifts_stride_x                           Stride of the destination shifts tensor in X dimension (in bytes)
 * @param[in]  dst_shifts_step_x                             dst_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_shifts_offset_first_element_in_bytes      The offset of the first element in the destination shifts tensor
 * @param[in]  bia_ptr                                       (Optional) Pointer to the bias tensor Supported data type: S32
 * @param[in]  bia_stride_x                                  (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                                    (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes             (Optional) The offset of the first element in the bias tensor
 */
//! @endcond
__kernel void dwc_native_quantized_nhwc(
    TENSOR4D(src, SRC_TENSOR_TYPE),
    TENSOR4D(dst, DST_TENSOR_TYPE),
    TENSOR4D(wei, WEI_TENSOR_TYPE),
    VECTOR_DECLARATION(dst_multipliers),
    VECTOR_DECLARATION(dst_shifts)
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
#define _IDST_WIDTH DST_WIDTH
#define _IDST_HEIGHT DST_HEIGHT
#define _IDST_CHANNELS DST_CHANNELS
#define _IM0_A M0_A        // _IWEI_WIDTH + (M0 - 1) Rows tile A (If M0 != 1, the tiles overlap of 1 element on the X dimension)
#define _IN0_A N0          // Cols tile A
#define _IM0_B _IWEI_WIDTH // Rows tile B
#define _IN0_B N0          // Cols tile B
#define _IBOUNDARY_CHECK (!((WEI_WIDTH == 1 && WEI_HEIGHT == 1 && PAD_LEFT == 0 && PAD_TOP == 0 && M0 == 1)))

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int xo   = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH
#if defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0) % _IDST_HEIGHT; // HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0) / _IDST_HEIGHT; // BATCH SIZE IDX
#else                                                         // defined(BATCHED_EXECUTION)
    const int yo   = GET_SPATIAL_IDX(2, 1, 0); // HEIGHT
    const int bout = 0;                        // BATCH SIZE IDX
#endif                                                        // defined(BATCHED_EXECUTION)

    int xi = xo * STRIDE_X;
    int yi = yo * STRIDE_Y;
    xi -= PAD_LEFT;
    yi -= PAD_TOP;

    int d = 0;
#if DEPTH_MULTIPLIER != 1
    for(; d < DEPTH_MULTIPLIER; d++)
#endif // DEPTH_MULTIPLIER != 1
    {
        TILE(ACC_DATA_TYPE, M0, N0, c);

        // Reset accumulators
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            c[i].v = 0;
        })

        LOOP_UNROLLING(int, yk, 0, 1, _IWEI_HEIGHT,
        {
            TILE(SRC_DATA_TYPE, _IM0_A, _IN0_A, a);

            LOOP_UNROLLING(int, i, 0, 1, _IM0_A,
            {
                a[i].v = ZERO_VALUE;
            })

            // Load tile from the src tensor (TILE A)
            T_LOAD_NHWC_WITH_DILATION(SRC_DATA_TYPE, 1, _IM0_A, _IN0_A, SRC_TENSOR_TYPE, src, bout, yi + yk * DILATION_Y, xi, cout, _ISRC_WIDTH, _ISRC_HEIGHT, DILATION_X, 1, src_stride_y, _IBOUNDARY_CHECK, a);

            TILE(WEI_DATA_TYPE, _IM0_B, _IN0_B, b);

            // Load tile from the weights tensor (TILE B)
            T_LOAD(WEI_DATA_TYPE, _IM0_B, _IN0_B, WEI_TENSOR_TYPE, wei, cout * DEPTH_MULTIPLIER + d, yk * _IM0_B, 1, wei_stride_y, b);

            // Optimized path for STRIDE_X == 1
            // If M0 != 1, we can skip the common loads between the two applied kernels on the X (WIDTH) dimension
            LOOP_UNROLLING(int, m0, 0, 1, M0,
            {
                LOOP_UNROLLING(int, n0, 0, 1, N0,
                {
#if _IWEI_WIDTH <= 16
                    // Optimized path for the dot instruction
                    TILE(SRC_DATA_TYPE, 1, _IWEI_WIDTH, x0);
                    TILE(WEI_DATA_TYPE, 1, _IWEI_WIDTH, y0);
                    ACC_DATA_TYPE offset_a = 0;
                    ACC_DATA_TYPE offset_b = 0;

                    LOOP_UNROLLING(int, xk, 0, 1, _IWEI_WIDTH,
                    {
                        x0[0].s[xk] = a[xk + m0].s[n0];
                        y0[0].s[xk] = b[xk].s[n0];
                    })
                    DOT_PRODUCT_INTEGER8(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, _IWEI_WIDTH, x0[0].v, y0[0].v, c[m0].s[n0]);
                    REDUCE_INTEGER8(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, _IWEI_WIDTH, x0[0].v, offset_a);
                    REDUCE_INTEGER8(SRC_DATA_TYPE, WEI_DATA_TYPE, ACC_DATA_TYPE, _IWEI_WIDTH, y0[0].v, offset_b);
                    c[m0].s[n0] += offset_a * (ACC_DATA_TYPE)WEI_OFFSET + offset_b * (ACC_DATA_TYPE)SRC_OFFSET;
#else  // _IWEI_WIDTH <= 16
                    LOOP_UNROLLING(int, xk, 0, 1, _IWEI_WIDTH,
                    {
                        c[m0].s[n0] += ((ACC_DATA_TYPE)a[xk + m0].s[n0] + (ACC_DATA_TYPE)SRC_OFFSET) * ((ACC_DATA_TYPE)b[xk].s[n0] + (ACC_DATA_TYPE)WEI_OFFSET);
                    })
#endif // _IWEI_WIDTH <= 16
                })
            })
        })

#if _IWEI_WIDTH <= 16
        T_ADD_CONSTANT(ACC_DATA_TYPE, M0, N0, c, (_IWEI_WIDTH * _IWEI_HEIGHT * SRC_OFFSET * WEI_OFFSET), c);
#endif // _IWEI_WIDTH <= 16

#if defined(HAS_BIAS)
        TILE(BIA_DATA_TYPE, 1, N0, bias0);

        // Load bias
        T_LOAD(BIA_DATA_TYPE, 1, N0, BUFFER, bia, cout * DEPTH_MULTIPLIER + d, 0, 0, 0, bias0);

        // c = c + bias[broadcasted]
        T_ADD_BROADCAST_X(ACC_DATA_TYPE, M0, N0, c, bias0, c);
#endif // HAS_BIAS

#define T_LOAD_MULTIPLIERS_SHIFT(QUANTIZATION_TYPE) T_LOAD_MULTIPLIERS_SHIFT_STR(QUANTIZATION_TYPE)
#define T_LOAD_MULTIPLIERS_SHIFT_STR(QUANTIZATION_TYPE) T_LOAD_MULTIPLIERS_SHIFT_##QUANTIZATION_TYPE()

#define T_LOAD_MULTIPLIERS_SHIFT_PER_TENSOR() \
    ({})

#define T_LOAD_MULTIPLIERS_SHIFT_PER_CHANNEL()                                                                           \
    TILE(DST_MULTIPLIERS_DATA_TYPE, 1, N0, multipliers);                                                                 \
    TILE(DST_SHIFTS_DATA_TYPE, 1, N0, shifts);                                                                           \
    T_LOAD(DST_MULTIPLIERS_DATA_TYPE, 1, N0, BUFFER, dst_multipliers, cout *DEPTH_MULTIPLIER + d, 0, 0, 0, multipliers); \
    T_LOAD(DST_SHIFTS_DATA_TYPE, 1, N0, BUFFER, dst_shifts, cout *DEPTH_MULTIPLIER + d, 0, 0, 0, shifts);

        T_LOAD_MULTIPLIERS_SHIFT(QUANTIZATION_TYPE);

        // Quantize the tile
        TILE(DST_DATA_TYPE, M0, N0, cq);
        T_QUANTIZE8(ACC_DATA_TYPE, DST_DATA_TYPE, QUANTIZATION_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, c, multipliers, shifts, cq);

        // Perform activation
        T_ACTIVATION_QUANTIZED(DST_DATA_TYPE, M0, N0, ACTIVATION_TYPE, DST_OFFSET, A_VAL, B_VAL, cq, cq);

        TILE(uint, M0, 1, dst_indirect_y);

        bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

        // Calculate the destination indirect Y
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            dst_indirect_y[i].v = min(xo + i, (int)(_IDST_WIDTH) - 1) + yo *_IDST_WIDTH;
            dst_indirect_y[i].v += bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);
        })

        // Store the tile in reverse order so the invalid values are overwritten with the valid ones
        T_STORE_INDIRECT_WIDTH_SELECT(DST_DATA_TYPE, M0, N0, PARTIAL_N0, DST_TENSOR_TYPE, dst, cout * DEPTH_MULTIPLIER + d, dst_stride_y, x_cond, cq, dst_indirect_y);
    }
}
#endif // defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DST_WIDTH) && defined(DST_HEIGHT) && defined(WEI_WIDTH) && defined(WEI_HEIGHT) && defined(N0) && defined(M0) && defined(DILATION_X) && defined(DILATION_Y) && defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP)