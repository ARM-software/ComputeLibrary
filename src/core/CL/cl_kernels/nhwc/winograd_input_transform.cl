/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#define OUTPUT_ROW_4x4_5x5(out, tmp, comm_fact)                     \
    ({                                                              \
        comm_fact.s0 = tmp.s2 - 4.25f * tmp.s4 + tmp.s6;            \
        comm_fact.s1 = tmp.s1 - 4.25f * tmp.s3 + tmp.s5;            \
        comm_fact.s2 = 2.5f * tmp.s3;                               \
        comm_fact.s3 = 0.5f * tmp.s1 + 2.f * tmp.s5 - comm_fact.s2; \
        comm_fact.s4 = 0.25f * tmp.s2 - 1.25f * tmp.s4 + tmp.s6;    \
        comm_fact.s5 = 4.f * tmp.s2 + tmp.s6 - 5.f * tmp.s4;        \
        comm_fact.s6 = 2.f * tmp.s1 + 0.5f * tmp.s5 - comm_fact.s2; \
        \
        out.s0 = tmp.s0 - tmp.s6 + 5.25f * tmp.s4 - 5.25f * tmp.s2; \
        out.s1 = comm_fact.s0 + comm_fact.s1;                       \
        out.s2 = comm_fact.s0 - comm_fact.s1;                       \
        out.s3 = comm_fact.s3 + comm_fact.s4;                       \
        out.s4 = comm_fact.s4 - comm_fact.s3;                       \
        out.s5 = comm_fact.s5 + comm_fact.s6;                       \
        out.s6 = comm_fact.s5 - comm_fact.s6;                       \
        out.s7 = tmp.s7 - tmp.s1 + 5.25f * tmp.s3 - 5.25f * tmp.s5; \
    })

#define OUTPUT_ROW_2x2_7x7(out, tmp, comm_fact)                                                    \
    ({                                                                                             \
        comm_fact.s0 = 36.0f * tmp.s2 - 13.0f * tmp.s4 + tmp.s6;                                   \
        comm_fact.s1 = 36.0f * tmp.s1 - 13.0f * tmp.s3 + 1.0f * tmp.s5;                            \
        comm_fact.s2 = 9.0f * tmp.s2 - 10.0f * tmp.s4 + tmp.s6;                                    \
        comm_fact.s3 = 18.0f * tmp.s1 - 20.0f * tmp.s3 + 2.0f * tmp.s5;                            \
        comm_fact.s4 = 4.0f * tmp.s2 - 5.0f * tmp.s4 + tmp.s6;                                     \
        comm_fact.s5 = 12.0f * tmp.s1 - 15.0f * tmp.s3 + 3.0f * tmp.s5;                            \
        out.s0       = -36.0f * tmp.s0 + 49.0f * tmp.s2 + -14.0f * tmp.s4 + tmp.s6;                \
        out.s1       = comm_fact.s0 - comm_fact.s1;                                                \
        out.s2       = comm_fact.s0 + comm_fact.s1;                                                \
        out.s3       = comm_fact.s2 - comm_fact.s3;                                                \
        out.s4       = comm_fact.s2 + comm_fact.s3;                                                \
        out.s5       = comm_fact.s4 - comm_fact.s5;                                                \
        out.s6       = comm_fact.s4 + comm_fact.s5;                                                \
        out.s7       = -36.0f * tmp.s1 + 0.0f * tmp.s2 + 49.0f * tmp.s3 - 14.0f * tmp.s5 + tmp.s7; \
    })

#if defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)

#if defined(NHWC) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(NUM_TILES_X) && defined(NUM_TILES_Y)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the output tile is 4x4, 4x1 or 1x4, the filter size 3x3, 3x1 or 1x3 and the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x4_3x3_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 6, 1, in);
    TILE(DATA_TYPE, 6, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 6, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 6, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    TILE(DATA_TYPE, 6, 1, com);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        in[i].v *= 4.0f;
    })

    com[0].v = in[2].v - 4.f * in[0].v;
    com[1].v = in[3].v - 4.f * in[1].v;
    com[2].v = in[4].v - 4.f * in[2].v;
    com[3].v = in[5].v - 4.f * in[3].v;
    com[4].v = in[3].v - in[1].v;
    com[4].v = com[4].v + com[4].v;
    com[5].v = in[4].v - in[2].v;

    out[0].v = com[2].v - com[0].v;
    out[1].v = com[2].v + com[1].v;
    out[2].v = com[2].v - com[1].v;
    out[3].v = com[5].v + com[4].v;
    out[4].v = com[5].v - com[4].v;
    out[5].v = com[3].v - com[1].v;

    TILE(uint, 6, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 6;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 6, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 36, 1, in);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 6, 6, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 6, 1, com);
    TILE(DATA_TYPE, 36, 1, tmp);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        com[0].v         = in[2 * 6 + i].v - (DATA_TYPE)4.0f * in[0 * 6 + i].v;
        com[1].v         = in[3 * 6 + i].v - (DATA_TYPE)4.0f * in[1 * 6 + i].v;
        com[2].v         = in[4 * 6 + i].v - (DATA_TYPE)4.0f * in[2 * 6 + i].v;
        com[3].v         = in[5 * 6 + i].v - (DATA_TYPE)4.0f * in[3 * 6 + i].v;
        com[4].v         = in[3 * 6 + i].v - in[1 * 6 + i].v;
        com[4].v         = com[4].v + com[4].v;
        com[5].v         = in[4 * 6 + i].v - in[2 * 6 + i].v;
        tmp[i + 0 * 6].v = com[2].v - com[0].v;
        tmp[i + 1 * 6].v = com[2].v + com[1].v;
        tmp[i + 2 * 6].v = com[2].v - com[1].v;
        tmp[i + 3 * 6].v = com[5].v + com[4].v;
        tmp[i + 4 * 6].v = com[5].v - com[4].v;
        tmp[i + 5 * 6].v = com[3].v - com[1].v;
    })

    TILE(DATA_TYPE, 36, 1, out);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        com[0].v         = tmp[i * 6 + 2].v - 4.f *tmp[i * 6 + 0].v;
        com[1].v         = tmp[i * 6 + 3].v - 4.f *tmp[i * 6 + 1].v;
        com[2].v         = tmp[i * 6 + 4].v - 4.f *tmp[i * 6 + 2].v;
        com[3].v         = tmp[i * 6 + 5].v - 4.f *tmp[i * 6 + 3].v;
        com[4].v         = tmp[i * 6 + 3].v - tmp[i * 6 + 1].v;
        com[4].v         = com[4].v + com[4].v;
        com[5].v         = tmp[i * 6 + 4].v - tmp[i * 6 + 2].v;
        out[i * 6 + 0].v = com[2].v - com[0].v;
        out[i * 6 + 1].v = com[2].v + com[1].v;
        out[i * 6 + 2].v = com[2].v - com[1].v;
        out[i * 6 + 3].v = com[5].v + com[4].v;
        out[i * 6 + 4].v = com[5].v - com[4].v;
        out[i * 6 + 5].v = com[3].v - com[1].v;
    })

    // Compute destination address
    TILE(uint, 36, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 36;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 36, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 5x5/5x1 or 1x5 and the output tile is 4x4/4x1 or 1x4 when the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x4_5x5_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 8, 1, in);
    TILE(DATA_TYPE, 8, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 8, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    TILE(DATA_TYPE, 1, 8, com);

    com[0].s[0] = in[2].v - 4.25f * in[4].v + in[6].v;
    com[0].s[1] = in[1].v - 4.25f * in[3].v + in[5].v;
    com[0].s[2] = 0.5f * in[1].v - 2.5f * in[3].v + 2.0f * in[5].v;
    com[0].s[3] = 0.25f * in[2].v - 1.25f * in[4].v + in[6].v;
    com[0].s[4] = 4.0f * in[2].v - 5.0f * in[4].v + in[6].v;
    com[0].s[5] = 2.0f * in[1].v - 2.5f * in[3].v + 0.5f * in[5].v;
    out[0].s[0] = in[0].v - 5.25f * in[2].v + 5.25f * in[4].v - in[6].v;
    out[1].s[0] = com[0].s[0] + com[0].s[1];
    out[2].s[0] = com[0].s[0] - com[0].s[1];
    out[3].s[0] = com[0].s[3] + com[0].s[2];
    out[4].s[0] = com[0].s[3] - com[0].s[2];
    out[5].s[0] = com[0].s[4] + com[0].s[5];
    out[6].s[0] = com[0].s[4] - com[0].s[5];
    out[7].s[0] = -in[1].v + 5.25f * in[3].v - 5.25f * in[5].v + in[7].v;

    TILE(uint, 8, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 8;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 8, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 64, 1, in);
    TILE(DATA_TYPE, 64, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 8, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 8, 8, com);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[i] = in[2 * 8 + i].s[0] - (DATA_TYPE)4.25f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];                                    // x
        com[1].s[i] = in[1 * 8 + i].s[0] - (DATA_TYPE)4.25f * in[3 * 8 + i].s[0] + in[5 * 8 + i].s[0];                                    // x
        com[2].s[i] = (DATA_TYPE)0.25f * in[2 * 8 + i].s[0] - (DATA_TYPE)1.25f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];                 // x
        com[3].s[i] = (DATA_TYPE)0.5f * in[1 * 8 + i].s[0] - (DATA_TYPE)2.5f * in[3 * 8 + i].s[0] + (DATA_TYPE)2.0f * in[5 * 8 + i].s[0]; // x
        com[4].s[i] = (DATA_TYPE)4.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)5.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[5].s[i] = (DATA_TYPE)2.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)2.5f * in[3 * 8 + i].s[0] + (DATA_TYPE)0.5f * in[5 * 8 + i].s[0];
        com[6].s[i] = in[0 * 8 + i].s[0] - (DATA_TYPE)5.25f * in[2 * 8 + i].s[0] + (DATA_TYPE)5.25f * in[4 * 8 + i].s[0] - in[6 * 8 + i].s[0];
        com[7].s[i] = -in[1 * 8 + i].s[0] + (DATA_TYPE)5.25f * in[3 * 8 + i].s[0] - (DATA_TYPE)5.25f * in[5 * 8 + i].s[0] + in[7 * 8 + i].s[0];
    })

    TILE(DATA_TYPE, 8, 8, tmp);
    tmp[0].v = com[6].v;
    tmp[1].v = com[0].v + com[1].v;
    tmp[2].v = com[0].v - com[1].v;
    tmp[3].v = com[2].v + com[3].v;
    tmp[4].v = com[2].v - com[3].v;
    tmp[5].v = com[4].v + com[5].v;
    tmp[6].v = com[4].v - com[5].v;
    tmp[7].v = com[7].v;

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[0]         = tmp[i].s[2] - 4.25f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[1]         = tmp[i].s[1] - 4.25f * tmp[i].s[3] + tmp[i].s[5];
        com[0].s[2]         = 0.5f * tmp[i].s[1] - 2.5f * tmp[i].s[3] + 2.0f * tmp[i].s[5];
        com[0].s[3]         = 0.25f * tmp[i].s[2] - 1.25f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[4]         = 4.0f * tmp[i].s[2] - 5.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[5]         = 2.0f * tmp[i].s[1] - 2.5f * tmp[i].s[3] + 0.5f * tmp[i].s[5];
        out[i * 8 + 0].s[0] = tmp[i].s[0] - 5.25f * tmp[i].s[2] + 5.25f * tmp[i].s[4] - tmp[i].s[6];
        out[i * 8 + 1].s[0] = com[0].s[0] + com[0].s[1];
        out[i * 8 + 2].s[0] = com[0].s[0] - com[0].s[1];
        out[i * 8 + 3].s[0] = com[0].s[3] + com[0].s[2];
        out[i * 8 + 4].s[0] = com[0].s[3] - com[0].s[2];
        out[i * 8 + 5].s[0] = com[0].s[4] + com[0].s[5];
        out[i * 8 + 6].s[0] = com[0].s[4] - com[0].s[5];
        out[i * 8 + 7].s[0] = -tmp[i].s[1] + 5.25f * tmp[i].s[3] - 5.25f * tmp[i].s[5] + tmp[i].s[7];
    })

    TILE(uint, 64, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 64;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 64, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 7x7/7x1/1x7 and the output tile is 2x2/7x1/1x7 when the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_2x2_7x7_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 8, 1, in);
    TILE(DATA_TYPE, 8, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 8, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v *= (DATA_TYPE) - 36.0f;
    })

    TILE(DATA_TYPE, 1, 8, com) = { { { 0 } } };

    com[0].s[0] = 36.0f * in[2].v - 13.0f * in[4].v + in[6].v;
    com[0].s[1] = 36.0f * in[1].v - 13.0f * in[3].v + 1.0f * in[5].v;
    com[0].s[2] = 9.0f * in[2].v - 10.0f * in[4].v + in[6].v;
    com[0].s[3] = 18.0f * in[1].v - 20.0f * in[3].v + 2.0f * in[5].v;
    com[0].s[4] = 4.0f * in[2].v - 5.0f * in[4].v + in[6].v;
    com[0].s[5] = 12.0f * in[1].v - 15.0f * in[3].v + 3.0f * in[5].v;
    out[0].s[0] = -36.0f * in[0].v + 49.0f * in[2].v + -14.0f * in[4].v + in[6].v;
    out[1].s[0] = com[0].s[0] - com[0].s[1];
    out[2].s[0] = com[0].s[0] + com[0].s[1];
    out[3].s[0] = com[0].s[2] - com[0].s[3];
    out[4].s[0] = com[0].s[2] + com[0].s[3];
    out[5].s[0] = com[0].s[4] - com[0].s[5];
    out[6].s[0] = com[0].s[4] + com[0].s[5];
    out[7].s[0] = -36.0f * in[1].v + 0.0f * in[2].v + 49.0f * in[3].v - 14.0f * in[5].v + in[7].v;

    TILE(uint, 8, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 8;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 8, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 64, 1, in);
    TILE(DATA_TYPE, 64, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 8, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 8, 8, com);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[i] = (DATA_TYPE)36.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)13.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[1].s[i] = (DATA_TYPE)36.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)13.0f * in[3 * 8 + i].s[0] + in[5 * 8 + i].s[0];
        com[2].s[i] = (DATA_TYPE)9.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)10.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[3].s[i] = (DATA_TYPE)18.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)20.0f * in[3 * 8 + i].s[0] + (DATA_TYPE)2.0f * in[5 * 8 + i].s[0];
        com[4].s[i] = (DATA_TYPE)4.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)5.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[5].s[i] = (DATA_TYPE)12.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)15.0f * in[3 * 8 + i].s[0] + (DATA_TYPE)3.0f * in[5 * 8 + i].s[0];
        com[6].s[i] = (DATA_TYPE)49.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)36.0f * in[0 * 8 + i].s[0] + in[6 * 8 + i].s[0] - (DATA_TYPE)14.0f * in[4 * 8 + i].s[0];
        com[7].s[i] = (DATA_TYPE)49.0f * in[3 * 8 + i].s[0] - (DATA_TYPE)36.0f * in[1 * 8 + i].s[0] + in[7 * 8 + i].s[0] - (DATA_TYPE)14.0f * in[5 * 8 + i].s[0];
    })

    TILE(DATA_TYPE, 8, 8, tmp);
    tmp[0].v = com[6].v;
    tmp[1].v = com[0].v - com[1].v;
    tmp[2].v = com[0].v + com[1].v;
    tmp[3].v = com[2].v - com[3].v;
    tmp[4].v = com[2].v + com[3].v;
    tmp[5].v = com[4].v - com[5].v;
    tmp[6].v = com[4].v + com[5].v;
    tmp[7].v = com[7].v;

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[0]         = 36.0f * tmp[i].s[2] - 13.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[1]         = 36.0f * tmp[i].s[1] - 13.0f * tmp[i].s[3] + 1.0f * tmp[i].s[5];
        com[0].s[2]         = 9.0f * tmp[i].s[2] - 10.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[3]         = 18.0f * tmp[i].s[1] - 20.0f * tmp[i].s[3] + 2.0f * tmp[i].s[5];
        com[0].s[4]         = 4.0f * tmp[i].s[2] - 5.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[5]         = 12.0f * tmp[i].s[1] - 15.0f * tmp[i].s[3] + 3.0f * tmp[i].s[5];
        out[i * 8 + 0].s[0] = -36.0f * tmp[i].s[0] + 49.0f * tmp[i].s[2] + -14.0f * tmp[i].s[4] + tmp[i].s[6];
        out[i * 8 + 1].s[0] = com[0].s[0] - com[0].s[1];
        out[i * 8 + 2].s[0] = com[0].s[0] + com[0].s[1];
        out[i * 8 + 3].s[0] = com[0].s[2] - com[0].s[3];
        out[i * 8 + 4].s[0] = com[0].s[2] + com[0].s[3];
        out[i * 8 + 5].s[0] = com[0].s[4] - com[0].s[5];
        out[i * 8 + 6].s[0] = com[0].s[4] + com[0].s[5];
        out[i * 8 + 7].s[0] = -36.0f * tmp[i].s[1] + 0.0f * tmp[i].s[2] + 49.0f * tmp[i].s[3] - 14.0f * tmp[i].s[5] + tmp[i].s[7];
    })

    TILE(uint, 64, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        dst_indirect_y[i].v = mout + i *_INUM_TILES_X *_INUM_TILES_Y;
        dst_indirect_y[i].v += bout *_INUM_TILES_X *_INUM_TILES_Y * 64;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 64, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x1_3x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 5x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x1_5x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 7x1 and the output tile is 2x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_2x1_7x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_2x2_7x7_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x4 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x4_1x3_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x5 and the output tile is 1x4 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x4_1x5_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x7 and the output tile is 1x2 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x2_1x7_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_2x2_7x7_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}
#endif // defined(NHWC) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(NUM_TILES_X) && defined(NUM_TILES_Y)
#endif // defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)
