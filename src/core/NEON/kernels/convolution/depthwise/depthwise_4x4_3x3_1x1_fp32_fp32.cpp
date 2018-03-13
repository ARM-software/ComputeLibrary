/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/convolution/depthwise/impl_fp32_fp32.hpp"

namespace depthwise
{
using Conv = DepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float>;
using ConvImpl = DepthwiseConvolutionImpl<4, 4, 3, 3, 1, 1, float, float>;

#ifdef __aarch64__

template <>
template <>
void ConvImpl::process_tile<true, 0, 0, 0, 0, 0, 0>(
  const int n_channels,
  const float* const weights,
  const int weight_row_stride,
  const int weight_col_stride,
  const float* const inptr,
  const int in_row_stride,
  const int in_col_stride,
  float* const outptr,
  const int out_row_stride,
  const int out_col_stride,
  const int, const int, const int, const int, const int, const int
)
{
  constexpr auto inner_tile_rows = DWC::inner_tile_rows;
  constexpr auto inner_tile_cols = DWC::inner_tile_cols;
  constexpr auto kernel_rows = DWC::kernel_rows;
  constexpr auto kernel_cols = DWC::kernel_cols;
  constexpr auto output_tile_rows = DWC::output_tile_rows;
  constexpr auto output_tile_cols = DWC::output_tile_cols;
  constexpr auto stride_rows = DWC::stride_rows;
  constexpr auto stride_cols = DWC::stride_cols;

  // Extract parameters
  const int in_pad_top = 0;
  const int in_pad_left = 0;
  const int in_pad_bottom = 0;
  const int in_pad_right = 0;
  const int out_pad_bottom = 0;
  const int out_pad_right = 0;

  // Compute valid ranges of the tile
  const int in_cells_i = inner_tile_rows - in_pad_bottom;
  const int in_cells_j = inner_tile_cols - in_pad_right;
  const int out_cells_i = output_tile_rows - out_pad_bottom;
  const int out_cells_j = output_tile_cols - out_pad_right;

  // Copy pointers
  const float *uptr0 = inptr;
  const float *wptr0 = weights;
  float *vptr0 = outptr;
  const bool same_strides = (
    weight_col_stride == in_col_stride &&
    weight_col_stride == out_col_stride
  );

  int channels_remaining = n_channels;
  if (channels_remaining >= 4 && same_strides)
  {
    int c4_rem = channels_remaining / 4;
    channels_remaining %= 4;
    const int prefetch_depth = 8;

    asm volatile (
      "qW22 .req q0\n" "vW22 .req v0\n"
      "qU64 .req q1\n" "qU35 .req q1\n" "qV41 .req q1\n"
      "vU64 .req v1\n" "vU35 .req v1\n" "vV41 .req v1\n"
      "qU34 .req q2\n" "qU21 .req q2\n" "qV43 .req q2\n"
      "vU34 .req v2\n" "vU21 .req v2\n" "vV43 .req v2\n"
      "qW21 .req q3\n" "vW21 .req v3\n"
      "qU24 .req q4\n" "qU54 .req q4\n" "qV31 .req q4\n"
      "vU24 .req v4\n" "vU54 .req v4\n" "vV31 .req v4\n"
      "qV12 .req q5\n" "qU61 .req q5\n" "vV12 .req v5\n" "vU61 .req v5\n"
      "qU26 .req q6\n" "qV32 .req q6\n" "vU26 .req v6\n" "vV32 .req v6\n"
      "qU36 .req q7\n" "qU51 .req q7\n" "qU66 .req q7\n" "qU12 .req q7\n"
      "vU36 .req v7\n" "vU51 .req v7\n" "vU66 .req v7\n" "vU12 .req v7\n"
      "qV14 .req q8\n" "qV11 .req q8\n" "qU65 .req q8\n"
      "vV14 .req v8\n" "vV11 .req v8\n" "vU65 .req v8\n"
      "qU15 .req q9\n" "qU22 .req q9\n" "qU45 .req q9\n"
      "vU15 .req v9\n" "vU22 .req v9\n" "vU45 .req v9\n"
      "qV22 .req q10\n" "qU14 .req q10\n" "vV22 .req v10\n" "vU14 .req v10\n"
      "qU44 .req q11\n" "qU43 .req q11\n" "qU11 .req q11\n"
      "vU44 .req v11\n" "vU43 .req v11\n" "vU11 .req v11\n"
      "qV24 .req q12\n" "qV42 .req q12\n" "vV24 .req v12\n" "vV42 .req v12\n"
      "qW31 .req q13\n" "vW31 .req v13\n" "qW13 .req q14\n" "vW13 .req v14\n"
      "qU33 .req q15\n" "qU62 .req q15\n" "qU25 .req q15\n" "qU56 .req q15\n"
      "vU33 .req v15\n" "vU62 .req v15\n" "vU25 .req v15\n" "vU56 .req v15\n"
      "qW33 .req q16\n" "vW33 .req v16\n"
      "qU42 .req q17\n" "qU16 .req q17\n" "qV44 .req q17\n"
      "vU42 .req v17\n" "vU16 .req v17\n" "vV44 .req v17\n"
      "qU63 .req q18\n" "qU31 .req q18\n" "qV34 .req q18\n"
      "vU63 .req v18\n" "vU31 .req v18\n" "vV34 .req v18\n"
      "qW11 .req q19\n" "vW11 .req v19\n" "qU41 .req q20\n" "qV13 .req q20\n"
      "vU41 .req v20\n" "vV13 .req v20\n" "qV33 .req q21\n" "vV33 .req v21\n"
      "qU46 .req q22\n" "qU32 .req q22\n" "qU13 .req q22\n"
      "vU46 .req v22\n" "vU32 .req v22\n" "vU13 .req v22\n" "qW23 .req q23\n"
      "vW23 .req v23\n" "qV23 .req q24\n" "vV23 .req v24\n"
      "qV21 .req q25\n" "qU55 .req q25\n" "vV21 .req v25\n" "vU55 .req v25\n"
      "qW12 .req q26\n" "vW12 .req v26\n" "qW32 .req q27\n" "vW32 .req v27\n"
      "qU23 .req q28\n" "qU52 .req q28\n"
      "vU23 .req v28\n" "vU52 .req v28\n" "qU53 .req q29\n" "vU53 .req v29\n"

      "uptr1 .req x0\n"
      "uptr2 .req x1\n"
      "uptr3 .req x2\n"
      "uptr4 .req x3\n"
      "uptr5 .req x4\n"

      "vptr1 .req x5\n"
      "vptr2 .req x6\n"
      "vptr3 .req x7\n"

      "wptr1 .req x8\n"
      "wptr2 .req x9\n"

      // Prepare pointers and strides
      "add uptr1, %x[uptr0], %x[u_row_stride]\n"
      "add uptr2,    uptr1 , %x[u_row_stride]\n"
      "add uptr3,    uptr2 , %x[u_row_stride]\n"
      "add uptr4,    uptr3 , %x[u_row_stride]\n"
      "add uptr5,    uptr4 , %x[u_row_stride]\n"

      "add vptr1, %x[vptr0], %x[v_row_stride]\n"
      "add vptr2,    vptr1 , %x[v_row_stride]\n"
      "add vptr3,    vptr2 , %x[v_row_stride]\n"

      "add wptr1, %x[wptr0], %x[w_row_stride]\n"
      "add wptr2,    wptr1 , %x[w_row_stride]\n"

      // Load initial operands
      "ldr qU16, [%x[uptr0], %x[uvw_col_stride5]]\n"
      "ldr qW13, [%x[wptr0], %x[uvw_col_stride2]]\n"
      "subs %x[c4_rem], %x[c4_rem], #1\n"
      "ldr qU15, [%x[uptr0], %x[uvw_col_stride4]]\n"
      "ldr qW23, [wptr1, %x[uvw_col_stride2]]\n"
      "ldr qU14, [%x[uptr0], %x[uvw_col_stride3]]\n"
      "ldr qW33, [wptr2, %x[uvw_col_stride2]]\n"
      "ldr qU26, [uptr1, %x[uvw_col_stride5]]\n"
      "ldr qW12, [%x[wptr0], %x[uvw_col_stride1]]\n"
      "ldr qU25, [uptr1, %x[uvw_col_stride4]]\n"
      "ldr qW22, [wptr1, %x[uvw_col_stride1]]\n"
      "ldr qU36, [uptr2, %x[uvw_col_stride5]]\n"
      "ldr qW32, [wptr2, %x[uvw_col_stride1]]\n"
      "ldr qW11, [%x[wptr0]], #0x10\n"
      "fmul vV14.4s, vU16.4s, vW13.4s\n"
      "ldr qU24, [uptr1, %x[uvw_col_stride3]]\n"
      "fmul vV13.4s, vU15.4s, vW13.4s\n"
      "ldr qW31, [wptr2], #0x10\n"
      "fmla vV14.4s, vU15.4s, vW12.4s\n"
      "ldr qW21, [wptr1], #0x10\n"
      "fmul vV12.4s, vU14.4s, vW13.4s\n"
      "ldr qU34, [uptr2, %x[uvw_col_stride3]]\n"
      "fmla vV13.4s, vU14.4s, vW12.4s\n"
      "ldr qU46, [uptr3, %x[uvw_col_stride5]]\n"
      "fmla vV14.4s, vU14.4s, vW11.4s\n"
      "ldr qU45, [uptr3, %x[uvw_col_stride4]]\n"
      "fmla vV14.4s, vU26.4s, vW23.4s\n"
      "ldr qU35, [uptr2, %x[uvw_col_stride4]]\n"
      "fmul vV24.4s, vU26.4s, vW13.4s\n"
      "ldr qU44, [uptr3, %x[uvw_col_stride3]]\n"
      "fmla vV13.4s, vU25.4s, vW23.4s\n"
      "beq 2f\n"  // Single iteration only

      "1:"  // Loop body
        "fmla vV14.4s, vU25.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[wptr0], %[prftch]]\n"
        "fmul vV23.4s, vU25.4s, vW13.4s\n"
        "prfm pldl1keep, [%x[wptr0], %x[prftch_uvw_col_stride1]]\n"
        "fmla vV24.4s, vU25.4s, vW12.4s\n"
        "ldr qU56, [uptr4, %x[uvw_col_stride5]]\n"
        "fmla vV12.4s, vU24.4s, vW23.4s\n"
        "prfm pldl1keep, [%x[wptr0], %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV13.4s, vU24.4s, vW22.4s\n"
        "prfm pldl1keep, [   wptr1 , %[prftch]]\n"
        "fmla vV14.4s, vU24.4s, vW21.4s\n"
        "prfm pldl1keep, [   wptr1 , %x[prftch_uvw_col_stride1]]\n"
        "fmul vV22.4s, vU24.4s, vW13.4s\n"
        "prfm pldl1keep, [   wptr1 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV23.4s, vU24.4s, vW12.4s\n"
        "prfm pldl1keep, [   wptr2 , %x[prftch]]\n"
        "fmla vV24.4s, vU24.4s, vW11.4s\n"
        "ldr qU55, [uptr4, %x[uvw_col_stride4]]\n"
        "fmla vV14.4s, vU36.4s, vW33.4s\n"
        "prfm pldl1keep, [   wptr2 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV24.4s, vU36.4s, vW23.4s\n"
        "prfm pldl1keep, [   wptr2 , %x[prftch_uvw_col_stride2] ]\n"
        "fmul vV34.4s, vU36.4s, vW13.4s\n"
        "ldr qU54, [uptr4, %x[uvw_col_stride3]]\n"
        "fmla vV13.4s, vU35.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV14.4s, vU35.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV23.4s, vU35.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV24.4s, vU35.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[prftch_uvw_col_stride4] ]\n"
        "fmul vV33.4s, vU35.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV34.4s, vU35.4s, vW12.4s\n"
        "ldr qU66, [uptr5, %x[uvw_col_stride5]]\n"
        "fmla vV12.4s, vU34.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr3 , %[prftch]]\n"
        "fmla vV13.4s, vU34.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV14.4s, vU34.4s, vW31.4s\n"
        "str qV14, [%x[vptr0], %x[uvw_col_stride3]]\n"
        "fmla vV22.4s, vU34.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV23.4s, vU34.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV24.4s, vU34.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[prftch_uvw_col_stride4] ]\n"
        "fmul vV32.4s, vU34.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV33.4s, vU34.4s, vW12.4s\n"
        "prfm pldl1keep, [   uptr4 , %[prftch]]\n"
        "fmla vV34.4s, vU34.4s, vW11.4s\n"
        "ldr qU65, [uptr5, %x[uvw_col_stride4]]\n"
        "fmla vV24.4s, vU46.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV34.4s, vU46.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[prftch_uvw_col_stride2] ]\n"
        "fmul vV44.4s, vU46.4s, vW13.4s\n"
        "ldr qU64, [uptr5, %x[uvw_col_stride3]]\n"
        "fmla vV23.4s, vU45.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV24.4s, vU45.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[prftch_uvw_col_stride4] ]\n"
        "fmla vV33.4s, vU45.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV34.4s, vU45.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr5 , %[prftch]]\n"
        "fmul vV43.4s, vU45.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV44.4s, vU45.4s, vW12.4s\n"
        "ldr qU13, [%x[uptr0], %x[uvw_col_stride2]]\n"
        "fmla vV22.4s, vU44.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV23.4s, vU44.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV24.4s, vU44.4s, vW31.4s\n"
        "str qV24, [vptr1, %x[uvw_col_stride3]]\n"
        "fmla vV32.4s, vU44.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[prftch_uvw_col_stride4] ]\n"
        "fmla vV33.4s, vU44.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV34.4s, vU44.4s, vW21.4s\n"
        "prfm pstl1keep, [%x[vptr0], %[prftch]]\n"
        "fmul vV42.4s, vU44.4s, vW13.4s\n"
        "prfm pstl1keep, [%x[vptr0], %x[prftch_uvw_col_stride1]]\n"
        "fmla vV43.4s, vU44.4s, vW12.4s\n"
        "prfm pstl1keep, [%x[vptr0], %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV44.4s, vU44.4s, vW11.4s\n"
        "ldr qU23, [uptr1, %x[uvw_col_stride2]]\n"
        "fmla vV34.4s, vU56.4s, vW33.4s\n"
        "prfm pstl1keep, [%x[vptr0], %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV44.4s, vU56.4s, vW23.4s\n"
        "ldr qU33, [uptr2, %x[uvw_col_stride2]]\n"
        "fmla vV33.4s, vU55.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr1 , %[prftch]]\n"
        "fmla vV34.4s, vU55.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr1 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV43.4s, vU55.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr1 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV44.4s, vU55.4s, vW22.4s\n"
        "ldr qU43, [uptr3, %x[uvw_col_stride2]]\n"
        "fmla vV32.4s, vU54.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr1 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV33.4s, vU54.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr2 , %[prftch]]\n"
        "fmla vV34.4s, vU54.4s, vW31.4s\n"
        "str qV34, [vptr2, %x[uvw_col_stride3]]\n"
        "fmla vV42.4s, vU54.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr2 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV43.4s, vU54.4s, vW22.4s\n"
        "prfm pstl1keep, [   vptr2 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV44.4s, vU54.4s, vW21.4s\n"
        "ldr qU53, [uptr4, %x[uvw_col_stride2]]\n"
        "fmla vV44.4s, vU66.4s, vW33.4s\n"
        "ldr qU63, [uptr5, %x[uvw_col_stride2]]\n"
        "fmla vV43.4s, vU65.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr2 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV44.4s, vU65.4s, vW32.4s\n"
        "ldr qU12, [%x[uptr0], %x[uvw_col_stride1]]\n"
        "fmla vV42.4s, vU64.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr3 , %[prftch]]\n"
        "fmla vV43.4s, vU64.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr3 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV44.4s, vU64.4s, vW31.4s\n"
        "str qV44, [vptr3, %x[uvw_col_stride3]]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "ldr qU22, [uptr1, %x[uvw_col_stride1]]\n"
        "fmla vV12.4s, vU13.4s, vW12.4s\n"
        "prfm pstl1keep, [   vptr3 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV13.4s, vU13.4s, vW11.4s\n"
        "ldr qU32, [uptr2, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr3 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV12.4s, vU23.4s, vW22.4s\n"
        "fmla vV13.4s, vU23.4s, vW21.4s\n"
        "fmul vV21.4s, vU23.4s, vW13.4s\n"
        "fmla vV22.4s, vU23.4s, vW12.4s\n"
        "fmla vV23.4s, vU23.4s, vW11.4s\n"
        "ldr qU42, [uptr3, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "fmla vV12.4s, vU33.4s, vW32.4s\n"
        "fmla vV13.4s, vU33.4s, vW31.4s\n"
        "str qV13, [%x[vptr0], %x[uvw_col_stride2]]\n"
        "fmla vV21.4s, vU33.4s, vW23.4s\n"
        "fmla vV22.4s, vU33.4s, vW22.4s\n"
        "fmla vV23.4s, vU33.4s, vW21.4s\n"
        "fmul vV31.4s, vU33.4s, vW13.4s\n"
        "fmla vV32.4s, vU33.4s, vW12.4s\n"
        "fmla vV33.4s, vU33.4s, vW11.4s\n"
        "ldr qU52, [uptr4, %x[uvw_col_stride1]]\n"
        "fmla vV21.4s, vU43.4s, vW33.4s\n"
        "fmla vV22.4s, vU43.4s, vW32.4s\n"
        "fmla vV23.4s, vU43.4s, vW31.4s\n"
        "str qV23, [vptr1, %x[uvw_col_stride2]]\n"
        "fmla vV31.4s, vU43.4s, vW23.4s\n"
        "fmla vV32.4s, vU43.4s, vW22.4s\n"
        "fmla vV33.4s, vU43.4s, vW21.4s\n"
        "fmul vV41.4s, vU43.4s, vW13.4s\n"
        "ldr qW13, [%x[wptr0], %x[uvw_col_stride2]]\n"
        "fmla vV42.4s, vU43.4s, vW12.4s\n"
        "fmla vV43.4s, vU43.4s, vW11.4s\n"
        "ldr qU62, [uptr5, %x[uvw_col_stride1]]\n"
        "fmla vV31.4s, vU53.4s, vW33.4s\n"
        "fmla vV32.4s, vU53.4s, vW32.4s\n"
        "fmla vV33.4s, vU53.4s, vW31.4s\n"
        "str qV33, [vptr2, %x[uvw_col_stride2]]\n"
        "fmla vV41.4s, vU53.4s, vW23.4s\n"
        "ldr qW23, [wptr1, %x[uvw_col_stride2]]\n"
        "fmla vV42.4s, vU53.4s, vW22.4s\n"
        "fmla vV43.4s, vU53.4s, vW21.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV41.4s, vU63.4s, vW33.4s\n"
        "ldr qW33, [wptr2, %x[uvw_col_stride2]]\n"
        "fmla vV42.4s, vU63.4s, vW32.4s\n"
        "prfm pldl1keep, [%x[uptr0], %[prftch]]\n"
        "fmla vV43.4s, vU63.4s, vW31.4s\n"
        "str qV43, [vptr3, %x[uvw_col_stride2]]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV12.4s, vU12.4s, vW11.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[prftch_uvw_col_stride1]]\n"
        "fmla vV12.4s, vU22.4s, vW21.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV21.4s, vU22.4s, vW12.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV22.4s, vU22.4s, vW11.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[prftch_uvw_col_stride4] ]\n"
        "fmla vV12.4s, vU32.4s, vW31.4s\n"
        "str qV12, [%x[vptr0], %x[uvw_col_stride1]]\n"
        "fmla vV21.4s, vU32.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV22.4s, vU32.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr1 , %[prftch]]\n"
        "fmla vV31.4s, vU32.4s, vW12.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[prftch_uvw_col_stride1]]\n"
        "fmla vV32.4s, vU32.4s, vW11.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[prftch_uvw_col_stride2] ]\n"
        "fmla vV22.4s, vU42.4s, vW31.4s\n"
        "str qV22, [vptr1, %x[uvw_col_stride1]]\n"
        "fmla vV31.4s, vU42.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[prftch_uvw_col_stride3] ]\n"
        "fmla vV32.4s, vU42.4s, vW21.4s\n"
        "subs %x[c4_rem], %x[c4_rem], #1\n"
        "fmla vV41.4s, vU42.4s, vW12.4s\n"
        "ldr qW12, [%x[wptr0], %x[uvw_col_stride1]]\n"
        "fmla vV42.4s, vU42.4s, vW11.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV31.4s, vU52.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[prftch_uvw_col_stride4] ]\n"
        "fmla vV32.4s, vU52.4s, vW31.4s\n"
        "str qV32, [vptr2, %x[uvw_col_stride1]]\n"
        "fmla vV41.4s, vU52.4s, vW22.4s\n"
        "ldr qW22, [wptr1, %x[uvw_col_stride1]]\n"
        "fmla vV42.4s, vU52.4s, vW21.4s\n"
        "ldr qU16, [%x[uptr0], %x[uvw_col_stride5]]\n"
        "fmla vV41.4s, vU62.4s, vW32.4s\n"
        "ldr qW32, [wptr2, %x[uvw_col_stride1]]\n"
        "fmla vV42.4s, vU62.4s, vW31.4s\n"
        "str qV42, [vptr3, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "ldr qU15, [%x[uptr0], %x[uvw_col_stride4]]\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "ldr qU14, [%x[uptr0], %x[uvw_col_stride3]]\n"
        "fmla vV21.4s, vU21.4s, vW11.4s\n"
        "ldr qU26, [uptr1, %x[uvw_col_stride5]]\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU31.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[prftch_uvw_col_stride5] ]\n"
        "fmla vV31.4s, vU31.4s, vW11.4s\n"
        "ldr qU25, [uptr1, %x[uvw_col_stride4]]\n"
        "fmla vV21.4s, vU41.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU41.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr2 , %[prftch]]\n"
        "fmla vV41.4s, vU41.4s, vW11.4s\n"
        "ldr qW11, [%x[wptr0]], #0x10\n"
        "fmla vV31.4s, vU51.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"
        "fmla vV41.4s, vU51.4s, vW21.4s\n"
        "ldr qU36, [uptr2, %x[uvw_col_stride5]]\n"
        "fmla vV41.4s, vU61.4s, vW31.4s\n"
        "str qV41, [vptr3], #0x10\n"
        "fmul vV14.4s, vU16.4s, vW13.4s\n"
        "ldr qU24, [uptr1, %x[uvw_col_stride3]]\n"
        "fmul vV13.4s, vU15.4s, vW13.4s\n"
        "ldr qW31, [wptr2], #0x10\n"
        "fmla vV14.4s, vU15.4s, vW12.4s\n"
        "ldr qW21, [wptr1], #0x10\n"
        "fmul vV12.4s, vU14.4s, vW13.4s\n"
        "ldr qU34, [uptr2, %x[uvw_col_stride3]]\n"
        "fmla vV13.4s, vU14.4s, vW12.4s\n"
        "ldr qU46, [uptr3, %x[uvw_col_stride5]]\n"
        "fmla vV14.4s, vU14.4s, vW11.4s\n"
        "ldr qU45, [uptr3, %x[uvw_col_stride4]]\n"
        "fmla vV14.4s, vU26.4s, vW23.4s\n"
        "ldr qU35, [uptr2, %x[uvw_col_stride4]]\n"
        "fmul vV24.4s, vU26.4s, vW13.4s\n"
        "ldr qU44, [uptr3, %x[uvw_col_stride3]]\n"
        "fmla vV13.4s, vU25.4s, vW23.4s\n"
        "bne 1b\n"

      "2:"  // Final iteration
        "fmla vV14.4s, vU25.4s, vW22.4s\n"
        "fmul vV23.4s, vU25.4s, vW13.4s\n"
        "fmla vV24.4s, vU25.4s, vW12.4s\n"
        "ldr qU56, [uptr4, %x[uvw_col_stride5]]\n"
        "fmla vV12.4s, vU24.4s, vW23.4s\n"
        "fmla vV13.4s, vU24.4s, vW22.4s\n"
        "fmla vV14.4s, vU24.4s, vW21.4s\n"
        "fmul vV22.4s, vU24.4s, vW13.4s\n"
        "fmla vV23.4s, vU24.4s, vW12.4s\n"
        "fmla vV24.4s, vU24.4s, vW11.4s\n"
        "ldr qU55, [uptr4, %x[uvw_col_stride4]]\n"
        "fmla vV14.4s, vU36.4s, vW33.4s\n"
        "fmla vV24.4s, vU36.4s, vW23.4s\n"
        "fmul vV34.4s, vU36.4s, vW13.4s\n"
        "ldr qU54, [uptr4, %x[uvw_col_stride3]]\n"
        "fmla vV13.4s, vU35.4s, vW33.4s\n"
        "fmla vV14.4s, vU35.4s, vW32.4s\n"
        "fmla vV23.4s, vU35.4s, vW23.4s\n"
        "fmla vV24.4s, vU35.4s, vW22.4s\n"
        "fmul vV33.4s, vU35.4s, vW13.4s\n"
        "fmla vV34.4s, vU35.4s, vW12.4s\n"
        "ldr qU66, [uptr5, %x[uvw_col_stride5]]\n"
        "fmla vV12.4s, vU34.4s, vW33.4s\n"
        "fmla vV13.4s, vU34.4s, vW32.4s\n"
        "fmla vV14.4s, vU34.4s, vW31.4s\n"
        "str qV14, [%x[vptr0], %x[uvw_col_stride3]]\n"
        "fmla vV22.4s, vU34.4s, vW23.4s\n"
        "fmla vV23.4s, vU34.4s, vW22.4s\n"
        "fmla vV24.4s, vU34.4s, vW21.4s\n"
        "fmul vV32.4s, vU34.4s, vW13.4s\n"
        "fmla vV33.4s, vU34.4s, vW12.4s\n"
        "fmla vV34.4s, vU34.4s, vW11.4s\n"
        "ldr qU65, [uptr5, %x[uvw_col_stride4]]\n"
        "fmla vV24.4s, vU46.4s, vW33.4s\n"
        "fmla vV34.4s, vU46.4s, vW23.4s\n"
        "fmul vV44.4s, vU46.4s, vW13.4s\n"
        "ldr qU64, [uptr5, %x[uvw_col_stride3]]\n"
        "fmla vV23.4s, vU45.4s, vW33.4s\n"
        "fmla vV24.4s, vU45.4s, vW32.4s\n"
        "fmla vV33.4s, vU45.4s, vW23.4s\n"
        "fmla vV34.4s, vU45.4s, vW22.4s\n"
        "fmul vV43.4s, vU45.4s, vW13.4s\n"
        "fmla vV44.4s, vU45.4s, vW12.4s\n"
        "ldr qU13, [%x[uptr0], %x[uvw_col_stride2]]\n"
        "fmla vV22.4s, vU44.4s, vW33.4s\n"
        "fmla vV23.4s, vU44.4s, vW32.4s\n"
        "fmla vV24.4s, vU44.4s, vW31.4s\n"
        "str qV24, [vptr1, %x[uvw_col_stride3]]\n"
        "fmla vV32.4s, vU44.4s, vW23.4s\n"
        "fmla vV33.4s, vU44.4s, vW22.4s\n"
        "fmla vV34.4s, vU44.4s, vW21.4s\n"
        "fmul vV42.4s, vU44.4s, vW13.4s\n"
        "fmla vV43.4s, vU44.4s, vW12.4s\n"
        "fmla vV44.4s, vU44.4s, vW11.4s\n"
        "ldr qU23, [uptr1, %x[uvw_col_stride2]]\n"
        "fmla vV34.4s, vU56.4s, vW33.4s\n"
        "fmla vV44.4s, vU56.4s, vW23.4s\n"
        "ldr qU33, [uptr2, %x[uvw_col_stride2]]\n"
        "fmla vV33.4s, vU55.4s, vW33.4s\n"
        "fmla vV34.4s, vU55.4s, vW32.4s\n"
        "fmla vV43.4s, vU55.4s, vW23.4s\n"
        "fmla vV44.4s, vU55.4s, vW22.4s\n"
        "ldr qU43, [uptr3, %x[uvw_col_stride2]]\n"
        "fmla vV32.4s, vU54.4s, vW33.4s\n"
        "fmla vV33.4s, vU54.4s, vW32.4s\n"
        "fmla vV34.4s, vU54.4s, vW31.4s\n"
        "str qV34, [vptr2, %x[uvw_col_stride3]]\n"
        "fmla vV42.4s, vU54.4s, vW23.4s\n"
        "fmla vV43.4s, vU54.4s, vW22.4s\n"
        "fmla vV44.4s, vU54.4s, vW21.4s\n"
        "ldr qU53, [uptr4, %x[uvw_col_stride2]]\n"
        "fmla vV44.4s, vU66.4s, vW33.4s\n"
        "ldr qU63, [uptr5, %x[uvw_col_stride2]]\n"
        "fmla vV43.4s, vU65.4s, vW33.4s\n"
        "fmla vV44.4s, vU65.4s, vW32.4s\n"
        "ldr qU12, [%x[uptr0], %x[uvw_col_stride1]]\n"
        "fmla vV42.4s, vU64.4s, vW33.4s\n"
        "fmla vV43.4s, vU64.4s, vW32.4s\n"
        "fmla vV44.4s, vU64.4s, vW31.4s\n"
        "str qV44, [vptr3, %x[uvw_col_stride3]]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "ldr qU22, [uptr1, %x[uvw_col_stride1]]\n"
        "fmla vV12.4s, vU13.4s, vW12.4s\n"
        "fmla vV13.4s, vU13.4s, vW11.4s\n"
        "ldr qU32, [uptr2, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "fmla vV12.4s, vU23.4s, vW22.4s\n"
        "fmla vV13.4s, vU23.4s, vW21.4s\n"
        "fmul vV21.4s, vU23.4s, vW13.4s\n"
        "fmla vV22.4s, vU23.4s, vW12.4s\n"
        "fmla vV23.4s, vU23.4s, vW11.4s\n"
        "ldr qU42, [uptr3, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "fmla vV12.4s, vU33.4s, vW32.4s\n"
        "fmla vV13.4s, vU33.4s, vW31.4s\n"
        "str qV13, [%x[vptr0], %x[uvw_col_stride2]]\n"
        "fmla vV21.4s, vU33.4s, vW23.4s\n"
        "fmla vV22.4s, vU33.4s, vW22.4s\n"
        "fmla vV23.4s, vU33.4s, vW21.4s\n"
        "fmul vV31.4s, vU33.4s, vW13.4s\n"
        "fmla vV32.4s, vU33.4s, vW12.4s\n"
        "fmla vV33.4s, vU33.4s, vW11.4s\n"
        "ldr qU52, [uptr4, %x[uvw_col_stride1]]\n"
        "fmla vV21.4s, vU43.4s, vW33.4s\n"
        "fmla vV22.4s, vU43.4s, vW32.4s\n"
        "fmla vV23.4s, vU43.4s, vW31.4s\n"
        "str qV23, [vptr1, %x[uvw_col_stride2]]\n"
        "fmla vV31.4s, vU43.4s, vW23.4s\n"
        "fmla vV32.4s, vU43.4s, vW22.4s\n"
        "fmla vV33.4s, vU43.4s, vW21.4s\n"
        "fmul vV41.4s, vU43.4s, vW13.4s\n"
        "fmla vV42.4s, vU43.4s, vW12.4s\n"
        "fmla vV43.4s, vU43.4s, vW11.4s\n"
        "ldr qU62, [uptr5, %x[uvw_col_stride1]]\n"
        "fmla vV31.4s, vU53.4s, vW33.4s\n"
        "fmla vV32.4s, vU53.4s, vW32.4s\n"
        "fmla vV33.4s, vU53.4s, vW31.4s\n"
        "str qV33, [vptr2, %x[uvw_col_stride2]]\n"
        "fmla vV41.4s, vU53.4s, vW23.4s\n"
        "fmla vV42.4s, vU53.4s, vW22.4s\n"
        "fmla vV43.4s, vU53.4s, vW21.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV41.4s, vU63.4s, vW33.4s\n"
        "fmla vV42.4s, vU63.4s, vW32.4s\n"
        "fmla vV43.4s, vU63.4s, vW31.4s\n"
        "str qV43, [vptr3, %x[uvw_col_stride2]]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV12.4s, vU12.4s, vW11.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "fmla vV12.4s, vU22.4s, vW21.4s\n"
        "fmla vV21.4s, vU22.4s, vW12.4s\n"
        "fmla vV22.4s, vU22.4s, vW11.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "fmla vV12.4s, vU32.4s, vW31.4s\n"
        "str qV12, [%x[vptr0], %x[uvw_col_stride1]]\n"
        "fmla vV21.4s, vU32.4s, vW22.4s\n"
        "fmla vV22.4s, vU32.4s, vW21.4s\n"
        "fmla vV31.4s, vU32.4s, vW12.4s\n"
        "fmla vV32.4s, vU32.4s, vW11.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW32.4s\n"
        "fmla vV22.4s, vU42.4s, vW31.4s\n"
        "str qV22, [vptr1, %x[uvw_col_stride1]]\n"
        "fmla vV31.4s, vU42.4s, vW22.4s\n"
        "fmla vV32.4s, vU42.4s, vW21.4s\n"
        "subs %x[c4_rem], %x[c4_rem], #1\n"
        "fmla vV41.4s, vU42.4s, vW12.4s\n"
        "fmla vV42.4s, vU42.4s, vW11.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV31.4s, vU52.4s, vW32.4s\n"
        "fmla vV32.4s, vU52.4s, vW31.4s\n"
        "str qV32, [vptr2, %x[uvw_col_stride1]]\n"
        "fmla vV41.4s, vU52.4s, vW22.4s\n"
        "fmla vV42.4s, vU52.4s, vW21.4s\n"
        "fmla vV41.4s, vU62.4s, vW32.4s\n"
        "fmla vV42.4s, vU62.4s, vW31.4s\n"
        "str qV42, [vptr3, %x[uvw_col_stride1]]\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "fmla vV21.4s, vU21.4s, vW11.4s\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU31.4s, vW21.4s\n"
        "fmla vV31.4s, vU31.4s, vW11.4s\n"
        "fmla vV21.4s, vU41.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU41.4s, vW21.4s\n"
        "fmla vV41.4s, vU41.4s, vW11.4s\n"
        "fmla vV31.4s, vU51.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"
        "fmla vV41.4s, vU51.4s, vW21.4s\n"
        "fmla vV41.4s, vU61.4s, vW31.4s\n"
        "str qV41, [vptr3], #0x10\n"

      ".unreq qW22\n" ".unreq qU64\n" ".unreq qU35\n" ".unreq qV41\n"
      ".unreq qU34\n" ".unreq qU21\n" ".unreq qV43\n" ".unreq qW21\n"
      ".unreq qU24\n" ".unreq qU54\n" ".unreq qV31\n" ".unreq qV12\n"
      ".unreq qU61\n" ".unreq qU26\n" ".unreq qV32\n"
      ".unreq qU36\n" ".unreq qU51\n" ".unreq qU66\n" ".unreq qU12\n"
      ".unreq qV14\n" ".unreq qV11\n" ".unreq qU65\n"
      ".unreq qU15\n" ".unreq qU22\n" ".unreq qU45\n"
      ".unreq qV22\n" ".unreq qU14\n"
      ".unreq qU44\n" ".unreq qU43\n" ".unreq qU11\n"
      ".unreq qV24\n" ".unreq qV42\n" ".unreq qW31\n" ".unreq qW13\n"
      ".unreq qU33\n" ".unreq qU62\n" ".unreq qU25\n" ".unreq qU56\n"
      ".unreq qW33\n"
      ".unreq qU42\n" ".unreq qU16\n" ".unreq qV44\n"
      ".unreq qU63\n" ".unreq qU31\n" ".unreq qV34\n"
      ".unreq qW11\n" ".unreq qU41\n" ".unreq qV13\n" ".unreq qV33\n"
      ".unreq qU46\n" ".unreq qU32\n" ".unreq qU13\n"
      ".unreq qW23\n" ".unreq qV23\n" ".unreq qV21\n" ".unreq qU55\n"
      ".unreq qW12\n" ".unreq qW32\n" ".unreq qU23\n" ".unreq qU52\n"
      ".unreq qU53\n" ".unreq vW22\n"
      ".unreq vU64\n" ".unreq vU35\n" ".unreq vV41\n"
      ".unreq vU34\n" ".unreq vU21\n" ".unreq vV43\n" ".unreq vW21\n"
      ".unreq vU24\n" ".unreq vU54\n" ".unreq vV31\n"
      ".unreq vV12\n" ".unreq vU61\n"
      ".unreq vU26\n" ".unreq vV32\n"
      ".unreq vU36\n" ".unreq vU51\n" ".unreq vU66\n" ".unreq vU12\n"
      ".unreq vV14\n" ".unreq vV11\n" ".unreq vU65\n"
      ".unreq vU15\n" ".unreq vU22\n" ".unreq vU45\n"
      ".unreq vV22\n" ".unreq vU14\n"
      ".unreq vU44\n" ".unreq vU43\n" ".unreq vU11\n"
      ".unreq vV24\n" ".unreq vV42\n" ".unreq vW31\n" ".unreq vW13\n"
      ".unreq vU33\n" ".unreq vU62\n" ".unreq vU25\n" ".unreq vU56\n"
      ".unreq vW33\n" ".unreq vU42\n" ".unreq vU16\n" ".unreq vV44\n"
      ".unreq vU63\n" ".unreq vU31\n" ".unreq vV34\n" ".unreq vW11\n"
      ".unreq vU41\n" ".unreq vV13\n" ".unreq vV33\n"
      ".unreq vU46\n" ".unreq vU32\n" ".unreq vU13\n" ".unreq vW23\n"
      ".unreq vV23\n" ".unreq vV21\n" ".unreq vU55\n" ".unreq vW12\n"
      ".unreq vW32\n" ".unreq vU23\n" ".unreq vU52\n" ".unreq vU53\n"
      : [uptr0] "+r" (uptr0), [vptr0] "+r" (vptr0), [wptr0] "+r" (wptr0),
        [c4_rem] "+r" (c4_rem)
      : [u_row_stride] "r" (in_row_stride * sizeof(float)),
        [v_row_stride] "r" (out_row_stride * sizeof(float)),
        [w_row_stride] "r" (weight_row_stride * sizeof(float)),
        [uvw_col_stride1] "r" (1 * in_col_stride * sizeof(float)),
        [uvw_col_stride2] "r" (2 * in_col_stride * sizeof(float)),
        [uvw_col_stride3] "r" (3 * in_col_stride * sizeof(float)),
        [uvw_col_stride4] "r" (4 * in_col_stride * sizeof(float)),
        [uvw_col_stride5] "r" (5 * in_col_stride * sizeof(float)),
        [prftch] "i" (prefetch_depth * sizeof(float)),
        [prftch_uvw_col_stride1] "r" ((prefetch_depth + 1 * in_col_stride) * sizeof(float)),
        [prftch_uvw_col_stride2] "r" ((prefetch_depth + 2 * in_col_stride) * sizeof(float)),
        [prftch_uvw_col_stride3] "r" ((prefetch_depth + 3 * in_col_stride) * sizeof(float)),
        [prftch_uvw_col_stride4] "r" ((prefetch_depth + 4 * in_col_stride) * sizeof(float)),
        [prftch_uvw_col_stride5] "r" ((prefetch_depth + 5 * in_col_stride) * sizeof(float))
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "x0",
        "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "cc", "memory"
    );
  }
  else if (channels_remaining >= 4)
  {
    int c4_rem = channels_remaining / 4;
    channels_remaining %= 4;

    asm volatile (
      "qW22 .req q0\n" "vW22 .req v0\n"
      "qU64 .req q1\n" "qU35 .req q1\n" "qV41 .req q1\n"
      "vU64 .req v1\n" "vU35 .req v1\n" "vV41 .req v1\n"
      "qU34 .req q2\n" "qU21 .req q2\n" "qV43 .req q2\n"
      "vU34 .req v2\n" "vU21 .req v2\n" "vV43 .req v2\n"
      "qW21 .req q3\n" "vW21 .req v3\n"
      "qU24 .req q4\n" "qU54 .req q4\n" "qV31 .req q4\n"
      "vU24 .req v4\n" "vU54 .req v4\n" "vV31 .req v4\n"
      "qV12 .req q5\n" "qU61 .req q5\n" "vV12 .req v5\n" "vU61 .req v5\n"
      "qU26 .req q6\n" "qV32 .req q6\n" "vU26 .req v6\n" "vV32 .req v6\n"
      "qU36 .req q7\n" "qU51 .req q7\n" "qU66 .req q7\n" "qU12 .req q7\n"
      "vU36 .req v7\n" "vU51 .req v7\n" "vU66 .req v7\n" "vU12 .req v7\n"
      "qV14 .req q8\n" "qV11 .req q8\n" "qU65 .req q8\n"
      "vV14 .req v8\n" "vV11 .req v8\n" "vU65 .req v8\n"
      "qU15 .req q9\n" "qU22 .req q9\n" "qU45 .req q9\n"
      "vU15 .req v9\n" "vU22 .req v9\n" "vU45 .req v9\n"
      "qV22 .req q10\n" "qU14 .req q10\n" "vV22 .req v10\n" "vU14 .req v10\n"
      "qU44 .req q11\n" "qU43 .req q11\n" "qU11 .req q11\n"
      "vU44 .req v11\n" "vU43 .req v11\n" "vU11 .req v11\n"
      "qV24 .req q12\n" "qV42 .req q12\n" "vV24 .req v12\n" "vV42 .req v12\n"
      "qW31 .req q13\n" "vW31 .req v13\n" "qW13 .req q14\n" "vW13 .req v14\n"
      "qU33 .req q15\n" "qU62 .req q15\n" "qU25 .req q15\n" "qU56 .req q15\n"
      "vU33 .req v15\n" "vU62 .req v15\n" "vU25 .req v15\n" "vU56 .req v15\n"
      "qW33 .req q16\n" "vW33 .req v16\n"
      "qU42 .req q17\n" "qU16 .req q17\n" "qV44 .req q17\n"
      "vU42 .req v17\n" "vU16 .req v17\n" "vV44 .req v17\n"
      "qU63 .req q18\n" "qU31 .req q18\n" "qV34 .req q18\n"
      "vU63 .req v18\n" "vU31 .req v18\n" "vV34 .req v18\n"
      "qW11 .req q19\n" "vW11 .req v19\n" "qU41 .req q20\n" "qV13 .req q20\n"
      "vU41 .req v20\n" "vV13 .req v20\n" "qV33 .req q21\n" "vV33 .req v21\n"
      "qU46 .req q22\n" "qU32 .req q22\n" "qU13 .req q22\n"
      "vU46 .req v22\n" "vU32 .req v22\n" "vU13 .req v22\n" "qW23 .req q23\n"
      "vW23 .req v23\n" "qV23 .req q24\n" "vV23 .req v24\n"
      "qV21 .req q25\n" "qU55 .req q25\n" "vV21 .req v25\n" "vU55 .req v25\n"
      "qW12 .req q26\n" "vW12 .req v26\n" "qW32 .req q27\n" "vW32 .req v27\n"
      "qU23 .req q28\n" "qU52 .req q28\n"
      "vU23 .req v28\n" "vU52 .req v28\n" "qU53 .req q29\n" "vU53 .req v29\n"

      "uptr1 .req x0\n"
      "uptr2 .req x1\n"
      "uptr3 .req x2\n"
      "uptr4 .req x3\n"
      "uptr5 .req x4\n"

      "vptr1 .req x5\n"
      "vptr2 .req x6\n"
      "vptr3 .req x7\n"

      "wptr1 .req x8\n"
      "wptr2 .req x9\n"

      "u_col_stride2 .req x10\n"
      "u_col_stride3 .req x11\n"
      "u_col_stride4 .req x12\n"
      "u_col_stride5 .req x13\n"

      "v_col_stride2 .req x14\n"
      "v_col_stride3 .req x15\n"

      "w_col_stride2 .req x16\n"

      // Prepare pointers and strides
      "add uptr1, %x[uptr0], %x[u_row_stride]\n"
      "add uptr2,    uptr1 , %x[u_row_stride]\n"
      "add uptr3,    uptr2 , %x[u_row_stride]\n"
      "add uptr4,    uptr3 , %x[u_row_stride]\n"
      "add uptr5,    uptr4 , %x[u_row_stride]\n"

      "add vptr1, %x[vptr0], %x[v_row_stride]\n"
      "add vptr2,    vptr1 , %x[v_row_stride]\n"
      "add vptr3,    vptr2 , %x[v_row_stride]\n"

      "add wptr1, %x[wptr0], %x[w_row_stride]\n"
      "add wptr2,    wptr1 , %x[w_row_stride]\n"

      "add u_col_stride2, %x[u_col_stride1], %x[u_col_stride1]\n"
      "add u_col_stride3,    u_col_stride2 , %x[u_col_stride1]\n"
      "add u_col_stride4,    u_col_stride3 , %x[u_col_stride1]\n"
      "add u_col_stride5,    u_col_stride4 , %x[u_col_stride1]\n"

      "add v_col_stride2, %x[v_col_stride1], %x[v_col_stride1]\n"
      "add v_col_stride3,    v_col_stride2 , %x[v_col_stride1]\n"

      "add w_col_stride2, %x[w_col_stride1], %x[w_col_stride1]\n"

      // Load initial operands
      "ldr qU16, [%x[uptr0], u_col_stride5]\n"
      "ldr qW13, [%x[wptr0], w_col_stride2]\n"
      "subs %x[c4_rem], %x[c4_rem], #1\n"
      "ldr qU15, [%x[uptr0], u_col_stride4]\n"
      "ldr qW23, [wptr1, w_col_stride2]\n"
      "ldr qU14, [%x[uptr0], u_col_stride3]\n"
      "ldr qW33, [wptr2, w_col_stride2]\n"
      "ldr qU26, [uptr1, u_col_stride5]\n"
      "ldr qW12, [%x[wptr0], %x[w_col_stride1]]\n"
      "ldr qU25, [uptr1, u_col_stride4]\n"
      "ldr qW22, [wptr1, %x[w_col_stride1]]\n"
      "ldr qU36, [uptr2, u_col_stride5]\n"
      "ldr qW32, [wptr2, %x[w_col_stride1]]\n"
      "ldr qW11, [%x[wptr0]], #0x10\n"
      "fmul vV14.4s, vU16.4s, vW13.4s\n"
      "ldr qU24, [uptr1, u_col_stride3]\n"
      "fmul vV13.4s, vU15.4s, vW13.4s\n"
      "ldr qW31, [wptr2], #0x10\n"
      "fmla vV14.4s, vU15.4s, vW12.4s\n"
      "ldr qW21, [wptr1], #0x10\n"
      "fmul vV12.4s, vU14.4s, vW13.4s\n"
      "ldr qU34, [uptr2, u_col_stride3]\n"
      "fmla vV13.4s, vU14.4s, vW12.4s\n"
      "ldr qU46, [uptr3, u_col_stride5]\n"
      "fmla vV14.4s, vU14.4s, vW11.4s\n"
      "ldr qU45, [uptr3, u_col_stride4]\n"
      "fmla vV14.4s, vU26.4s, vW23.4s\n"
      "ldr qU35, [uptr2, u_col_stride4]\n"
      "fmul vV24.4s, vU26.4s, vW13.4s\n"
      "ldr qU44, [uptr3, u_col_stride3]\n"
      "fmla vV13.4s, vU25.4s, vW23.4s\n"
      "beq 2f\n"  // Single iteration only

      "1:"  // Loop body
        "fmla vV14.4s, vU25.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[wptr0]]\n"
        "fmul vV23.4s, vU25.4s, vW13.4s\n"
        "prfm pldl1keep, [%x[wptr0], %x[w_col_stride1]]\n"
        "fmla vV24.4s, vU25.4s, vW12.4s\n"
        "ldr qU56, [uptr4, u_col_stride5]\n"
        "fmla vV12.4s, vU24.4s, vW23.4s\n"
        "prfm pldl1keep, [%x[wptr0],    w_col_stride2 ]\n"
        "fmla vV13.4s, vU24.4s, vW22.4s\n"
        "prfm pldl1keep, [   wptr1 ]\n"
        "fmla vV14.4s, vU24.4s, vW21.4s\n"
        "prfm pldl1keep, [   wptr1 , %x[w_col_stride1]]\n"
        "fmul vV22.4s, vU24.4s, vW13.4s\n"
        "prfm pldl1keep, [   wptr1 ,    w_col_stride2 ]\n"
        "fmla vV23.4s, vU24.4s, vW12.4s\n"
        "prfm pldl1keep, [   wptr2 ]\n"
        "fmla vV24.4s, vU24.4s, vW11.4s\n"
        "ldr qU55, [uptr4, u_col_stride4]\n"
        "fmla vV14.4s, vU36.4s, vW33.4s\n"
        "prfm pldl1keep, [   wptr2 , %x[w_col_stride1]]\n"
        "fmla vV24.4s, vU36.4s, vW23.4s\n"
        "prfm pldl1keep, [   wptr2 ,    w_col_stride2 ]\n"
        "fmul vV34.4s, vU36.4s, vW13.4s\n"
        "ldr qU54, [uptr4, u_col_stride3]\n"
        "fmla vV13.4s, vU35.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr2 , %x[u_col_stride1]]\n"
        "fmla vV14.4s, vU35.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr2 ,    u_col_stride2 ]\n"
        "fmla vV23.4s, vU35.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr2 ,    u_col_stride3 ]\n"
        "fmla vV24.4s, vU35.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr2 ,    u_col_stride4 ]\n"
        "fmul vV33.4s, vU35.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr2 ,    u_col_stride5 ]\n"
        "fmla vV34.4s, vU35.4s, vW12.4s\n"
        "ldr qU66, [uptr5, u_col_stride5]\n"
        "fmla vV12.4s, vU34.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr3 ]\n"
        "fmla vV13.4s, vU34.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr3 , %x[u_col_stride1]]\n"
        "fmla vV14.4s, vU34.4s, vW31.4s\n"
        "str qV14, [%x[vptr0], v_col_stride3]\n"
        "fmla vV22.4s, vU34.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr3 ,    u_col_stride2 ]\n"
        "fmla vV23.4s, vU34.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr3 ,    u_col_stride3 ]\n"
        "fmla vV24.4s, vU34.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr3 ,    u_col_stride4 ]\n"
        "fmul vV32.4s, vU34.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr3 ,    u_col_stride5 ]\n"
        "fmla vV33.4s, vU34.4s, vW12.4s\n"
        "prfm pldl1keep, [   uptr4 ]\n"
        "fmla vV34.4s, vU34.4s, vW11.4s\n"
        "ldr qU65, [uptr5, u_col_stride4]\n"
        "fmla vV24.4s, vU46.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr4 , %x[u_col_stride1]]\n"
        "fmla vV34.4s, vU46.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr4 ,    u_col_stride2 ]\n"
        "fmul vV44.4s, vU46.4s, vW13.4s\n"
        "ldr qU64, [uptr5, u_col_stride3]\n"
        "fmla vV23.4s, vU45.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr4 ,    u_col_stride3 ]\n"
        "fmla vV24.4s, vU45.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr4 ,    u_col_stride4 ]\n"
        "fmla vV33.4s, vU45.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr4 ,    u_col_stride5 ]\n"
        "fmla vV34.4s, vU45.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr5 ]\n"
        "fmul vV43.4s, vU45.4s, vW13.4s\n"
        "prfm pldl1keep, [   uptr5 , %x[u_col_stride1]]\n"
        "fmla vV44.4s, vU45.4s, vW12.4s\n"
        "ldr qU13, [%x[uptr0], u_col_stride2]\n"
        "fmla vV22.4s, vU44.4s, vW33.4s\n"
        "prfm pldl1keep, [   uptr5 ,    u_col_stride2 ]\n"
        "fmla vV23.4s, vU44.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr5 ,    u_col_stride3 ]\n"
        "fmla vV24.4s, vU44.4s, vW31.4s\n"
        "str qV24, [vptr1, v_col_stride3]\n"
        "fmla vV32.4s, vU44.4s, vW23.4s\n"
        "prfm pldl1keep, [   uptr5 ,    u_col_stride4 ]\n"
        "fmla vV33.4s, vU44.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr5 ,    u_col_stride5 ]\n"
        "fmla vV34.4s, vU44.4s, vW21.4s\n"
        "prfm pstl1keep, [%x[vptr0]]\n"
        "fmul vV42.4s, vU44.4s, vW13.4s\n"
        "prfm pstl1keep, [%x[vptr0], %x[v_col_stride1]]\n"
        "fmla vV43.4s, vU44.4s, vW12.4s\n"
        "prfm pstl1keep, [%x[vptr0],    v_col_stride2 ]\n"
        "fmla vV44.4s, vU44.4s, vW11.4s\n"
        "ldr qU23, [uptr1, u_col_stride2]\n"
        "fmla vV34.4s, vU56.4s, vW33.4s\n"
        "prfm pstl1keep, [%x[vptr0],    v_col_stride3 ]\n"
        "fmla vV44.4s, vU56.4s, vW23.4s\n"
        "ldr qU33, [uptr2, u_col_stride2]\n"
        "fmla vV33.4s, vU55.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr1 ]\n"
        "fmla vV34.4s, vU55.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr1 , %x[v_col_stride1]]\n"
        "fmla vV43.4s, vU55.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr1 ,    v_col_stride2 ]\n"
        "fmla vV44.4s, vU55.4s, vW22.4s\n"
        "ldr qU43, [uptr3, u_col_stride2]\n"
        "fmla vV32.4s, vU54.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr1 ,    v_col_stride3 ]\n"
        "fmla vV33.4s, vU54.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr2 ]\n"
        "fmla vV34.4s, vU54.4s, vW31.4s\n"
        "str qV34, [vptr2, v_col_stride3]\n"
        "fmla vV42.4s, vU54.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr2 , %x[v_col_stride1]]\n"
        "fmla vV43.4s, vU54.4s, vW22.4s\n"
        "prfm pstl1keep, [   vptr2 ,    v_col_stride2 ]\n"
        "fmla vV44.4s, vU54.4s, vW21.4s\n"
        "ldr qU53, [uptr4, u_col_stride2]\n"
        "fmla vV44.4s, vU66.4s, vW33.4s\n"
        "ldr qU63, [uptr5, u_col_stride2]\n"
        "fmla vV43.4s, vU65.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr2 ,    v_col_stride3 ]\n"
        "fmla vV44.4s, vU65.4s, vW32.4s\n"
        "ldr qU12, [%x[uptr0], %x[u_col_stride1]]\n"
        "fmla vV42.4s, vU64.4s, vW33.4s\n"
        "prfm pstl1keep, [   vptr3 ]\n"
        "fmla vV43.4s, vU64.4s, vW32.4s\n"
        "prfm pstl1keep, [   vptr3 , %x[v_col_stride1]]\n"
        "fmla vV44.4s, vU64.4s, vW31.4s\n"
        "str qV44, [vptr3, v_col_stride3]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "ldr qU22, [uptr1, %x[u_col_stride1]]\n"
        "fmla vV12.4s, vU13.4s, vW12.4s\n"
        "prfm pstl1keep, [   vptr3 ,    v_col_stride2 ]\n"
        "fmla vV13.4s, vU13.4s, vW11.4s\n"
        "ldr qU32, [uptr2, %x[u_col_stride1]]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "prfm pstl1keep, [   vptr3 ,    v_col_stride3 ]\n"
        "fmla vV12.4s, vU23.4s, vW22.4s\n"
        "fmla vV13.4s, vU23.4s, vW21.4s\n"
        "fmul vV21.4s, vU23.4s, vW13.4s\n"
        "fmla vV22.4s, vU23.4s, vW12.4s\n"
        "fmla vV23.4s, vU23.4s, vW11.4s\n"
        "ldr qU42, [uptr3, %x[u_col_stride1]]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "fmla vV12.4s, vU33.4s, vW32.4s\n"
        "fmla vV13.4s, vU33.4s, vW31.4s\n"
        "str qV13, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21.4s, vU33.4s, vW23.4s\n"
        "fmla vV22.4s, vU33.4s, vW22.4s\n"
        "fmla vV23.4s, vU33.4s, vW21.4s\n"
        "fmul vV31.4s, vU33.4s, vW13.4s\n"
        "fmla vV32.4s, vU33.4s, vW12.4s\n"
        "fmla vV33.4s, vU33.4s, vW11.4s\n"
        "ldr qU52, [uptr4, %x[u_col_stride1]]\n"
        "fmla vV21.4s, vU43.4s, vW33.4s\n"
        "fmla vV22.4s, vU43.4s, vW32.4s\n"
        "fmla vV23.4s, vU43.4s, vW31.4s\n"
        "str qV23, [vptr1, v_col_stride2]\n"
        "fmla vV31.4s, vU43.4s, vW23.4s\n"
        "fmla vV32.4s, vU43.4s, vW22.4s\n"
        "fmla vV33.4s, vU43.4s, vW21.4s\n"
        "fmul vV41.4s, vU43.4s, vW13.4s\n"
        "ldr qW13, [%x[wptr0], w_col_stride2]\n"
        "fmla vV42.4s, vU43.4s, vW12.4s\n"
        "fmla vV43.4s, vU43.4s, vW11.4s\n"
        "ldr qU62, [uptr5, %x[u_col_stride1]]\n"
        "fmla vV31.4s, vU53.4s, vW33.4s\n"
        "fmla vV32.4s, vU53.4s, vW32.4s\n"
        "fmla vV33.4s, vU53.4s, vW31.4s\n"
        "str qV33, [vptr2, v_col_stride2]\n"
        "fmla vV41.4s, vU53.4s, vW23.4s\n"
        "ldr qW23, [wptr1, w_col_stride2]\n"
        "fmla vV42.4s, vU53.4s, vW22.4s\n"
        "fmla vV43.4s, vU53.4s, vW21.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV41.4s, vU63.4s, vW33.4s\n"
        "ldr qW33, [wptr2, w_col_stride2]\n"
        "fmla vV42.4s, vU63.4s, vW32.4s\n"
        "prfm pldl1keep, [%x[uptr0]]\n"
        "fmla vV43.4s, vU63.4s, vW31.4s\n"
        "str qV43, [vptr3, v_col_stride2]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV12.4s, vU12.4s, vW11.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[uptr0], %x[u_col_stride1]]\n"
        "fmla vV12.4s, vU22.4s, vW21.4s\n"
        "prfm pldl1keep, [%x[uptr0],    u_col_stride2 ]\n"
        "fmla vV21.4s, vU22.4s, vW12.4s\n"
        "prfm pldl1keep, [%x[uptr0],    u_col_stride3 ]\n"
        "fmla vV22.4s, vU22.4s, vW11.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "prfm pldl1keep, [%x[uptr0],    u_col_stride4 ]\n"
        "fmla vV12.4s, vU32.4s, vW31.4s\n"
        "str qV12, [%x[vptr0], %x[v_col_stride1]]\n"
        "fmla vV21.4s, vU32.4s, vW22.4s\n"
        "prfm pldl1keep, [%x[uptr0],    u_col_stride5 ]\n"
        "fmla vV22.4s, vU32.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr1 ]\n"
        "fmla vV31.4s, vU32.4s, vW12.4s\n"
        "prfm pldl1keep, [   uptr1 , %x[u_col_stride1]]\n"
        "fmla vV32.4s, vU32.4s, vW11.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr1 ,    u_col_stride2 ]\n"
        "fmla vV22.4s, vU42.4s, vW31.4s\n"
        "str qV22, [vptr1, %x[v_col_stride1]]\n"
        "fmla vV31.4s, vU42.4s, vW22.4s\n"
        "prfm pldl1keep, [   uptr1 ,    u_col_stride3 ]\n"
        "fmla vV32.4s, vU42.4s, vW21.4s\n"
        "subs %x[c4_rem], %x[c4_rem], #1\n"
        "fmla vV41.4s, vU42.4s, vW12.4s\n"
        "ldr qW12, [%x[wptr0], %x[w_col_stride1]]\n"
        "fmla vV42.4s, vU42.4s, vW11.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV31.4s, vU52.4s, vW32.4s\n"
        "prfm pldl1keep, [   uptr1 ,    u_col_stride4 ]\n"
        "fmla vV32.4s, vU52.4s, vW31.4s\n"
        "str qV32, [vptr2, %x[v_col_stride1]]\n"
        "fmla vV41.4s, vU52.4s, vW22.4s\n"
        "ldr qW22, [wptr1, %x[w_col_stride1]]\n"
        "fmla vV42.4s, vU52.4s, vW21.4s\n"
        "ldr qU16, [%x[uptr0], u_col_stride5]\n"
        "fmla vV41.4s, vU62.4s, vW32.4s\n"
        "ldr qW32, [wptr2, %x[w_col_stride1]]\n"
        "fmla vV42.4s, vU62.4s, vW31.4s\n"
        "str qV42, [vptr3, %x[v_col_stride1]]\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "ldr qU15, [%x[uptr0], u_col_stride4]\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "ldr qU14, [%x[uptr0], u_col_stride3]\n"
        "fmla vV21.4s, vU21.4s, vW11.4s\n"
        "ldr qU26, [uptr1, u_col_stride5]\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU31.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr1 ,    u_col_stride5 ]\n"
        "fmla vV31.4s, vU31.4s, vW11.4s\n"
        "ldr qU25, [uptr1, u_col_stride4]\n"
        "fmla vV21.4s, vU41.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU41.4s, vW21.4s\n"
        "prfm pldl1keep, [   uptr2 ]\n"
        "fmla vV41.4s, vU41.4s, vW11.4s\n"
        "ldr qW11, [%x[wptr0]], #0x10\n"
        "fmla vV31.4s, vU51.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"
        "fmla vV41.4s, vU51.4s, vW21.4s\n"
        "ldr qU36, [uptr2, u_col_stride5]\n"
        "fmla vV41.4s, vU61.4s, vW31.4s\n"
        "str qV41, [vptr3], #0x10\n"
        "fmul vV14.4s, vU16.4s, vW13.4s\n"
        "ldr qU24, [uptr1, u_col_stride3]\n"
        "fmul vV13.4s, vU15.4s, vW13.4s\n"
        "ldr qW31, [wptr2], #0x10\n"
        "fmla vV14.4s, vU15.4s, vW12.4s\n"
        "ldr qW21, [wptr1], #0x10\n"
        "fmul vV12.4s, vU14.4s, vW13.4s\n"
        "ldr qU34, [uptr2, u_col_stride3]\n"
        "fmla vV13.4s, vU14.4s, vW12.4s\n"
        "ldr qU46, [uptr3, u_col_stride5]\n"
        "fmla vV14.4s, vU14.4s, vW11.4s\n"
        "ldr qU45, [uptr3, u_col_stride4]\n"
        "fmla vV14.4s, vU26.4s, vW23.4s\n"
        "ldr qU35, [uptr2, u_col_stride4]\n"
        "fmul vV24.4s, vU26.4s, vW13.4s\n"
        "ldr qU44, [uptr3, u_col_stride3]\n"
        "fmla vV13.4s, vU25.4s, vW23.4s\n"
        "bne 1b\n"

      "2:"  // Final iteration
        "fmla vV14.4s, vU25.4s, vW22.4s\n"
        "fmul vV23.4s, vU25.4s, vW13.4s\n"
        "fmla vV24.4s, vU25.4s, vW12.4s\n"
        "ldr qU56, [uptr4, u_col_stride5]\n"
        "fmla vV12.4s, vU24.4s, vW23.4s\n"
        "fmla vV13.4s, vU24.4s, vW22.4s\n"
        "fmla vV14.4s, vU24.4s, vW21.4s\n"
        "fmul vV22.4s, vU24.4s, vW13.4s\n"
        "fmla vV23.4s, vU24.4s, vW12.4s\n"
        "fmla vV24.4s, vU24.4s, vW11.4s\n"
        "ldr qU55, [uptr4, u_col_stride4]\n"
        "fmla vV14.4s, vU36.4s, vW33.4s\n"
        "fmla vV24.4s, vU36.4s, vW23.4s\n"
        "fmul vV34.4s, vU36.4s, vW13.4s\n"
        "ldr qU54, [uptr4, u_col_stride3]\n"
        "fmla vV13.4s, vU35.4s, vW33.4s\n"
        "fmla vV14.4s, vU35.4s, vW32.4s\n"
        "fmla vV23.4s, vU35.4s, vW23.4s\n"
        "fmla vV24.4s, vU35.4s, vW22.4s\n"
        "fmul vV33.4s, vU35.4s, vW13.4s\n"
        "fmla vV34.4s, vU35.4s, vW12.4s\n"
        "ldr qU66, [uptr5, u_col_stride5]\n"
        "fmla vV12.4s, vU34.4s, vW33.4s\n"
        "fmla vV13.4s, vU34.4s, vW32.4s\n"
        "fmla vV14.4s, vU34.4s, vW31.4s\n"
        "str qV14, [%x[vptr0], v_col_stride3]\n"
        "fmla vV22.4s, vU34.4s, vW23.4s\n"
        "fmla vV23.4s, vU34.4s, vW22.4s\n"
        "fmla vV24.4s, vU34.4s, vW21.4s\n"
        "fmul vV32.4s, vU34.4s, vW13.4s\n"
        "fmla vV33.4s, vU34.4s, vW12.4s\n"
        "fmla vV34.4s, vU34.4s, vW11.4s\n"
        "ldr qU65, [uptr5, u_col_stride4]\n"
        "fmla vV24.4s, vU46.4s, vW33.4s\n"
        "fmla vV34.4s, vU46.4s, vW23.4s\n"
        "fmul vV44.4s, vU46.4s, vW13.4s\n"
        "ldr qU64, [uptr5, u_col_stride3]\n"
        "fmla vV23.4s, vU45.4s, vW33.4s\n"
        "fmla vV24.4s, vU45.4s, vW32.4s\n"
        "fmla vV33.4s, vU45.4s, vW23.4s\n"
        "fmla vV34.4s, vU45.4s, vW22.4s\n"
        "fmul vV43.4s, vU45.4s, vW13.4s\n"
        "fmla vV44.4s, vU45.4s, vW12.4s\n"
        "ldr qU13, [%x[uptr0], u_col_stride2]\n"
        "fmla vV22.4s, vU44.4s, vW33.4s\n"
        "fmla vV23.4s, vU44.4s, vW32.4s\n"
        "fmla vV24.4s, vU44.4s, vW31.4s\n"
        "str qV24, [vptr1, v_col_stride3]\n"
        "fmla vV32.4s, vU44.4s, vW23.4s\n"
        "fmla vV33.4s, vU44.4s, vW22.4s\n"
        "fmla vV34.4s, vU44.4s, vW21.4s\n"
        "fmul vV42.4s, vU44.4s, vW13.4s\n"
        "fmla vV43.4s, vU44.4s, vW12.4s\n"
        "fmla vV44.4s, vU44.4s, vW11.4s\n"
        "ldr qU23, [uptr1, u_col_stride2]\n"
        "fmla vV34.4s, vU56.4s, vW33.4s\n"
        "fmla vV44.4s, vU56.4s, vW23.4s\n"
        "ldr qU33, [uptr2, u_col_stride2]\n"
        "fmla vV33.4s, vU55.4s, vW33.4s\n"
        "fmla vV34.4s, vU55.4s, vW32.4s\n"
        "fmla vV43.4s, vU55.4s, vW23.4s\n"
        "fmla vV44.4s, vU55.4s, vW22.4s\n"
        "ldr qU43, [uptr3, u_col_stride2]\n"
        "fmla vV32.4s, vU54.4s, vW33.4s\n"
        "fmla vV33.4s, vU54.4s, vW32.4s\n"
        "fmla vV34.4s, vU54.4s, vW31.4s\n"
        "str qV34, [vptr2, v_col_stride3]\n"
        "fmla vV42.4s, vU54.4s, vW23.4s\n"
        "fmla vV43.4s, vU54.4s, vW22.4s\n"
        "fmla vV44.4s, vU54.4s, vW21.4s\n"
        "ldr qU53, [uptr4, u_col_stride2]\n"
        "fmla vV44.4s, vU66.4s, vW33.4s\n"
        "ldr qU63, [uptr5, u_col_stride2]\n"
        "fmla vV43.4s, vU65.4s, vW33.4s\n"
        "fmla vV44.4s, vU65.4s, vW32.4s\n"
        "ldr qU12, [%x[uptr0], %x[u_col_stride1]]\n"
        "fmla vV42.4s, vU64.4s, vW33.4s\n"
        "fmla vV43.4s, vU64.4s, vW32.4s\n"
        "fmla vV44.4s, vU64.4s, vW31.4s\n"
        "str qV44, [vptr3, v_col_stride3]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "ldr qU22, [uptr1, %x[u_col_stride1]]\n"
        "fmla vV12.4s, vU13.4s, vW12.4s\n"
        "fmla vV13.4s, vU13.4s, vW11.4s\n"
        "ldr qU32, [uptr2, %x[u_col_stride1]]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "fmla vV12.4s, vU23.4s, vW22.4s\n"
        "fmla vV13.4s, vU23.4s, vW21.4s\n"
        "fmul vV21.4s, vU23.4s, vW13.4s\n"
        "fmla vV22.4s, vU23.4s, vW12.4s\n"
        "fmla vV23.4s, vU23.4s, vW11.4s\n"
        "ldr qU42, [uptr3, %x[u_col_stride1]]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "fmla vV12.4s, vU33.4s, vW32.4s\n"
        "fmla vV13.4s, vU33.4s, vW31.4s\n"
        "str qV13, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21.4s, vU33.4s, vW23.4s\n"
        "fmla vV22.4s, vU33.4s, vW22.4s\n"
        "fmla vV23.4s, vU33.4s, vW21.4s\n"
        "fmul vV31.4s, vU33.4s, vW13.4s\n"
        "fmla vV32.4s, vU33.4s, vW12.4s\n"
        "fmla vV33.4s, vU33.4s, vW11.4s\n"
        "ldr qU52, [uptr4, %x[u_col_stride1]]\n"
        "fmla vV21.4s, vU43.4s, vW33.4s\n"
        "fmla vV22.4s, vU43.4s, vW32.4s\n"
        "fmla vV23.4s, vU43.4s, vW31.4s\n"
        "str qV23, [vptr1, v_col_stride2]\n"
        "fmla vV31.4s, vU43.4s, vW23.4s\n"
        "fmla vV32.4s, vU43.4s, vW22.4s\n"
        "fmla vV33.4s, vU43.4s, vW21.4s\n"
        "fmul vV41.4s, vU43.4s, vW13.4s\n"
        "fmla vV42.4s, vU43.4s, vW12.4s\n"
        "fmla vV43.4s, vU43.4s, vW11.4s\n"
        "ldr qU62, [uptr5, %x[u_col_stride1]]\n"
        "fmla vV31.4s, vU53.4s, vW33.4s\n"
        "fmla vV32.4s, vU53.4s, vW32.4s\n"
        "fmla vV33.4s, vU53.4s, vW31.4s\n"
        "str qV33, [vptr2, v_col_stride2]\n"
        "fmla vV41.4s, vU53.4s, vW23.4s\n"
        "fmla vV42.4s, vU53.4s, vW22.4s\n"
        "fmla vV43.4s, vU53.4s, vW21.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV41.4s, vU63.4s, vW33.4s\n"
        "fmla vV42.4s, vU63.4s, vW32.4s\n"
        "fmla vV43.4s, vU63.4s, vW31.4s\n"
        "str qV43, [vptr3, v_col_stride2]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV12.4s, vU12.4s, vW11.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "fmla vV12.4s, vU22.4s, vW21.4s\n"
        "fmla vV21.4s, vU22.4s, vW12.4s\n"
        "fmla vV22.4s, vU22.4s, vW11.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "fmla vV12.4s, vU32.4s, vW31.4s\n"
        "str qV12, [%x[vptr0], %x[v_col_stride1]]\n"
        "fmla vV21.4s, vU32.4s, vW22.4s\n"
        "fmla vV22.4s, vU32.4s, vW21.4s\n"
        "fmla vV31.4s, vU32.4s, vW12.4s\n"
        "fmla vV32.4s, vU32.4s, vW11.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW32.4s\n"
        "fmla vV22.4s, vU42.4s, vW31.4s\n"
        "str qV22, [vptr1, %x[v_col_stride1]]\n"
        "fmla vV31.4s, vU42.4s, vW22.4s\n"
        "fmla vV32.4s, vU42.4s, vW21.4s\n"
        "subs %x[c4_rem], %x[c4_rem], #1\n"
        "fmla vV41.4s, vU42.4s, vW12.4s\n"
        "fmla vV42.4s, vU42.4s, vW11.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV31.4s, vU52.4s, vW32.4s\n"
        "fmla vV32.4s, vU52.4s, vW31.4s\n"
        "str qV32, [vptr2, %x[v_col_stride1]]\n"
        "fmla vV41.4s, vU52.4s, vW22.4s\n"
        "fmla vV42.4s, vU52.4s, vW21.4s\n"
        "fmla vV41.4s, vU62.4s, vW32.4s\n"
        "fmla vV42.4s, vU62.4s, vW31.4s\n"
        "str qV42, [vptr3, %x[v_col_stride1]]\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "fmla vV21.4s, vU21.4s, vW11.4s\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU31.4s, vW21.4s\n"
        "fmla vV31.4s, vU31.4s, vW11.4s\n"
        "fmla vV21.4s, vU41.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU41.4s, vW21.4s\n"
        "fmla vV41.4s, vU41.4s, vW11.4s\n"
        "fmla vV31.4s, vU51.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"
        "fmla vV41.4s, vU51.4s, vW21.4s\n"
        "fmla vV41.4s, vU61.4s, vW31.4s\n"
        "str qV41, [vptr3], #0x10\n"

      ".unreq qW22\n" ".unreq qU64\n" ".unreq qU35\n" ".unreq qV41\n"
      ".unreq qU34\n" ".unreq qU21\n" ".unreq qV43\n" ".unreq qW21\n"
      ".unreq qU24\n" ".unreq qU54\n" ".unreq qV31\n" ".unreq qV12\n"
      ".unreq qU61\n" ".unreq qU26\n" ".unreq qV32\n"
      ".unreq qU36\n" ".unreq qU51\n" ".unreq qU66\n" ".unreq qU12\n"
      ".unreq qV14\n" ".unreq qV11\n" ".unreq qU65\n"
      ".unreq qU15\n" ".unreq qU22\n" ".unreq qU45\n"
      ".unreq qV22\n" ".unreq qU14\n"
      ".unreq qU44\n" ".unreq qU43\n" ".unreq qU11\n"
      ".unreq qV24\n" ".unreq qV42\n" ".unreq qW31\n" ".unreq qW13\n"
      ".unreq qU33\n" ".unreq qU62\n" ".unreq qU25\n" ".unreq qU56\n"
      ".unreq qW33\n"
      ".unreq qU42\n" ".unreq qU16\n" ".unreq qV44\n"
      ".unreq qU63\n" ".unreq qU31\n" ".unreq qV34\n"
      ".unreq qW11\n" ".unreq qU41\n" ".unreq qV13\n" ".unreq qV33\n"
      ".unreq qU46\n" ".unreq qU32\n" ".unreq qU13\n"
      ".unreq qW23\n" ".unreq qV23\n" ".unreq qV21\n" ".unreq qU55\n"
      ".unreq qW12\n" ".unreq qW32\n" ".unreq qU23\n" ".unreq qU52\n"
      ".unreq qU53\n" ".unreq vW22\n"
      ".unreq vU64\n" ".unreq vU35\n" ".unreq vV41\n"
      ".unreq vU34\n" ".unreq vU21\n" ".unreq vV43\n" ".unreq vW21\n"
      ".unreq vU24\n" ".unreq vU54\n" ".unreq vV31\n"
      ".unreq vV12\n" ".unreq vU61\n"
      ".unreq vU26\n" ".unreq vV32\n"
      ".unreq vU36\n" ".unreq vU51\n" ".unreq vU66\n" ".unreq vU12\n"
      ".unreq vV14\n" ".unreq vV11\n" ".unreq vU65\n"
      ".unreq vU15\n" ".unreq vU22\n" ".unreq vU45\n"
      ".unreq vV22\n" ".unreq vU14\n"
      ".unreq vU44\n" ".unreq vU43\n" ".unreq vU11\n"
      ".unreq vV24\n" ".unreq vV42\n" ".unreq vW31\n" ".unreq vW13\n"
      ".unreq vU33\n" ".unreq vU62\n" ".unreq vU25\n" ".unreq vU56\n"
      ".unreq vW33\n" ".unreq vU42\n" ".unreq vU16\n" ".unreq vV44\n"
      ".unreq vU63\n" ".unreq vU31\n" ".unreq vV34\n" ".unreq vW11\n"
      ".unreq vU41\n" ".unreq vV13\n" ".unreq vV33\n"
      ".unreq vU46\n" ".unreq vU32\n" ".unreq vU13\n" ".unreq vW23\n"
      ".unreq vV23\n" ".unreq vV21\n" ".unreq vU55\n" ".unreq vW12\n"
      ".unreq vW32\n" ".unreq vU23\n" ".unreq vU52\n" ".unreq vU53\n"
      : [uptr0] "+r" (uptr0), [vptr0] "+r" (vptr0), [wptr0] "+r" (wptr0),
        [c4_rem] "+r" (c4_rem)
      : [u_row_stride] "r" (in_row_stride * sizeof(float)),
        [u_col_stride1] "r" (in_col_stride * sizeof(float)),
        [v_row_stride] "r" (out_row_stride * sizeof(float)),
        [v_col_stride1] "r" (out_col_stride * sizeof(float)),
        [w_row_stride] "r" (weight_row_stride * sizeof(float)),
        [w_col_stride1] "r" (weight_col_stride * sizeof(float))
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "x0",
        "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
        "x12", "x13", "x14", "x15", "x16", "cc", "memory"
    );
  }
  for (; channels_remaining; channels_remaining--)
  {
    // Load input tile
    float u[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      const float* const inptr_row = uptr0 + (i - in_pad_top)*in_row_stride;
      for (int j = 0; j < inner_tile_cols; j++)
      {
        if (i < in_pad_top || in_cells_i <= i ||
            j < in_pad_left || in_cells_j <= j)
        {
          u[i][j] = static_cast<float>(0);
        }
        else
        {
          u[i][j] = *(inptr_row + (j - in_pad_left)*in_col_stride);
        }
      }
    }
    uptr0++;

    // Load weights tile
    float w[kernel_rows][kernel_cols];
    for (int i = 0; i < kernel_rows; i++)
    {
      const float* const wptr_row = wptr0 + i*weight_row_stride;
      for (int j = 0; j < kernel_cols; j++)
      {
        w[i][j] = *(wptr_row + j*weight_col_stride);
      }
    }
    wptr0++;

    // Perform the convolution
    float v[output_tile_rows][output_tile_cols];
    for (int out_i = 0; out_i < out_cells_i; out_i++)
    {
      for (int out_j = 0; out_j < out_cells_j; out_j++)
      {
        // Clear the accumulator
        v[out_i][out_j] = static_cast<float>(0);

        // Base co-ordinate
        const int base_i = out_i * stride_rows;
        const int base_j = out_j * stride_cols;

        // Fill the accumulator
        for (int in_i = 0; in_i < kernel_rows; in_i++)
        {
          const int i = base_i + in_i;
          for (int in_j = 0; in_j < kernel_cols; in_j++)
          {
            const int j = base_j + in_j;
            v[out_i][out_j] += w[in_i][in_j] * u[i][j];
          }
        }
      }
    }

    // Store the output tile
    for (int i = 0; i < out_cells_i; i++)
    {
      float* const outptr_row = vptr0 + i*out_row_stride;
      for (int j = 0; j < out_cells_j; j++)
      {
        *(outptr_row + j*out_col_stride) = v[i][j];
      }
    }
    vptr0++;
  }
}

#endif  // __aarch64__

template <>
const Conv::TileFn Conv::tilefn_unpadded = ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>;

template <>
const Conv::TileFn Conv::tilefn_top[n_in_pad_top_fns] = {
  ConvImpl::template process_tile<true, 1, 0, 0, 0, 0, 0>,
};

template <>
const Conv::TileFn Conv::tilefn_left[n_in_pad_left_fns] = {
  ConvImpl::template process_tile<true, 0, 1, 0, 0, 0, 0>,
};

template <>
const Conv::TileFn Conv::tilefn_bottom[n_in_pad_bottom_fns][n_out_pad_bottom_fns] = {
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 3, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 3, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 3, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 3, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 3, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 2, 0>,
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 3, 0>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_right[n_in_pad_right_fns][n_out_pad_right_fns] = {
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 3>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 3>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 3>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 3>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 3>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 2>,
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 3>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_generic = ConvImpl::template process_tile<false>;

template class DepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float>;
}  // namespace depthwise
