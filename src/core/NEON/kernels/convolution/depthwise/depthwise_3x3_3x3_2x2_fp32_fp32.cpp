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
#include "impl_fp32_fp32.hpp"

namespace depthwise
{
using Conv = DepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float>;
using ConvImpl = DepthwiseConvolutionImpl<3, 3, 3, 3, 2, 2, float, float>;

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
  // Copy pointers
  const float *uptr0 = inptr;
  const float *wptr0 = weights;
  float *vptr0 = outptr;

  int channels_remaining = n_channels;
  if (channels_remaining >= 4)
  {
    // Process blocks of 4 channels at a time
    int n_iters = channels_remaining / 4 - 1;
    channels_remaining %= 4;

    asm volatile(
        // Prepare aliases
        "qW13 .req q0\n" "vW13 .req v0\n"
        "qU15 .req q1\n" "qU73 .req q1\n" "qU45 .req q1\n" "qU14 .req q1\n"
        "vU15 .req v1\n" "vU73 .req v1\n" "vU45 .req v1\n" "vU14 .req v1\n"
        "qU62 .req q2\n" "qV12 .req q2\n" "vU62 .req v2\n" "vV12 .req v2\n"
        "qU51 .req q3\n" "qU43 .req q3\n" "qU55 .req q3\n"
        "vU51 .req v3\n" "vU43 .req v3\n" "vU55 .req v3\n"
        "qU77 .req q4\n" "qV13 .req q4\n" "qV31 .req q4\n" "qU44 .req q4\n"
        "vU77 .req v4\n" "vV13 .req v4\n" "vV31 .req v4\n" "vU44 .req v4\n"
        "qV33 .req q5\n" "qU46 .req q5\n" "qU11 .req q5\n" "qU37 .req q5\n"
        "vV33 .req v5\n" "vU46 .req v5\n" "vU11 .req v5\n" "vU37 .req v5\n"
        "qU56 .req q6\n" "qU25 .req q6\n" "qU32 .req q6\n"
        "vU56 .req v6\n" "vU25 .req v6\n" "vU32 .req v6\n"
        "qU72 .req q7\n" "qV22 .req q7\n" "vU72 .req v7\n" "vV22 .req v7\n"
        "qU67 .req q8\n" "qU61 .req q8\n" "qU13 .req q8\n"
        "vU67 .req v8\n" "vU61 .req v8\n" "vU13 .req v8\n"
        "qU74 .req q9\n" "qU34 .req q9\n" "qU17 .req q9\n" "qU66 .req q9\n"
        "vU74 .req v9\n" "vU34 .req v9\n" "vU17 .req v9\n" "vU66 .req v9\n"
        "qU33 .req q10\n" "qU57 .req q10\n" "qU21 .req q10\n"
        "vU33 .req v10\n" "vU57 .req v10\n" "vU21 .req v10\n" "qW23 .req q11\n"
        "vW23 .req v11\n" "qU42 .req q12\n" "qV23 .req q12\n" "qU23 .req q12\n"
        "vU42 .req v12\n" "vV23 .req v12\n" "vU23 .req v12\n"
        "qW33 .req q13\n" "vW33 .req v13\n"
        "qU76 .req q14\n" "qU47 .req q14\n" "qU64 .req q14\n" "qU41 .req q14\n"
        "vU76 .req v14\n" "vU47 .req v14\n" "vU64 .req v14\n" "vU41 .req v14\n"
        "qU52 .req q15\n" "qU54 .req q15\n" "qU75 .req q15\n" "qU26 .req q15\n"
        "vU52 .req v15\n" "vU54 .req v15\n" "vU75 .req v15\n" "vU26 .req v15\n"
        "qU53 .req q16\n" "qU27 .req q16\n" "vU53 .req v16\n" "vU27 .req v16\n"
        "qV21 .req q17\n" "qU65 .req q17\n" "vV21 .req v17\n" "vU65 .req v17\n"
        "qU31 .req q18\n" "qU24 .req q18\n" "qU36 .req q18\n"
        "vU31 .req v18\n" "vU24 .req v18\n" "vU36 .req v18\n" "qU22 .req q19\n"
        "vU22 .req v19\n" "qU35 .req q20\n" "qU63 .req q20\n"
        "vU35 .req v20\n" "vU63 .req v20\n" "qW12 .req q21\n"
        "vW12 .req v21\n" "qV32 .req q22\n" "qU16 .req q22\n"
        "vV32 .req v22\n" "vU16 .req v22\n" "qW11 .req q23\n" "vW11 .req v23\n"
        "qU12 .req q24\n" "vU12 .req v24\n" "qW31 .req q25\n" "vW31 .req v25\n"
        "qW22 .req q26\n" "vW22 .req v26\n" "qU71 .req q27\n" "vU71 .req v27\n"
        "qV11 .req q28\n" "vV11 .req v28\n" "qW21 .req q29\n" "vW21 .req v29\n"
        "qW32 .req q30\n" "vW32 .req v30\n"

        "uptr1 .req x0\n"
        "uptr2 .req x1\n"
        "uptr3 .req x2\n"
        "uptr4 .req x3\n"
        "uptr5 .req x4\n"
        "uptr6 .req x5\n"
        "u_col_stride1 .req %x[u_col_stride]\n"
        "u_col_stride2 .req  x6\n"
        "u_col_stride3 .req  x7\n"
        "u_col_stride4 .req  x8\n"
        "u_col_stride5 .req  x9\n"
        "u_col_stride6 .req x10\n"
        "wptr1 .req x11\n"
        "wptr2 .req x12\n"
        "w_col_stride1 .req %x[w_col_stride]\n"
        "w_col_stride2 .req x13\n"
        "vptr1 .req x14\n"
        "vptr2 .req x15\n"
        "v_col_stride1 .req %x[v_col_stride]\n"
        "v_col_stride2 .req x16\n"

        // Prepare strides and pointers
        "add uptr1, %x[uptr0], %x[u_row_stride]\n"
        "add uptr2,    uptr1 , %x[u_row_stride]\n"
        "add uptr3,    uptr2 , %x[u_row_stride]\n"
        "add uptr4,    uptr3 , %x[u_row_stride]\n"
        "add uptr5,    uptr4 , %x[u_row_stride]\n"
        "add uptr6,    uptr5 , %x[u_row_stride]\n"
        "add u_col_stride2, u_col_stride1, u_col_stride1\n"
        "add u_col_stride3, u_col_stride2, u_col_stride1\n"
        "add u_col_stride4, u_col_stride3, u_col_stride1\n"
        "add u_col_stride5, u_col_stride4, u_col_stride1\n"
        "add u_col_stride6, u_col_stride5, u_col_stride1\n"

        "add wptr1, %x[wptr0], %x[w_row_stride]\n"
        "add wptr2,    wptr1 , %x[w_row_stride]\n"
        "add w_col_stride2, w_col_stride1, w_col_stride1\n"

        "add vptr1, %x[vptr0], %x[v_row_stride]\n"
        "add vptr2,    vptr1 , %x[v_row_stride]\n"
        "add v_col_stride2, v_col_stride1, v_col_stride1\n"

        // Prepare for first iteration
        "ldr qW13, [%x[wptr0], w_col_stride2]\n"
        "ldr qW23, [wptr1, w_col_stride2]\n"
        "ldr qW33, [wptr2, w_col_stride2]\n"
        "ldr qW12, [%x[wptr0], w_col_stride1]\n"
        "ldr qW22, [wptr1, w_col_stride1]\n"
        "ldr qW32, [wptr2, w_col_stride1]\n"
        "ldr qW11, [%x[wptr0]], #0x10\n"
        "ldr qW21, [wptr1], #0x10\n"
        "ldr qU17, [%x[uptr0], u_col_stride6]\n"
        "ldr qU15, [%x[uptr0], u_col_stride4]\n"
        "ldr qU16, [%x[uptr0], u_col_stride5]\n"
        "ldr qU37, [uptr2, u_col_stride6]\n"
        "ldr qU35, [uptr2, u_col_stride4]\n"
        "ldr qU36, [uptr2, u_col_stride5]\n"
        "ldr qU27, [uptr1, u_col_stride6]\n"
        "ldr qU25, [uptr1, u_col_stride4]\n"
        "fmul vV13.4s, vU17.4s, vW13.4s\n"
        "fmul vV12.4s, vU15.4s, vW13.4s\n"
        "fmla vV13.4s, vU15.4s, vW11.4s\n"
        "ldr qW31, [wptr2], #0x10\n"
        "fmla vV13.4s, vU16.4s, vW12.4s\n"
        "ldr qU26, [uptr1, u_col_stride5]\n"
        "fmla vV13.4s, vU37.4s, vW33.4s\n"
        "ldr qU47, [uptr3, u_col_stride6]\n"
        "fmul vV23.4s, vU37.4s, vW13.4s\n"
        "ldr qU45, [uptr3, u_col_stride4]\n"
        "fmla vV12.4s, vU35.4s, vW33.4s\n"
        "ldr qU46, [uptr3, u_col_stride5]\n"
        "fmla vV13.4s, vU35.4s, vW31.4s\n"
        "ldr qU67, [uptr5, u_col_stride6]\n"
        "fmul vV22.4s, vU35.4s, vW13.4s\n"
        "cbz %x[n_iters], 2f\n"  // Jump to tail if no iterations

        "1:"  // Loop body
        "fmla vV23.4s, vU35.4s, vW11.4s\n"
        "ldr qU65, [uptr5, u_col_stride4]\n"
        "fmla vV13.4s, vU36.4s, vW32.4s\n"
        "fmla vV23.4s, vU36.4s, vW12.4s\n"
        "ldr qU66, [uptr5, u_col_stride5]\n"
        "fmla vV13.4s, vU27.4s, vW23.4s\n"
        "ldr qU57, [uptr4, u_col_stride6]\n"
        "fmla vV12.4s, vU25.4s, vW23.4s\n"
        "ldr qU55, [uptr4, u_col_stride4]\n"
        "fmla vV13.4s, vU25.4s, vW21.4s\n"
        "ldr qU56, [uptr4, u_col_stride5]\n"
        "fmla vV13.4s, vU26.4s, vW22.4s\n"
        "str qV13, [%x[vptr0], v_col_stride2]\n"
        "fmla vV23.4s, vU47.4s, vW23.4s\n"
        "ldr qU77, [uptr6, u_col_stride6]\n"
        "fmla vV22.4s, vU45.4s, vW23.4s\n"
        "fmla vV23.4s, vU45.4s, vW21.4s\n"
        "ldr qU75, [uptr6, u_col_stride4]\n"
        "fmla vV23.4s, vU46.4s, vW22.4s\n"
        "ldr qU76, [uptr6, u_col_stride5]\n"
        "fmul vV33.4s, vU67.4s, vW23.4s\n"
        "ldr qU14, [%x[uptr0], u_col_stride3]\n"
        "fmul vV32.4s, vU65.4s, vW23.4s\n"
        "fmla vV33.4s, vU65.4s, vW21.4s\n"
        "ldr qU13, [%x[uptr0], u_col_stride2]\n"
        "fmla vV33.4s, vU66.4s, vW22.4s\n"
        "ldr qU34, [uptr2, u_col_stride3]\n"
        "fmla vV23.4s, vU57.4s, vW33.4s\n"
        "fmla vV33.4s, vU57.4s, vW13.4s\n"
        "ldr qU33, [uptr2, u_col_stride2]\n"
        "fmla vV22.4s, vU55.4s, vW33.4s\n"
        "fmla vV23.4s, vU55.4s, vW31.4s\n"
        "fmla vV32.4s, vU55.4s, vW13.4s\n"
        "fmla vV33.4s, vU55.4s, vW11.4s\n"
        "ldr qU24, [uptr1, u_col_stride3]\n"
        "fmla vV23.4s, vU56.4s, vW32.4s\n"
        "str qV23, [vptr1, v_col_stride2]\n"
        "fmla vV33.4s, vU56.4s, vW12.4s\n"
        "ldr qU23, [uptr1, u_col_stride2]\n"
        "fmla vV33.4s, vU77.4s, vW33.4s\n"
        "ldr qU44, [uptr3, u_col_stride3]\n"
        "fmla vV32.4s, vU75.4s, vW33.4s\n"
        "fmla vV33.4s, vU75.4s, vW31.4s\n"
        "ldr qU43, [uptr3, u_col_stride2]\n"
        "fmla vV33.4s, vU76.4s, vW32.4s\n"
        "str qV33, [vptr2, v_col_stride2]\n"
        "ldr qU64, [uptr5, u_col_stride3]\n"
        "fmla vV12.4s, vU14.4s, vW12.4s\n"
        "ldr qU63, [uptr5, u_col_stride2]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "fmla vV12.4s, vU13.4s, vW11.4s\n"
        "ldr qU54, [uptr4, u_col_stride3]\n"
        "fmla vV12.4s, vU34.4s, vW32.4s\n"
        "fmla vV22.4s, vU34.4s, vW12.4s\n"
        "ldr qU53, [uptr4, u_col_stride2]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "ldr qU74, [uptr6, u_col_stride3]\n"
        "fmla vV12.4s, vU33.4s, vW31.4s\n"
        "ldr qU73, [uptr6, u_col_stride2]\n"
        "fmul vV21.4s, vU33.4s, vW13.4s\n"
        "ldr qU12, [%x[uptr0], u_col_stride1]\n"
        "fmla vV22.4s, vU33.4s, vW11.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV12.4s, vU24.4s, vW22.4s\n"
        "ldr qU32, [uptr2, u_col_stride1]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV12.4s, vU23.4s, vW21.4s\n"
        "str qV12, [%x[vptr0], v_col_stride1]\n"
        "fmla vV22.4s, vU44.4s, vW22.4s\n"
        "ldr qU22, [uptr1, u_col_stride1]\n"
        "fmla vV21.4s, vU43.4s, vW23.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV22.4s, vU43.4s, vW21.4s\n"
        "ldr qU42, [uptr3, u_col_stride1]\n"
        "fmla vV32.4s, vU64.4s, vW22.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmul vV31.4s, vU63.4s, vW23.4s\n"
        "ldr qW23, [wptr1, w_col_stride2]\n"
        "fmla vV32.4s, vU63.4s, vW21.4s\n"
        "ldr qU62, [uptr5, u_col_stride1]\n"
        "fmla vV22.4s, vU54.4s, vW32.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV32.4s, vU54.4s, vW12.4s\n"
        "ldr qU52, [uptr4, u_col_stride1]\n"
        "fmla vV21.4s, vU53.4s, vW33.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV22.4s, vU53.4s, vW31.4s\n"
        "str qV22, [vptr1, v_col_stride1]\n"
        "fmla vV31.4s, vU53.4s, vW13.4s\n"
        "ldr qW13, [%x[wptr0], w_col_stride2]\n"
        "fmla vV32.4s, vU53.4s, vW11.4s\n"
        "ldr qU72, [uptr6, u_col_stride1]\n"
        "fmla vV32.4s, vU74.4s, vW32.4s\n"
        "ldr qU71, [uptr6], #0x10\n"
        "fmla vV31.4s, vU73.4s, vW33.4s\n"
        "ldr qW33, [wptr2, w_col_stride2]\n"
        "fmla vV32.4s, vU73.4s, vW31.4s\n"
        "str qV32, [vptr2, v_col_stride1]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "ldr qU17, [%x[uptr0], u_col_stride6]\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "ldr qU15, [%x[uptr0], u_col_stride4]\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "ldr qU16, [%x[uptr0], u_col_stride5]\n"
        "fmla vV21.4s, vU32.4s, vW12.4s\n"
        "ldr qU37, [uptr2, u_col_stride6]\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "ldr qU35, [uptr2, u_col_stride4]\n"
        "fmla vV21.4s, vU31.4s, vW11.4s\n"
        "ldr qU36, [uptr2, u_col_stride5]\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "ldr qU27, [uptr1, u_col_stride6]\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW22.4s\n"
        "ldr qU25, [uptr1, u_col_stride4]\n"
        "fmla vV21.4s, vU41.4s, vW21.4s\n"
        "fmla vV31.4s, vU62.4s, vW22.4s\n"
        "ldr qW22, [wptr1, w_col_stride1]\n"
        "fmla vV31.4s, vU61.4s, vW21.4s\n"
        "ldr qW21, [wptr1], #0x10\n"
        "fmla vV21.4s, vU52.4s, vW32.4s\n"
        "fmla vV31.4s, vU52.4s, vW12.4s\n"
        "ldr qW12, [%x[wptr0], w_col_stride1]\n"
        "fmla vV21.4s, vU51.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU51.4s, vW11.4s\n"
        "ldr qW11, [%x[wptr0]], #0x10\n"
        "fmla vV31.4s, vU72.4s, vW32.4s\n"
        "ldr qW32, [wptr2, w_col_stride1]\n"
        "fmla vV31.4s, vU71.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"
        "fmul vV13.4s, vU17.4s, vW13.4s\n"
        "fmul vV12.4s, vU15.4s, vW13.4s\n"
        "subs %x[n_iters], %x[n_iters], #1\n"
        "fmla vV13.4s, vU15.4s, vW11.4s\n"
        "ldr qW31, [wptr2], #0x10\n"
        "fmla vV13.4s, vU16.4s, vW12.4s\n"
        "ldr qU26, [uptr1, u_col_stride5]\n"
        "fmla vV13.4s, vU37.4s, vW33.4s\n"
        "ldr qU47, [uptr3, u_col_stride6]\n"
        "fmul vV23.4s, vU37.4s, vW13.4s\n"
        "ldr qU45, [uptr3, u_col_stride4]\n"
        "fmla vV12.4s, vU35.4s, vW33.4s\n"
        "ldr qU46, [uptr3, u_col_stride5]\n"
        "fmla vV13.4s, vU35.4s, vW31.4s\n"
        "ldr qU67, [uptr5, u_col_stride6]\n"
        "fmul vV22.4s, vU35.4s, vW13.4s\n"
        "bne 1b\n"

        "2:"  // Tail iteration
        "fmla vV23.4s, vU35.4s, vW11.4s\n"
        "ldr qU65, [uptr5, u_col_stride4]\n"
        "fmla vV13.4s, vU36.4s, vW32.4s\n"
        "fmla vV23.4s, vU36.4s, vW12.4s\n"
        "ldr qU66, [uptr5, u_col_stride5]\n"
        "fmla vV13.4s, vU27.4s, vW23.4s\n"
        "ldr qU57, [uptr4, u_col_stride6]\n"
        "fmla vV12.4s, vU25.4s, vW23.4s\n"
        "ldr qU55, [uptr4, u_col_stride4]\n"
        "fmla vV13.4s, vU25.4s, vW21.4s\n"
        "ldr qU56, [uptr4, u_col_stride5]\n"
        "fmla vV13.4s, vU26.4s, vW22.4s\n"
        "str qV13, [%x[vptr0], v_col_stride2]\n"
        "fmla vV23.4s, vU47.4s, vW23.4s\n"
        "ldr qU77, [uptr6, u_col_stride6]\n"
        "fmla vV22.4s, vU45.4s, vW23.4s\n"
        "fmla vV23.4s, vU45.4s, vW21.4s\n"
        "ldr qU75, [uptr6, u_col_stride4]\n"
        "fmla vV23.4s, vU46.4s, vW22.4s\n"
        "ldr qU76, [uptr6, u_col_stride5]\n"
        "fmul vV33.4s, vU67.4s, vW23.4s\n"
        "ldr qU14, [%x[uptr0], u_col_stride3]\n"
        "fmul vV32.4s, vU65.4s, vW23.4s\n"
        "fmla vV33.4s, vU65.4s, vW21.4s\n"
        "ldr qU13, [%x[uptr0], u_col_stride2]\n"
        "fmla vV33.4s, vU66.4s, vW22.4s\n"
        "ldr qU34, [uptr2, u_col_stride3]\n"
        "fmla vV23.4s, vU57.4s, vW33.4s\n"
        "fmla vV33.4s, vU57.4s, vW13.4s\n"
        "ldr qU33, [uptr2, u_col_stride2]\n"
        "fmla vV22.4s, vU55.4s, vW33.4s\n"
        "fmla vV23.4s, vU55.4s, vW31.4s\n"
        "fmla vV32.4s, vU55.4s, vW13.4s\n"
        "fmla vV33.4s, vU55.4s, vW11.4s\n"
        "ldr qU24, [uptr1, u_col_stride3]\n"
        "fmla vV23.4s, vU56.4s, vW32.4s\n"
        "str qV23, [vptr1, v_col_stride2]\n"
        "fmla vV33.4s, vU56.4s, vW12.4s\n"
        "ldr qU23, [uptr1, u_col_stride2]\n"
        "fmla vV33.4s, vU77.4s, vW33.4s\n"
        "ldr qU44, [uptr3, u_col_stride3]\n"
        "fmla vV32.4s, vU75.4s, vW33.4s\n"
        "fmla vV33.4s, vU75.4s, vW31.4s\n"
        "ldr qU43, [uptr3, u_col_stride2]\n"
        "fmla vV33.4s, vU76.4s, vW32.4s\n"
        "str qV33, [vptr2, v_col_stride2]\n"
        "ldr qU64, [uptr5, u_col_stride3]\n"
        "fmla vV12.4s, vU14.4s, vW12.4s\n"
        "ldr qU63, [uptr5, u_col_stride2]\n"
        "fmul vV11.4s, vU13.4s, vW13.4s\n"
        "fmla vV12.4s, vU13.4s, vW11.4s\n"
        "ldr qU54, [uptr4, u_col_stride3]\n"
        "fmla vV12.4s, vU34.4s, vW32.4s\n"
        "fmla vV22.4s, vU34.4s, vW12.4s\n"
        "ldr qU53, [uptr4, u_col_stride2]\n"
        "fmla vV11.4s, vU33.4s, vW33.4s\n"
        "ldr qU74, [uptr6, u_col_stride3]\n"
        "fmla vV12.4s, vU33.4s, vW31.4s\n"
        "ldr qU73, [uptr6, u_col_stride2]\n"
        "fmul vV21.4s, vU33.4s, vW13.4s\n"
        "ldr qU12, [%x[uptr0], u_col_stride1]\n"
        "fmla vV22.4s, vU33.4s, vW11.4s\n"
        "ldr qU11, [%x[uptr0]], #0x10\n"
        "fmla vV12.4s, vU24.4s, vW22.4s\n"
        "ldr qU32, [uptr2, u_col_stride1]\n"
        "fmla vV11.4s, vU23.4s, vW23.4s\n"
        "ldr qU31, [uptr2], #0x10\n"
        "fmla vV12.4s, vU23.4s, vW21.4s\n"
        "str qV12, [%x[vptr0], v_col_stride1]\n"
        "fmla vV22.4s, vU44.4s, vW22.4s\n"
        "ldr qU22, [uptr1, u_col_stride1]\n"
        "fmla vV21.4s, vU43.4s, vW23.4s\n"
        "ldr qU21, [uptr1], #0x10\n"
        "fmla vV22.4s, vU43.4s, vW21.4s\n"
        "ldr qU42, [uptr3, u_col_stride1]\n"
        "fmla vV32.4s, vU64.4s, vW22.4s\n"
        "ldr qU41, [uptr3], #0x10\n"
        "fmul vV31.4s, vU63.4s, vW23.4s\n"
        "fmla vV32.4s, vU63.4s, vW21.4s\n"
        "ldr qU62, [uptr5, u_col_stride1]\n"
        "fmla vV22.4s, vU54.4s, vW32.4s\n"
        "ldr qU61, [uptr5], #0x10\n"
        "fmla vV32.4s, vU54.4s, vW12.4s\n"
        "ldr qU52, [uptr4, u_col_stride1]\n"
        "fmla vV21.4s, vU53.4s, vW33.4s\n"
        "ldr qU51, [uptr4], #0x10\n"
        "fmla vV22.4s, vU53.4s, vW31.4s\n"
        "str qV22, [vptr1, v_col_stride1]\n"
        "fmla vV31.4s, vU53.4s, vW13.4s\n"
        "fmla vV32.4s, vU53.4s, vW11.4s\n"
        "ldr qU72, [uptr6, u_col_stride1]\n"
        "fmla vV32.4s, vU74.4s, vW32.4s\n"
        "ldr qU71, [uptr6], #0x10\n"
        "fmla vV31.4s, vU73.4s, vW33.4s\n"
        "fmla vV32.4s, vU73.4s, vW31.4s\n"
        "str qV32, [vptr2, v_col_stride1]\n"
        "fmla vV11.4s, vU12.4s, vW12.4s\n"
        "fmla vV11.4s, vU11.4s, vW11.4s\n"
        "fmla vV11.4s, vU32.4s, vW32.4s\n"
        "fmla vV21.4s, vU32.4s, vW12.4s\n"
        "fmla vV11.4s, vU31.4s, vW31.4s\n"
        "fmla vV21.4s, vU31.4s, vW11.4s\n"
        "fmla vV11.4s, vU22.4s, vW22.4s\n"
        "fmla vV11.4s, vU21.4s, vW21.4s\n"
        "str qV11, [%x[vptr0]], #0x10\n"
        "fmla vV21.4s, vU42.4s, vW22.4s\n"
        "fmla vV21.4s, vU41.4s, vW21.4s\n"
        "fmla vV31.4s, vU62.4s, vW22.4s\n"
        "fmla vV31.4s, vU61.4s, vW21.4s\n"
        "fmla vV21.4s, vU52.4s, vW32.4s\n"
        "fmla vV31.4s, vU52.4s, vW12.4s\n"
        "fmla vV21.4s, vU51.4s, vW31.4s\n"
        "str qV21, [vptr1], #0x10\n"
        "fmla vV31.4s, vU51.4s, vW11.4s\n"
        "fmla vV31.4s, vU72.4s, vW32.4s\n"
        "fmla vV31.4s, vU71.4s, vW31.4s\n"
        "str qV31, [vptr2], #0x10\n"

        // Clear aliases
        ".unreq uptr1\n" ".unreq uptr2\n" ".unreq uptr3\n" ".unreq uptr4\n"
        ".unreq uptr5\n" ".unreq uptr6\n"
        ".unreq u_col_stride1\n" ".unreq u_col_stride2\n" ".unreq u_col_stride3\n"
        ".unreq u_col_stride4\n" ".unreq u_col_stride5\n" ".unreq u_col_stride6\n"
        ".unreq wptr1\n" ".unreq wptr2\n"
        ".unreq w_col_stride1\n" ".unreq w_col_stride2\n"
        ".unreq vptr1\n" ".unreq vptr2\n"
        ".unreq v_col_stride1\n" ".unreq v_col_stride2\n"
        ".unreq qU15\n" ".unreq qU73\n" ".unreq qU45\n" ".unreq qU14\n"
        ".unreq qW13\n" ".unreq qU62\n" ".unreq qV12\n"
        ".unreq qU51\n" ".unreq qU43\n" ".unreq qU55\n"
        ".unreq qU77\n" ".unreq qV13\n" ".unreq qV31\n" ".unreq qU44\n"
        ".unreq qV33\n" ".unreq qU46\n" ".unreq qU11\n" ".unreq qU37\n"
        ".unreq qU56\n" ".unreq qU25\n" ".unreq qU32\n"
        ".unreq qU72\n" ".unreq qV22\n"
        ".unreq qU67\n" ".unreq qU61\n" ".unreq qU13\n" ".unreq qW33\n"
        ".unreq qU74\n" ".unreq qU34\n" ".unreq qU17\n" ".unreq qU66\n"
        ".unreq qU33\n" ".unreq qU57\n" ".unreq qU21\n"
        ".unreq qW23\n" ".unreq qU42\n" ".unreq qV23\n" ".unreq qU23\n"
        ".unreq qU76\n" ".unreq qU47\n" ".unreq qU64\n" ".unreq qU41\n"
        ".unreq qU52\n" ".unreq qU54\n" ".unreq qU75\n" ".unreq qU26\n"
        ".unreq qU53\n" ".unreq qU27\n"
        ".unreq qV21\n" ".unreq qU65\n"
        ".unreq qU31\n" ".unreq qU24\n" ".unreq qU36\n" ".unreq qU22\n"
        ".unreq qU35\n" ".unreq qU63\n" ".unreq qW12\n"
        ".unreq qV32\n" ".unreq qU16\n" ".unreq qW11\n" ".unreq qU12\n"
        ".unreq qW31\n" ".unreq qW22\n" ".unreq qU71\n" ".unreq qV11\n"
        ".unreq qW21\n" ".unreq qW32\n" ".unreq vW13\n"
        ".unreq vU15\n" ".unreq vU73\n" ".unreq vU45\n" ".unreq vU14\n"
        ".unreq vU62\n" ".unreq vV12\n"
        ".unreq vU51\n" ".unreq vU43\n" ".unreq vU55\n"
        ".unreq vU77\n" ".unreq vV13\n" ".unreq vV31\n" ".unreq vU44\n"
        ".unreq vV33\n" ".unreq vU46\n" ".unreq vU11\n" ".unreq vU37\n"
        ".unreq vU56\n" ".unreq vU25\n" ".unreq vU32\n"
        ".unreq vU72\n" ".unreq vV22\n" ".unreq vW21\n" ".unreq vW32\n"
        ".unreq vU67\n" ".unreq vU61\n" ".unreq vU13\n"
        ".unreq vU74\n" ".unreq vU34\n" ".unreq vU17\n" ".unreq vU66\n"
        ".unreq vU33\n" ".unreq vU57\n" ".unreq vU21\n" ".unreq vW23\n"
        ".unreq vU42\n" ".unreq vV23\n" ".unreq vU23\n" ".unreq vW33\n"
        ".unreq vU76\n" ".unreq vU47\n" ".unreq vU64\n" ".unreq vU41\n"
        ".unreq vU52\n" ".unreq vU54\n" ".unreq vU75\n" ".unreq vU26\n"
        ".unreq vU53\n" ".unreq vU27\n" ".unreq vV21\n" ".unreq vU65\n"
        ".unreq vU31\n" ".unreq vU24\n" ".unreq vU36\n" ".unreq vU22\n"
        ".unreq vU35\n" ".unreq vU63\n" ".unreq vW12\n"
        ".unreq vV32\n" ".unreq vU16\n" ".unreq vW11\n" ".unreq vU12\n"
        ".unreq vW31\n" ".unreq vW22\n" ".unreq vU71\n" ".unreq vV11\n"
        : [uptr0] "+r" (uptr0), [wptr0] "+r" (wptr0), [vptr0] "+r" (vptr0),
          [n_iters] "+r" (n_iters)
        : [u_row_stride] "r" (in_row_stride * sizeof(float)),
          [u_col_stride] "r" (in_col_stride * sizeof(float)),
          [w_row_stride] "r" (weight_row_stride * sizeof(float)),
          [w_col_stride] "r" (weight_col_stride * sizeof(float)),
          [v_row_stride] "r" (out_row_stride * sizeof(float)),
          [v_col_stride] "r" (out_col_stride * sizeof(float))
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0",
          "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
          "x12", "x13", "x14", "x15", "x16", "cc", "memory"
    );
  }
  if (channels_remaining)
  {
    // Fall back on the unoptimised version to clean up the tail
    ConvImpl::process_tile<false>(
        channels_remaining,
        wptr0, weight_row_stride, weight_col_stride,
        uptr0, in_row_stride, in_col_stride,
        vptr0, out_row_stride, out_col_stride,
        0, 0, 0, 0, 0, 0
    );
  }
}

#endif  // __aarch64__

template <>
const Conv::TileFn Conv::tilefn_unpadded = ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>;

template <>
const Conv::TileFn Conv::tilefn_top[n_in_pad_top_fns] = {
  ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
  ConvImpl::template process_tile<true, 1, 0, 0, 0, 0, 0>,
};

template <>
const Conv::TileFn Conv::tilefn_left[n_in_pad_left_fns] = {
  ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
  ConvImpl::template process_tile<true, 0, 1, 0, 0, 0, 0>,
};

template <>
const Conv::TileFn Conv::tilefn_bottom[n_in_pad_bottom_fns][n_out_pad_bottom_fns] = {
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 4, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 5, 0, 2, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 6, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 6, 0, 1, 0>,
    ConvImpl::template process_tile<true, 0, 0, 6, 0, 2, 0>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_right[n_in_pad_right_fns][n_out_pad_right_fns] = {
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 4, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 5, 0, 2>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 1>,
    ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 2>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_generic = ConvImpl::template process_tile<false>;

template class DepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float>;
}  // namespace depthwise
