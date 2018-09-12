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
using Conv = DepthwiseConvolution<3, 3, 3, 3, 1, 1, float, float>;
using ConvImpl = DepthwiseConvolutionImpl<3, 3, 3, 3, 1, 1, float, float>;

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
    int n_iters = ((channels_remaining / 4) + 1)/2 - 1;
    const bool odd_tail = (channels_remaining / 4) & 1;
    channels_remaining %= 4;

    asm volatile (
        "qU22B .req q0\n" "qU23B .req q0\n" "qW22A .req q0\n"
        "vU22B .req v0\n" "vU23B .req v0\n" "vW22A .req v0\n"
        "qV12A .req q1\n" "qW11B .req q1\n"
        "vV12A .req v1\n" "vW11B .req v1\n"
        "qU41A .req q2\n" "qU32B .req q2\n" "qU33A .req q2\n" "qV13B .req q2\n"
        "vU41A .req v2\n" "vU32B .req v2\n" "vU33A .req v2\n" "vV13B .req v2\n"
        "qU42B .req q3\n" "qU13B .req q3\n" "qU44B .req q3\n" "qU55A .req q3\n"
        "vU42B .req v3\n" "vU13B .req v3\n" "vU44B .req v3\n" "vU55A .req v3\n"
        "qU34B .req q4\n" "qU15A .req q4\n" "qU42A .req q4\n" "qU44A .req q4\n" "qU12B .req q4\n"
        "vU34B .req v4\n" "vU15A .req v4\n" "vU42A .req v4\n" "vU44A .req v4\n" "vU12B .req v4\n"
        "qU33B .req q5\n" "qU52A .req q5\n" "qW23A .req q5\n"
        "vU33B .req v5\n" "vU52A .req v5\n" "vW23A .req v5\n"
        "qV31A .req q6\n" "qU13A .req q6\n" "qV12B .req q6\n"
        "vV31A .req v6\n" "vU13A .req v6\n" "vV12B .req v6\n"
        "qU35B .req q7\n" "qU51B .req q7\n" "qV11A .req q7\n" "qU53B .req q7\n"
        "vU35B .req v7\n" "vU51B .req v7\n" "vV11A .req v7\n" "vU53B .req v7\n"
        "qW21A .req q8\n" "qV22B .req q8\n"
        "vW21A .req v8\n" "vV22B .req v8\n"
        "qV33B .req q9\n" "qU14A .req q9\n" "qV23A .req q9\n" "qU25B .req q9\n"
        "vV33B .req v9\n" "vU14A .req v9\n" "vV23A .req v9\n" "vU25B .req v9\n"
        "qW21B .req q10\n" "qV32A .req q10\n" "qU35A .req q10\n"
        "vW21B .req v10\n" "vV32A .req v10\n" "vU35A .req v10\n"
        "qV11B .req q11\n" "qU15B .req q11\n" "qV33A .req q11\n"
        "vV11B .req v11\n" "vU15B .req v11\n" "vV33A .req v11\n"
        "qU11B .req q12\n" "qW23B .req q12\n" "qU45A .req q12\n"
        "vU11B .req v12\n" "vW23B .req v12\n" "vU45A .req v12\n"
        "qW11A .req q13\n" "qU45B .req q13\n" "qU52B .req q13\n"
        "vW11A .req v13\n" "vU45B .req v13\n" "vU52B .req v13\n"
        "qU55B .req q14\n" "qU25A .req q14\n" "qV21A .req q14\n"
        "vU55B .req v14\n" "vU25A .req v14\n" "vV21A .req v14\n"
        "qU53A .req q15\n" "qV21B .req q15\n" "qU31A .req q15\n"
        "vU53A .req v15\n" "vV21B .req v15\n" "vU31A .req v15\n"
        "qW13B .req q16\n" "qU23A .req q16\n"
        "vW13B .req v16\n" "vU23A .req v16\n"
        "qW33B .req q17\n" "qW33A .req q17\n"
        "vW33B .req v17\n" "vW33A .req v17\n"
        "qU24B .req q18\n" "qU32A .req q18\n" "qV31B .req q18\n" "qV13A .req q18\n"
        "vU24B .req v18\n" "vU32A .req v18\n" "vV31B .req v18\n" "vV13A .req v18\n"
        "qU31B .req q19\n" "qU11A .req q19\n" "qU54B .req q19\n" "qU43A .req q19\n"
        "vU31B .req v19\n" "vU11A .req v19\n" "vU54B .req v19\n" "vU43A .req v19\n"
        "qU24A .req q20\n" "qW12B .req q20\n" "qU54A .req q20\n"
        "vU24A .req v20\n" "vW12B .req v20\n" "vU54A .req v20\n"
        "qV23B .req q21\n" "qW12A .req q21\n"
        "vV23B .req v21\n" "vW12A .req v21\n"
        "qW32A .req q22\n" "qU43B .req q22\n"
        "vW32A .req v22\n" "vU43B .req v22\n"
        "qW31A .req q23\n" "qV32B .req q23\n"
        "vW31A .req v23\n" "vV32B .req v23\n"
        "qU22A .req q24\n" "qW31B .req q24\n"
        "vU22A .req v24\n" "vW31B .req v24\n"
        "qU21B .req q25\n" "qV22A .req q25\n"
        "vU21B .req v25\n" "vV22A .req v25\n"
        "qU34A .req q26\n" "qW22B .req q26\n" "qU12A .req q26\n"
        "vU34A .req v26\n" "vW22B .req v26\n" "vU12A .req v26\n"
        "qW13A .req q27\n" "qU51A .req q27\n"
        "vW13A .req v27\n" "vU51A .req v27\n"
        "qW32B .req q28\n"
        "vW32B .req v28\n"
        "qU41B .req q29\n" "qU14B .req q29\n"
        "vU41B .req v29\n" "vU14B .req v29\n"
        "qU21A .req q30\n"
        "vU21A .req v30\n"

        "uptr1 .req x0\n"
        "uptr2 .req x1\n"
        "uptr3 .req x2\n"
        "uptr4 .req x3\n"

        "u_col_stride1 .req %x[u_col_stride]\n"
        "u_col_stride2 .req x4\n"
        "u_col_stride3 .req x5\n"
        "u_col_stride4 .req x6\n"

        "wptr1 .req x7\n"
        "wptr2 .req x8\n"
        "w_col_stride1 .req %x[w_col_stride]\n"
        "w_col_stride2 .req x9\n"

        "vptr1 .req x10\n"
        "vptr2 .req x11\n"
        "v_col_stride1 .req %x[v_col_stride]\n"
        "v_col_stride2 .req x12\n"

        // Prepare strides and pointers
        "add uptr1, %x[uptr0], %x[u_row_stride]\n"
        "add uptr2,    uptr1 , %x[u_row_stride]\n"
        "add uptr3,    uptr2 , %x[u_row_stride]\n"
        "add uptr4,    uptr3 , %x[u_row_stride]\n"
        "add u_col_stride2, u_col_stride1, u_col_stride1\n"
        "add u_col_stride3, u_col_stride2, u_col_stride1\n"
        "add u_col_stride4, u_col_stride3, u_col_stride1\n"

        "add wptr1, %x[wptr0], %x[w_row_stride]\n"
        "add wptr2,    wptr1 , %x[w_row_stride]\n"
        "add w_col_stride2, w_col_stride1, w_col_stride1\n"

        "add vptr1, %x[vptr0], %x[v_row_stride]\n"
        "add vptr2,    vptr1 , %x[v_row_stride]\n"
        "add v_col_stride2, v_col_stride1, v_col_stride1\n"

        // Pre-load for A
        "ldr qW13A, [%x[wptr0], w_col_stride2]\n"
        "ldr qW23A, [wptr1, w_col_stride2]\n"
        "ldr qW33A, [wptr2, w_col_stride2]\n"
        "ldr qW12A, [%x[wptr0], w_col_stride1]\n"
        "ldr qU15A, [%x[uptr0], u_col_stride4]\n"
        "ldr qW22A, [wptr1, w_col_stride1]\n"
        "ldr qU14A, [%x[uptr0], u_col_stride3]\n"
        "ldr qW32A, [wptr2, w_col_stride1]\n"
        "ldr qU13A, [%x[uptr0], u_col_stride2]\n"
        "ldr qU25A, [uptr1, u_col_stride4]\n"
        "ldr qU24A, [uptr1, u_col_stride3]\n"
        "ldr qW11A, [%x[wptr0]], #0x10\n"
        "ldr qU23A, [uptr1, u_col_stride2]\n"
        "ldr qW21A, [wptr1], #0x10\n"
        "ldr qW31A, [wptr2], #0x10\n"
        "ldr qU34A, [uptr2, u_col_stride3]\n"
        "ldr qU35A, [uptr2, u_col_stride4]\n"

        // First part of A
        "fmul vV13A.4s, vU15A.4s, vW13A.4s\n"
        "ldr qU33A, [uptr2, u_col_stride2]\n"
        "fmul vV12A.4s, vU14A.4s, vW13A.4s\n"
        "cbz %x[n_iters], 2f\n"  // Jump to tail if not looping

        "1:"  // Main loop, double unrolled
        // A Part
        "fmla vV13A.4s, vU14A.4s, vW12A.4s\n"
        "ldr qU45A, [uptr3, u_col_stride4]\n"
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV13A.4s, vU13A.4s, vW11A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV13A.4s, vU25A.4s, vW23A.4s\n"
        "fmul vV23A.4s, vU25A.4s, vW13A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmla vV13A.4s, vU24A.4s, vW22A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV23A.4s, vU24A.4s, vW12A.4s\n"
        "ldr qU55A, [uptr4, u_col_stride4]\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "fmla vV13A.4s, vU23A.4s, vW21A.4s\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "fmla vV23A.4s, vU23A.4s, vW11A.4s\n"
        "ldr qU54A, [uptr4, u_col_stride3]\n"
        "fmla vV13A.4s, vU35A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU35A.4s, vW23A.4s\n"
        "fmul vV33A.4s, vU35A.4s, vW13A.4s\n"
        "ldr qU53A, [uptr4, u_col_stride2]\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "fmla vV13A.4s, vU34A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "fmla vV23A.4s, vU34A.4s, vW22A.4s\n"
        "fmul vV32A.4s, vU34A.4s, vW13A.4s\n"
        "fmla vV33A.4s, vU34A.4s, vW12A.4s\n"
        "ldr qU12A, [%x[uptr0], u_col_stride1]\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "fmla vV13A.4s, vU33A.4s, vW31A.4s\n"
        "str qV13A, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "fmla vV23A.4s, vU33A.4s, vW21A.4s\n"
        "fmul vV31A.4s, vU33A.4s, vW13A.4s\n"
        "ldr qW13B, [%x[wptr0], w_col_stride2]\n"
        "fmla vV32A.4s, vU33A.4s, vW12A.4s\n"
        "fmla vV33A.4s, vU33A.4s, vW11A.4s\n"
        "ldr qU22A, [uptr1, u_col_stride1]\n"
        "fmla vV23A.4s, vU45A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU45A.4s, vW23A.4s\n"
        "ldr qU32A, [uptr2, u_col_stride1]\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU44A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU44A.4s, vW23A.4s\n"
        "fmla vV33A.4s, vU44A.4s, vW22A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "fmla vV23A.4s, vU43A.4s, vW31A.4s\n"
        "str qV23A, [vptr1, v_col_stride2]\n"
        "fmla vV31A.4s, vU43A.4s, vW23A.4s\n"
        "ldr qW23B, [wptr1, w_col_stride2]\n"
        "fmla vV32A.4s, vU43A.4s, vW22A.4s\n"
        "fmla vV33A.4s, vU43A.4s, vW21A.4s\n"
        "ldr qU52A, [uptr4, u_col_stride1]\n"
        "fmla vV33A.4s, vU55A.4s, vW33A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV32A.4s, vU54A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU54A.4s, vW32A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV31A.4s, vU53A.4s, vW33A.4s\n"
        "ldr qW33B, [wptr2, w_col_stride2]\n"
        "fmla vV32A.4s, vU53A.4s, vW32A.4s\n"
        "fmla vV33A.4s, vU53A.4s, vW31A.4s\n"
        "str qV33A, [vptr2, v_col_stride2]\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU51A, [uptr4], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "ldr qW12B, [%x[wptr0], w_col_stride1]\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "ldr qU15B, [%x[uptr0], u_col_stride4]\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "ldr qW22B, [wptr1, w_col_stride1]\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "ldr qU14B, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], v_col_stride1]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "ldr qW32B, [wptr2, w_col_stride1]\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "ldr qU13B, [%x[uptr0], u_col_stride2]\n"
        "fmla vV31A.4s, vU32A.4s, vW12A.4s\n"
        "ldr qU25B, [uptr1, u_col_stride4]\n"
        "fmla vV32A.4s, vU32A.4s, vW11A.4s\n"
        "ldr qU24B, [uptr1, u_col_stride3]\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, v_col_stride1]\n"
        "fmla vV31A.4s, vU42A.4s, vW22A.4s\n"
        "fmla vV32A.4s, vU42A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU52A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU52A.4s, vW31A.4s\n"
        "str qV32A, [vptr2, v_col_stride1]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "ldr qW11B, [%x[wptr0]], #0x10\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "ldr qU23B, [uptr1, u_col_stride2]\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "ldr qW21B, [wptr1], #0x10\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "ldr qW31B, [wptr2], #0x10\n"
        "fmla vV31A.4s, vU31A.4s, vW11A.4s\n"
        "ldr qU34B, [uptr2, u_col_stride3]\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"
        "fmla vV31A.4s, vU41A.4s, vW21A.4s\n"
        "ldr qU35B, [uptr2, u_col_stride4]\n"
        "fmla vV31A.4s, vU51A.4s, vW31A.4s\n"
        "str qV31A, [vptr2], #0x10\n"

        // B Part
        "fmul vV13B.4s, vU15B.4s, vW13B.4s\n"
        "ldr qU33B, [uptr2, u_col_stride2]\n"
        "fmul vV12B.4s, vU14B.4s, vW13B.4s\n"
        "fmla vV13B.4s, vU14B.4s, vW12B.4s\n"
        "ldr qU45B, [uptr3, u_col_stride4]\n"
        "fmul vV11B.4s, vU13B.4s, vW13B.4s\n"
        "fmla vV12B.4s, vU13B.4s, vW12B.4s\n"
        "fmla vV13B.4s, vU13B.4s, vW11B.4s\n"
        "ldr qU44B, [uptr3, u_col_stride3]\n"
        "fmla vV13B.4s, vU25B.4s, vW23B.4s\n"
        "fmul vV23B.4s, vU25B.4s, vW13B.4s\n"
        "ldr qU43B, [uptr3, u_col_stride2]\n"
        "fmla vV12B.4s, vU24B.4s, vW23B.4s\n"
        "fmla vV13B.4s, vU24B.4s, vW22B.4s\n"
        "fmul vV22B.4s, vU24B.4s, vW13B.4s\n"
        "fmla vV23B.4s, vU24B.4s, vW12B.4s\n"
        "ldr qU55B, [uptr4, u_col_stride4]\n"
        "fmla vV11B.4s, vU23B.4s, vW23B.4s\n"
        "fmla vV12B.4s, vU23B.4s, vW22B.4s\n"
        "fmla vV13B.4s, vU23B.4s, vW21B.4s\n"
        "fmul vV21B.4s, vU23B.4s, vW13B.4s\n"
        "fmla vV22B.4s, vU23B.4s, vW12B.4s\n"
        "fmla vV23B.4s, vU23B.4s, vW11B.4s\n"
        "ldr qU54B, [uptr4, u_col_stride3]\n"
        "fmla vV13B.4s, vU35B.4s, vW33B.4s\n"
        "fmla vV23B.4s, vU35B.4s, vW23B.4s\n"
        "fmul vV33B.4s, vU35B.4s, vW13B.4s\n"
        "ldr qU53B, [uptr4, u_col_stride2]\n"
        "fmla vV12B.4s, vU34B.4s, vW33B.4s\n"
        "fmla vV13B.4s, vU34B.4s, vW32B.4s\n"
        "fmla vV22B.4s, vU34B.4s, vW23B.4s\n"
        "fmla vV23B.4s, vU34B.4s, vW22B.4s\n"
        "fmul vV32B.4s, vU34B.4s, vW13B.4s\n"
        "fmla vV33B.4s, vU34B.4s, vW12B.4s\n"
        "ldr qU12B, [%x[uptr0], u_col_stride1]\n"
        "fmla vV11B.4s, vU33B.4s, vW33B.4s\n"
        "fmla vV12B.4s, vU33B.4s, vW32B.4s\n"
        "fmla vV13B.4s, vU33B.4s, vW31B.4s\n"
        "str qV13B, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21B.4s, vU33B.4s, vW23B.4s\n"
        "fmla vV22B.4s, vU33B.4s, vW22B.4s\n"
        "fmla vV23B.4s, vU33B.4s, vW21B.4s\n"
        "fmul vV31B.4s, vU33B.4s, vW13B.4s\n"
        "ldr qW13A, [%x[wptr0], w_col_stride2]\n"
        "fmla vV32B.4s, vU33B.4s, vW12B.4s\n"
        "fmla vV33B.4s, vU33B.4s, vW11B.4s\n"
        "ldr qU22B, [uptr1, u_col_stride1]\n"
        "fmla vV23B.4s, vU45B.4s, vW33B.4s\n"
        "fmla vV33B.4s, vU45B.4s, vW23B.4s\n"
        "ldr qU32B, [uptr2, u_col_stride1]\n"
        "fmla vV22B.4s, vU44B.4s, vW33B.4s\n"
        "fmla vV23B.4s, vU44B.4s, vW32B.4s\n"
        "fmla vV32B.4s, vU44B.4s, vW23B.4s\n"
        "fmla vV33B.4s, vU44B.4s, vW22B.4s\n"
        "ldr qU42B, [uptr3, u_col_stride1]\n"
        "fmla vV21B.4s, vU43B.4s, vW33B.4s\n"
        "fmla vV22B.4s, vU43B.4s, vW32B.4s\n"
        "fmla vV23B.4s, vU43B.4s, vW31B.4s\n"
        "str qV23B, [vptr1, v_col_stride2]\n"
        "fmla vV31B.4s, vU43B.4s, vW23B.4s\n"
        "ldr qW23A, [wptr1, w_col_stride2]\n"
        "fmla vV32B.4s, vU43B.4s, vW22B.4s\n"
        "fmla vV33B.4s, vU43B.4s, vW21B.4s\n"
        "ldr qU52B, [uptr4, u_col_stride1]\n"
        "fmla vV33B.4s, vU55B.4s, vW33B.4s\n"
        "ldr qU11B, [%x[uptr0]], #0x10\n"
        "fmla vV32B.4s, vU54B.4s, vW33B.4s\n"
        "fmla vV33B.4s, vU54B.4s, vW32B.4s\n"
        "ldr qU21B, [uptr1], #0x10\n"
        "fmla vV31B.4s, vU53B.4s, vW33B.4s\n"
        "ldr qW33A, [wptr2, w_col_stride2]\n"
        "fmla vV32B.4s, vU53B.4s, vW32B.4s\n"
        "fmla vV33B.4s, vU53B.4s, vW31B.4s\n"
        "str qV33B, [vptr2, v_col_stride2]\n"
        "fmla vV11B.4s, vU12B.4s, vW12B.4s\n"
        "ldr qU31B, [uptr2], #0x10\n"
        "fmla vV12B.4s, vU12B.4s, vW11B.4s\n"
        "ldr qU41B, [uptr3], #0x10\n"
        "fmla vV11B.4s, vU22B.4s, vW22B.4s\n"
        "ldr qU51B, [uptr4], #0x10\n"
        "fmla vV12B.4s, vU22B.4s, vW21B.4s\n"
        "ldr qW12A, [%x[wptr0], w_col_stride1]\n"
        "fmla vV21B.4s, vU22B.4s, vW12B.4s\n"
        "ldr qU15A, [%x[uptr0], u_col_stride4]\n"
        "fmla vV22B.4s, vU22B.4s, vW11B.4s\n"
        "ldr qW22A, [wptr1, w_col_stride1]\n"
        "fmla vV11B.4s, vU32B.4s, vW32B.4s\n"
        "ldr qU14A, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12B.4s, vU32B.4s, vW31B.4s\n"
        "str qV12B, [%x[vptr0], v_col_stride1]\n"
        "fmla vV21B.4s, vU32B.4s, vW22B.4s\n"
        "ldr qW32A, [wptr2, w_col_stride1]\n"
        "fmla vV22B.4s, vU32B.4s, vW21B.4s\n"
        "ldr qU13A, [%x[uptr0], u_col_stride2]\n"
        "fmla vV31B.4s, vU32B.4s, vW12B.4s\n"
        "ldr qU25A, [uptr1, u_col_stride4]\n"
        "fmla vV32B.4s, vU32B.4s, vW11B.4s\n"
        "ldr qU24A, [uptr1, u_col_stride3]\n"
        "fmla vV21B.4s, vU42B.4s, vW32B.4s\n"
        "fmla vV22B.4s, vU42B.4s, vW31B.4s\n"
        "str qV22B, [vptr1, v_col_stride1]\n"
        "fmla vV31B.4s, vU42B.4s, vW22B.4s\n"
        "fmla vV32B.4s, vU42B.4s, vW21B.4s\n"
        "fmla vV31B.4s, vU52B.4s, vW32B.4s\n"
        "subs %x[n_iters], %x[n_iters], #1\n"
        "fmla vV32B.4s, vU52B.4s, vW31B.4s\n"
        "str qV32B, [vptr2, v_col_stride1]\n"
        "fmla vV11B.4s, vU11B.4s, vW11B.4s\n"
        "ldr qW11A, [%x[wptr0]], #0x10\n"
        "fmla vV11B.4s, vU21B.4s, vW21B.4s\n"
        "ldr qU23A, [uptr1, u_col_stride2]\n"
        "fmla vV21B.4s, vU21B.4s, vW11B.4s\n"
        "ldr qW21A, [wptr1], #0x10\n"
        "fmla vV11B.4s, vU31B.4s, vW31B.4s\n"
        "str qV11B, [%x[vptr0]], #0x10\n"
        "fmla vV21B.4s, vU31B.4s, vW21B.4s\n"
        "ldr qW31A, [wptr2], #0x10\n"
        "fmla vV31B.4s, vU31B.4s, vW11B.4s\n"
        "ldr qU34A, [uptr2, u_col_stride3]\n"
        "fmla vV21B.4s, vU41B.4s, vW31B.4s\n"
        "str qV21B, [vptr1], #0x10\n"
        "fmla vV31B.4s, vU41B.4s, vW21B.4s\n"
        "ldr qU35A, [uptr2, u_col_stride4]\n"
        "fmla vV31B.4s, vU51B.4s, vW31B.4s\n"
        "str qV31B, [vptr2], #0x10\n"

        // First part of A
        "fmul vV13A.4s, vU15A.4s, vW13A.4s\n"
        "ldr qU33A, [uptr2, u_col_stride2]\n"
        "fmul vV12A.4s, vU14A.4s, vW13A.4s\n"
        "bne 1b\n"  // Loop

        "2:"  // Tail dispatch
        "cbnz %w[odd_tail], 3f\n"

        // Even tail
        // A Part
        "fmla vV13A.4s, vU14A.4s, vW12A.4s\n"
        "ldr qU45A, [uptr3, u_col_stride4]\n"
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV13A.4s, vU13A.4s, vW11A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV13A.4s, vU25A.4s, vW23A.4s\n"
        "fmul vV23A.4s, vU25A.4s, vW13A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmla vV13A.4s, vU24A.4s, vW22A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV23A.4s, vU24A.4s, vW12A.4s\n"
        "ldr qU55A, [uptr4, u_col_stride4]\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "fmla vV13A.4s, vU23A.4s, vW21A.4s\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "fmla vV23A.4s, vU23A.4s, vW11A.4s\n"
        "ldr qU54A, [uptr4, u_col_stride3]\n"
        "fmla vV13A.4s, vU35A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU35A.4s, vW23A.4s\n"
        "fmul vV33A.4s, vU35A.4s, vW13A.4s\n"
        "ldr qU53A, [uptr4, u_col_stride2]\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "fmla vV13A.4s, vU34A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "fmla vV23A.4s, vU34A.4s, vW22A.4s\n"
        "fmul vV32A.4s, vU34A.4s, vW13A.4s\n"
        "fmla vV33A.4s, vU34A.4s, vW12A.4s\n"
        "ldr qU12A, [%x[uptr0], u_col_stride1]\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "fmla vV13A.4s, vU33A.4s, vW31A.4s\n"
        "str qV13A, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "fmla vV23A.4s, vU33A.4s, vW21A.4s\n"
        "fmul vV31A.4s, vU33A.4s, vW13A.4s\n"
        "ldr qW13B, [%x[wptr0], w_col_stride2]\n"
        "fmla vV32A.4s, vU33A.4s, vW12A.4s\n"
        "fmla vV33A.4s, vU33A.4s, vW11A.4s\n"
        "ldr qU22A, [uptr1, u_col_stride1]\n"
        "fmla vV23A.4s, vU45A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU45A.4s, vW23A.4s\n"
        "ldr qU32A, [uptr2, u_col_stride1]\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU44A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU44A.4s, vW23A.4s\n"
        "fmla vV33A.4s, vU44A.4s, vW22A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "fmla vV23A.4s, vU43A.4s, vW31A.4s\n"
        "str qV23A, [vptr1, v_col_stride2]\n"
        "fmla vV31A.4s, vU43A.4s, vW23A.4s\n"
        "ldr qW23B, [wptr1, w_col_stride2]\n"
        "fmla vV32A.4s, vU43A.4s, vW22A.4s\n"
        "fmla vV33A.4s, vU43A.4s, vW21A.4s\n"
        "ldr qU52A, [uptr4, u_col_stride1]\n"
        "fmla vV33A.4s, vU55A.4s, vW33A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV32A.4s, vU54A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU54A.4s, vW32A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV31A.4s, vU53A.4s, vW33A.4s\n"
        "ldr qW33B, [wptr2, w_col_stride2]\n"
        "fmla vV32A.4s, vU53A.4s, vW32A.4s\n"
        "fmla vV33A.4s, vU53A.4s, vW31A.4s\n"
        "str qV33A, [vptr2, v_col_stride2]\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU51A, [uptr4], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "ldr qW12B, [%x[wptr0], w_col_stride1]\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "ldr qU15B, [%x[uptr0], u_col_stride4]\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "ldr qW22B, [wptr1, w_col_stride1]\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "ldr qU14B, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], v_col_stride1]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "ldr qW32B, [wptr2, w_col_stride1]\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "ldr qU13B, [%x[uptr0], u_col_stride2]\n"
        "fmla vV31A.4s, vU32A.4s, vW12A.4s\n"
        "ldr qU25B, [uptr1, u_col_stride4]\n"
        "fmla vV32A.4s, vU32A.4s, vW11A.4s\n"
        "ldr qU24B, [uptr1, u_col_stride3]\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, v_col_stride1]\n"
        "fmla vV31A.4s, vU42A.4s, vW22A.4s\n"
        "fmla vV32A.4s, vU42A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU52A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU52A.4s, vW31A.4s\n"
        "str qV32A, [vptr2, v_col_stride1]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "ldr qW11B, [%x[wptr0]], #0x10\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "ldr qU23B, [uptr1, u_col_stride2]\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "ldr qW21B, [wptr1], #0x10\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "ldr qW31B, [wptr2], #0x10\n"
        "fmla vV31A.4s, vU31A.4s, vW11A.4s\n"
        "ldr qU34B, [uptr2, u_col_stride3]\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"
        "fmla vV31A.4s, vU41A.4s, vW21A.4s\n"
        "ldr qU35B, [uptr2, u_col_stride4]\n"
        "fmla vV31A.4s, vU51A.4s, vW31A.4s\n"
        "str qV31A, [vptr2], #0x10\n"

        // B Part
        "fmul vV13B.4s, vU15B.4s, vW13B.4s\n"
        "ldr qU33B, [uptr2, u_col_stride2]\n"
        "fmul vV12B.4s, vU14B.4s, vW13B.4s\n"
        "fmla vV13B.4s, vU14B.4s, vW12B.4s\n"
        "ldr qU45B, [uptr3, u_col_stride4]\n"
        "fmul vV11B.4s, vU13B.4s, vW13B.4s\n"
        "fmla vV12B.4s, vU13B.4s, vW12B.4s\n"
        "fmla vV13B.4s, vU13B.4s, vW11B.4s\n"
        "ldr qU44B, [uptr3, u_col_stride3]\n"
        "fmla vV13B.4s, vU25B.4s, vW23B.4s\n"
        "fmul vV23B.4s, vU25B.4s, vW13B.4s\n"
        "ldr qU43B, [uptr3, u_col_stride2]\n"
        "fmla vV12B.4s, vU24B.4s, vW23B.4s\n"
        "fmla vV13B.4s, vU24B.4s, vW22B.4s\n"
        "fmul vV22B.4s, vU24B.4s, vW13B.4s\n"
        "fmla vV23B.4s, vU24B.4s, vW12B.4s\n"
        "ldr qU55B, [uptr4, u_col_stride4]\n"
        "fmla vV11B.4s, vU23B.4s, vW23B.4s\n"
        "fmla vV12B.4s, vU23B.4s, vW22B.4s\n"
        "fmla vV13B.4s, vU23B.4s, vW21B.4s\n"
        "fmul vV21B.4s, vU23B.4s, vW13B.4s\n"
        "fmla vV22B.4s, vU23B.4s, vW12B.4s\n"
        "fmla vV23B.4s, vU23B.4s, vW11B.4s\n"
        "ldr qU54B, [uptr4, u_col_stride3]\n"
        "fmla vV13B.4s, vU35B.4s, vW33B.4s\n"
        "fmla vV23B.4s, vU35B.4s, vW23B.4s\n"
        "fmul vV33B.4s, vU35B.4s, vW13B.4s\n"
        "ldr qU53B, [uptr4, u_col_stride2]\n"
        "fmla vV12B.4s, vU34B.4s, vW33B.4s\n"
        "fmla vV13B.4s, vU34B.4s, vW32B.4s\n"
        "fmla vV22B.4s, vU34B.4s, vW23B.4s\n"
        "fmla vV23B.4s, vU34B.4s, vW22B.4s\n"
        "fmul vV32B.4s, vU34B.4s, vW13B.4s\n"
        "fmla vV33B.4s, vU34B.4s, vW12B.4s\n"
        "ldr qU12B, [%x[uptr0], u_col_stride1]\n"
        "fmla vV11B.4s, vU33B.4s, vW33B.4s\n"
        "fmla vV12B.4s, vU33B.4s, vW32B.4s\n"
        "fmla vV13B.4s, vU33B.4s, vW31B.4s\n"
        "str qV13B, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21B.4s, vU33B.4s, vW23B.4s\n"
        "fmla vV22B.4s, vU33B.4s, vW22B.4s\n"
        "fmla vV23B.4s, vU33B.4s, vW21B.4s\n"
        "fmul vV31B.4s, vU33B.4s, vW13B.4s\n"
        "fmla vV32B.4s, vU33B.4s, vW12B.4s\n"
        "fmla vV33B.4s, vU33B.4s, vW11B.4s\n"
        "ldr qU22B, [uptr1, u_col_stride1]\n"
        "fmla vV23B.4s, vU45B.4s, vW33B.4s\n"
        "fmla vV33B.4s, vU45B.4s, vW23B.4s\n"
        "ldr qU32B, [uptr2, u_col_stride1]\n"
        "fmla vV22B.4s, vU44B.4s, vW33B.4s\n"
        "fmla vV23B.4s, vU44B.4s, vW32B.4s\n"
        "fmla vV32B.4s, vU44B.4s, vW23B.4s\n"
        "fmla vV33B.4s, vU44B.4s, vW22B.4s\n"
        "ldr qU42B, [uptr3, u_col_stride1]\n"
        "fmla vV21B.4s, vU43B.4s, vW33B.4s\n"
        "fmla vV22B.4s, vU43B.4s, vW32B.4s\n"
        "fmla vV23B.4s, vU43B.4s, vW31B.4s\n"
        "str qV23B, [vptr1, v_col_stride2]\n"
        "fmla vV31B.4s, vU43B.4s, vW23B.4s\n"
        "fmla vV32B.4s, vU43B.4s, vW22B.4s\n"
        "fmla vV33B.4s, vU43B.4s, vW21B.4s\n"
        "ldr qU52B, [uptr4, u_col_stride1]\n"
        "fmla vV33B.4s, vU55B.4s, vW33B.4s\n"
        "ldr qU11B, [%x[uptr0]], #0x10\n"
        "fmla vV32B.4s, vU54B.4s, vW33B.4s\n"
        "fmla vV33B.4s, vU54B.4s, vW32B.4s\n"
        "ldr qU21B, [uptr1], #0x10\n"
        "fmla vV31B.4s, vU53B.4s, vW33B.4s\n"
        "fmla vV32B.4s, vU53B.4s, vW32B.4s\n"
        "fmla vV33B.4s, vU53B.4s, vW31B.4s\n"
        "str qV33B, [vptr2, v_col_stride2]\n"
        "fmla vV11B.4s, vU12B.4s, vW12B.4s\n"
        "ldr qU31B, [uptr2], #0x10\n"
        "fmla vV12B.4s, vU12B.4s, vW11B.4s\n"
        "ldr qU41B, [uptr3], #0x10\n"
        "fmla vV11B.4s, vU22B.4s, vW22B.4s\n"
        "ldr qU51B, [uptr4], #0x10\n"
        "fmla vV12B.4s, vU22B.4s, vW21B.4s\n"
        "fmla vV21B.4s, vU22B.4s, vW12B.4s\n"
        "fmla vV22B.4s, vU22B.4s, vW11B.4s\n"
        "fmla vV11B.4s, vU32B.4s, vW32B.4s\n"
        "fmla vV12B.4s, vU32B.4s, vW31B.4s\n"
        "str qV12B, [%x[vptr0], v_col_stride1]\n"
        "fmla vV21B.4s, vU32B.4s, vW22B.4s\n"
        "fmla vV22B.4s, vU32B.4s, vW21B.4s\n"
        "fmla vV31B.4s, vU32B.4s, vW12B.4s\n"
        "fmla vV32B.4s, vU32B.4s, vW11B.4s\n"
        "fmla vV21B.4s, vU42B.4s, vW32B.4s\n"
        "fmla vV22B.4s, vU42B.4s, vW31B.4s\n"
        "str qV22B, [vptr1, v_col_stride1]\n"
        "fmla vV31B.4s, vU42B.4s, vW22B.4s\n"
        "fmla vV32B.4s, vU42B.4s, vW21B.4s\n"
        "fmla vV31B.4s, vU52B.4s, vW32B.4s\n"
        "subs %x[n_iters], %x[n_iters], #1\n"
        "fmla vV32B.4s, vU52B.4s, vW31B.4s\n"
        "str qV32B, [vptr2, v_col_stride1]\n"
        "fmla vV11B.4s, vU11B.4s, vW11B.4s\n"
        "fmla vV11B.4s, vU21B.4s, vW21B.4s\n"
        "fmla vV21B.4s, vU21B.4s, vW11B.4s\n"
        "fmla vV11B.4s, vU31B.4s, vW31B.4s\n"
        "str qV11B, [%x[vptr0]], #0x10\n"
        "fmla vV21B.4s, vU31B.4s, vW21B.4s\n"
        "fmla vV31B.4s, vU31B.4s, vW11B.4s\n"
        "fmla vV21B.4s, vU41B.4s, vW31B.4s\n"
        "str qV21B, [vptr1], #0x10\n"
        "fmla vV31B.4s, vU41B.4s, vW21B.4s\n"
        "fmla vV31B.4s, vU51B.4s, vW31B.4s\n"
        "str qV31B, [vptr2], #0x10\n"

        "b 4f\n"  // Branch to end of method

        "3:"  // Odd tail, finish off A
        "fmla vV13A.4s, vU14A.4s, vW12A.4s\n"
        "ldr qU45A, [uptr3, u_col_stride4]\n"
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV13A.4s, vU13A.4s, vW11A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV13A.4s, vU25A.4s, vW23A.4s\n"
        "fmul vV23A.4s, vU25A.4s, vW13A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmla vV13A.4s, vU24A.4s, vW22A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV23A.4s, vU24A.4s, vW12A.4s\n"
        "ldr qU55A, [uptr4, u_col_stride4]\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "fmla vV13A.4s, vU23A.4s, vW21A.4s\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "fmla vV23A.4s, vU23A.4s, vW11A.4s\n"
        "ldr qU54A, [uptr4, u_col_stride3]\n"
        "fmla vV13A.4s, vU35A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU35A.4s, vW23A.4s\n"
        "fmul vV33A.4s, vU35A.4s, vW13A.4s\n"
        "ldr qU53A, [uptr4, u_col_stride2]\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "fmla vV13A.4s, vU34A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "fmla vV23A.4s, vU34A.4s, vW22A.4s\n"
        "fmul vV32A.4s, vU34A.4s, vW13A.4s\n"
        "fmla vV33A.4s, vU34A.4s, vW12A.4s\n"
        "ldr qU12A, [%x[uptr0], u_col_stride1]\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "fmla vV13A.4s, vU33A.4s, vW31A.4s\n"
        "str qV13A, [%x[vptr0], v_col_stride2]\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "fmla vV23A.4s, vU33A.4s, vW21A.4s\n"
        "fmul vV31A.4s, vU33A.4s, vW13A.4s\n"
        "fmla vV32A.4s, vU33A.4s, vW12A.4s\n"
        "fmla vV33A.4s, vU33A.4s, vW11A.4s\n"
        "ldr qU22A, [uptr1, u_col_stride1]\n"
        "fmla vV23A.4s, vU45A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU45A.4s, vW23A.4s\n"
        "ldr qU32A, [uptr2, u_col_stride1]\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "fmla vV23A.4s, vU44A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU44A.4s, vW23A.4s\n"
        "fmla vV33A.4s, vU44A.4s, vW22A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "fmla vV23A.4s, vU43A.4s, vW31A.4s\n"
        "str qV23A, [vptr1, v_col_stride2]\n"
        "fmla vV31A.4s, vU43A.4s, vW23A.4s\n"
        "fmla vV32A.4s, vU43A.4s, vW22A.4s\n"
        "fmla vV33A.4s, vU43A.4s, vW21A.4s\n"
        "ldr qU52A, [uptr4, u_col_stride1]\n"
        "fmla vV33A.4s, vU55A.4s, vW33A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV32A.4s, vU54A.4s, vW33A.4s\n"
        "fmla vV33A.4s, vU54A.4s, vW32A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV31A.4s, vU53A.4s, vW33A.4s\n"
        "fmla vV32A.4s, vU53A.4s, vW32A.4s\n"
        "fmla vV33A.4s, vU53A.4s, vW31A.4s\n"
        "str qV33A, [vptr2, v_col_stride2]\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU51A, [uptr4], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], v_col_stride1]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU32A.4s, vW12A.4s\n"
        "fmla vV32A.4s, vU32A.4s, vW11A.4s\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, v_col_stride1]\n"
        "fmla vV31A.4s, vU42A.4s, vW22A.4s\n"
        "fmla vV32A.4s, vU42A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU52A.4s, vW32A.4s\n"
        "fmla vV32A.4s, vU52A.4s, vW31A.4s\n"
        "str qV32A, [vptr2, v_col_stride1]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU31A.4s, vW11A.4s\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"
        "fmla vV31A.4s, vU41A.4s, vW21A.4s\n"
        "fmla vV31A.4s, vU51A.4s, vW31A.4s\n"
        "str qV31A, [vptr2], #0x10\n"

        "4:"  // End of method
        ".unreq uptr1\n" ".unreq uptr2\n" ".unreq uptr3\n" ".unreq uptr4\n"
        ".unreq u_col_stride1\n" ".unreq u_col_stride2\n"
        ".unreq u_col_stride3\n" ".unreq u_col_stride4\n"
        ".unreq wptr1\n" ".unreq wptr2\n"
        ".unreq w_col_stride1\n" ".unreq w_col_stride2\n"
        ".unreq vptr1\n" ".unreq vptr2\n"
        ".unreq v_col_stride1\n" ".unreq v_col_stride2\n"

        ".unreq qU22B\n" ".unreq qW13B\n" ".unreq qW13A\n" ".unreq qU51B\n"
        ".unreq qU54B\n" ".unreq qU45A\n" ".unreq qU15A\n" ".unreq qU41B\n"
        ".unreq qU24B\n" ".unreq qU21A\n"
        ".unreq qV11B\n" ".unreq qU51A\n" ".unreq qU35A\n" ".unreq qU12A\n"
        ".unreq qU42B\n" ".unreq qU44B\n" ".unreq qU13B\n" ".unreq qW33A\n"
        ".unreq qV31B\n" ".unreq qV23A\n" ".unreq qU31A\n" ".unreq qU35B\n" ".unreq qU13A\n"
        ".unreq qV23B\n" ".unreq qU11A\n" ".unreq qU25A\n" ".unreq qU43A\n" ".unreq qU52B\n"
        ".unreq qU24A\n" ".unreq qU23B\n" ".unreq qV21A\n" ".unreq qV32B\n"
        ".unreq qV33B\n" ".unreq qW11A\n" ".unreq qU31B\n"
        ".unreq qW12B\n" ".unreq qU33A\n" ".unreq qU14A\n" ".unreq qU22A\n"
        ".unreq qU25B\n" ".unreq qU53B\n" ".unreq qU42A\n" ".unreq qU44A\n"
        ".unreq qU43B\n" ".unreq qW31A\n" ".unreq qU11B\n"
        ".unreq qW11B\n" ".unreq qW32A\n"
        ".unreq qU12B\n" ".unreq qU34B\n" ".unreq qW21A\n"
        ".unreq qU14B\n" ".unreq qV21B\n" ".unreq qW22A\n"
        ".unreq qW23B\n" ".unreq qW23A\n" ".unreq qU21B\n"
        ".unreq qU32B\n" ".unreq qU34A\n" ".unreq qU45B\n" ".unreq qV31A\n"
        ".unreq qW12A\n" ".unreq qU33B\n" ".unreq qU15B\n"
        ".unreq qW33B\n" ".unreq qU54A\n" ".unreq qU23A\n"
        ".unreq qW32B\n" ".unreq qV33A\n" ".unreq qW31B\n" ".unreq qV12A\n"
        ".unreq qV12B\n" ".unreq qU41A\n" ".unreq qU53A\n"
        ".unreq qV13A\n" ".unreq qU32A\n" ".unreq qW22B\n"
        ".unreq qV22B\n" ".unreq qU52A\n" ".unreq qV13B\n" ".unreq qV32A\n"
        ".unreq qU55A\n" ".unreq qU55B\n" ".unreq qV22A\n" ".unreq qW21B\n"
        ".unreq qV11A\n"
        ".unreq vU22B\n" ".unreq vW13B\n" ".unreq vW13A\n" ".unreq vU51B\n"
        ".unreq vU54B\n" ".unreq vU45A\n" ".unreq vU15A\n" ".unreq vU41B\n"
        ".unreq vU24B\n" ".unreq vU21A\n"
        ".unreq vV11B\n" ".unreq vU51A\n" ".unreq vU35A\n" ".unreq vU12A\n"
        ".unreq vU42B\n" ".unreq vU44B\n" ".unreq vU13B\n" ".unreq vW33A\n"
        ".unreq vV31B\n" ".unreq vV23A\n" ".unreq vU31A\n" ".unreq vU35B\n" ".unreq vU13A\n"
        ".unreq vV23B\n" ".unreq vU11A\n" ".unreq vU25A\n" ".unreq vU43A\n" ".unreq vU52B\n"
        ".unreq vU24A\n" ".unreq vU23B\n" ".unreq vV21A\n" ".unreq vV32B\n"
        ".unreq vV33B\n" ".unreq vW11A\n" ".unreq vU31B\n"
        ".unreq vW12B\n" ".unreq vU33A\n" ".unreq vU14A\n" ".unreq vU22A\n"
        ".unreq vU25B\n" ".unreq vU53B\n" ".unreq vU42A\n" ".unreq vU44A\n"
        ".unreq vU43B\n" ".unreq vW31A\n" ".unreq vU11B\n"
        ".unreq vW11B\n" ".unreq vW32A\n"
        ".unreq vU12B\n" ".unreq vU34B\n" ".unreq vW21A\n"
        ".unreq vU14B\n" ".unreq vV21B\n" ".unreq vW22A\n"
        ".unreq vW23B\n" ".unreq vW23A\n" ".unreq vU21B\n"
        ".unreq vU32B\n" ".unreq vU34A\n" ".unreq vU45B\n" ".unreq vV31A\n"
        ".unreq vW12A\n" ".unreq vU33B\n" ".unreq vU15B\n"
        ".unreq vW33B\n" ".unreq vU54A\n" ".unreq vU23A\n"
        ".unreq vW32B\n" ".unreq vV33A\n" ".unreq vW31B\n" ".unreq vV12A\n"
        ".unreq vV12B\n" ".unreq vU41A\n" ".unreq vU53A\n"
        ".unreq vV13A\n" ".unreq vU32A\n" ".unreq vW22B\n"
        ".unreq vV22B\n" ".unreq vU52A\n" ".unreq vV13B\n" ".unreq vV32A\n"
        ".unreq vU55A\n" ".unreq vU55B\n" ".unreq vV22A\n" ".unreq vW21B\n"
        ".unreq vV11A\n"
        : [uptr0] "+r" (uptr0), [wptr0] "+r" (wptr0), [vptr0] "+r" (vptr0),
          [n_iters] "+r" (n_iters)
        : [u_row_stride] "r" (in_row_stride * sizeof(float)),
          [u_col_stride] "r" (in_col_stride * sizeof(float)),
          [w_row_stride] "r" (weight_row_stride * sizeof(float)),
          [w_col_stride] "r" (weight_col_stride * sizeof(float)),
          [v_row_stride] "r" (out_row_stride * sizeof(float)),
          [v_col_stride] "r" (out_col_stride * sizeof(float)),
          [odd_tail] "r" (odd_tail)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0",
          "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
          "x12", "cc", "memory"
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
};

template <>
const Conv::TileFn Conv::tilefn_generic = ConvImpl::template process_tile<false>;

template class DepthwiseConvolution<3, 3, 3, 3, 1, 1, float, float>;
}  // namespace depthwise
