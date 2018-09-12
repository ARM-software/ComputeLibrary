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
using Conv = DepthwiseConvolution<2, 2, 3, 3, 1, 1, float, float>;
using ConvImpl = DepthwiseConvolutionImpl<2, 2, 3, 3, 1, 1, float, float>;

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
      "qW11B .req q0\n" "vW11B .req v0\n" "qW33A .req q1\n" "qU32B .req q1\n"
      "vW33A .req v1\n" "vU32B .req v1\n" "qU44B .req q2\n" "qW21A .req q2\n"
      "vU44B .req v2\n" "vW21A .req v2\n" "qU21B .req q3\n" "qU32A .req q3\n"
      "vU21B .req v3\n" "vU32A .req v3\n" "qU43A .req q4\n" "qV21B .req q4\n"
      "vU43A .req v4\n" "vV21B .req v4\n" "qU24A .req q5\n" "qU44A .req q5\n"
      "qU33B .req q5\n" "vU24A .req v5\n" "vU44A .req v5\n" "vU33B .req v5\n"
      "qU31A .req q6\n" "qV12B .req q6\n" "qU23A .req q6\n" "vU31A .req v6\n"
      "vV12B .req v6\n" "vU23A .req v6\n" "qW31B .req q7\n" "qV22A .req q7\n"
      "vW31B .req v7\n" "vV22A .req v7\n" "qV12A .req q8\n" "qW21B .req q8\n"
      "vV12A .req v8\n" "vW21B .req v8\n" "qU22B .req q9\n" "qU34A .req q9\n"
      "vU22B .req v9\n" "vU34A .req v9\n" "qU13B .req q10\n" "qU13A .req q10\n"
      "vU13B .req v10\n" "vU13A .req v10\n" "qU34B .req q11\n" "qU22A .req q11\n"
      "vU34B .req v11\n" "vU22A .req v11\n" "qU24B .req q12\n" "qU31B .req q12\n"
      "vU24B .req v12\n" "vU31B .req v12\n" "qW12B .req q13\n" "qW13A .req q13\n"
      "vW12B .req v13\n" "vW13A .req v13\n" "qV21A .req q14\n" "qV11B .req q14\n"
      "vV21A .req v14\n" "vV11B .req v14\n" "qW32A .req q15\n" "qW32B .req q15\n"
      "vW32A .req v15\n" "vW32B .req v15\n" "qW31A .req q16\n" "qV22B .req q16\n"
      "vW31A .req v16\n" "vV22B .req v16\n"
      "qW11A .req q17\n" "vW11A .req v17\n" "qW13B .req q18\n" "qU14A .req q18\n"
      "vW13B .req v18\n" "vU14A .req v18\n" "qU33A .req q19\n" "qW33B .req q19\n"
      "vU33A .req v19\n" "vW33B .req v19\n" "qW22A .req q20\n" "qU23B .req q20\n"
      "vW22A .req v20\n" "vU23B .req v20\n" "qU12A .req q21\n" "qU42A .req q21\n"
      "vU12A .req v21\n" "vU42A .req v21\n" "qU41A .req q22\n" "qU42B .req q22\n"
      "vU41A .req v22\n" "vU42B .req v22\n" "qW23A .req q23\n" "qW23B .req q23\n"
      "vW23A .req v23\n" "vW23B .req v23\n" "qU43B .req q24\n" "qU11A .req q24\n"
      "vU43B .req v24\n" "vU11A .req v24\n" "qU12B .req q25\n" "qW12A .req q25\n"
      "vU12B .req v25\n" "vW12A .req v25\n" "qU41B .req q26\n" "qV11A .req q26\n"
      "vU41B .req v26\n" "vV11A .req v26\n" "qW22B .req q27\n" "vW22B .req v27\n"
      "qU11B .req q28\n" "qU14B .req q28\n" "vU11B .req v28\n" "vU14B .req v28\n"
      "qU21A .req q29\n" "vU21A .req v29\n"

      "u_col_stride1 .req %x[u_col_stride]\n"
      "u_col_stride2 .req x0\n"
      "u_col_stride3 .req x1\n"
      "uptr1 .req x2\n"
      "uptr2 .req x3\n"
      "uptr3 .req x4\n"
      "wptr1 .req x5\n"
      "wptr2 .req x6\n"
      "vptr1 .req x7\n"
      "w_col_stride1 .req %x[w_col_stride]\n"
      "w_col_stride2 .req x8\n"

      // Prepare strides and pointers
      "add uptr1, %x[uptr0], %x[u_row_stride]\n"
      "add uptr2,    uptr1 , %x[u_row_stride]\n"
      "add uptr3,    uptr2 , %x[u_row_stride]\n"
      "add wptr1, %x[wptr0], %x[w_row_stride]\n"
      "add wptr2,    wptr1 , %x[w_row_stride]\n"
      "add vptr1, %x[vptr0], %x[v_row_stride]\n"
      "add u_col_stride2, %x[u_col_stride], %x[u_col_stride]\n"
      "add u_col_stride3,    u_col_stride2 , %x[u_col_stride]\n"
      "add w_col_stride2, %x[w_col_stride], %x[w_col_stride]\n"

      // Load in preparation for execution
      "ldr qU14A, [%x[uptr0], u_col_stride3]\n"
      "ldr qW13A, [%x[wptr0], w_col_stride2]\n"
      "ldr qU13A, [%x[uptr0], u_col_stride2]\n"
      "ldr qW12A, [%x[wptr0], w_col_stride1]\n"
      "ldr qU12A, [%x[uptr0], u_col_stride1]\n"
      "ldr qW11A, [%x[wptr0]], #0x10\n"
      "ldr qU24A, [uptr1, u_col_stride3]\n"
      "ldr qW23A, [wptr1, w_col_stride2]\n"
      "ldr qU23A, [uptr1, u_col_stride2]\n"
      "ldr qW22A, [wptr1, w_col_stride1]\n"
      "ldr qU22A, [uptr1, u_col_stride1]\n"
      "ldr qW21A, [wptr1], #0x10\n"
      "ldr qU34A, [uptr2, u_col_stride3]\n"
      "ldr qW33A, [wptr2, w_col_stride2]\n"
      "ldr qU33A, [uptr2, u_col_stride2]\n"
      "ldr qW32A, [wptr2, w_col_stride1]\n"
      "ldr qU32A, [uptr2, u_col_stride1]\n"
      "ldr qW31A, [wptr2], #0x10\n"
      "fmul vV12A.4s, vU14A.4s, vW13A.4s\n"
      "cbz %x[iters], 2f\n"  // Jump to tail if doing zero iterations of loop

      "1:"  // Main loop body
        // A part
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "ldr qU14B, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "ldr qW13B, [%x[wptr0], w_col_stride2]\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "ldr qU13B, [%x[uptr0], u_col_stride2]\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "ldr qW12B, [%x[wptr0], w_col_stride1]\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "ldr qU12B, [%x[uptr0], u_col_stride1]\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "ldr qW11B, [%x[wptr0]], #0x10\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "ldr qU24B, [uptr1, u_col_stride3]\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "ldr qW23B, [wptr1, w_col_stride2]\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], %x[v_col_stride]]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "ldr qU23B, [uptr1, u_col_stride2]\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "ldr qW22B, [wptr1, w_col_stride1]\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "ldr qU22B, [uptr1, u_col_stride1]\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "ldr qW21B, [wptr1], #0x10\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "ldr qU34B, [uptr2, u_col_stride3]\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "ldr qW33B, [wptr2, w_col_stride2]\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, %x[v_col_stride]]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "ldr qU33B, [uptr2, u_col_stride2]\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "ldr qW32B, [wptr2, w_col_stride1]\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "ldr qU32B, [uptr2, u_col_stride1]\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "ldr qW31B, [wptr2], #0x10\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"

        // B part
        "fmul vV12B.4s, vU14B.4s, vW13B.4s\n"
        "fmul vV11B.4s, vU13B.4s, vW13B.4s\n"
        "fmla vV12B.4s, vU13B.4s, vW12B.4s\n"
        "fmla vV11B.4s, vU12B.4s, vW12B.4s\n"
        "fmla vV12B.4s, vU12B.4s, vW11B.4s\n"
        "fmla vV12B.4s, vU24B.4s, vW23B.4s\n"
        "fmul vV22B.4s, vU24B.4s, vW13B.4s\n"
        "subs %x[iters], %x[iters], #1\n"
        "fmla vV11B.4s, vU23B.4s, vW23B.4s\n"
        "ldr qU44B, [uptr3, u_col_stride3]\n"
        "fmla vV12B.4s, vU23B.4s, vW22B.4s\n"
        "ldr qU43B, [uptr3, u_col_stride2]\n"
        "fmul vV21B.4s, vU23B.4s, vW13B.4s\n"
        "ldr qU42B, [uptr3, u_col_stride1]\n"
        "fmla vV22B.4s, vU23B.4s, vW12B.4s\n"
        "ldr qU11B, [%x[uptr0]], #0x10\n"
        "fmla vV11B.4s, vU22B.4s, vW22B.4s\n"
        "ldr qU21B, [uptr1], #0x10\n"
        "fmla vV12B.4s, vU22B.4s, vW21B.4s\n"
        "ldr qU31B, [uptr2], #0x10\n"
        "fmla vV21B.4s, vU22B.4s, vW12B.4s\n"
        "ldr qU41B, [uptr3], #0x10\n"
        "fmla vV22B.4s, vU22B.4s, vW11B.4s\n"
        "ldr qU14A, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12B.4s, vU34B.4s, vW33B.4s\n"
        "ldr qW13A, [%x[wptr0], w_col_stride2]\n"
        "fmla vV22B.4s, vU34B.4s, vW23B.4s\n"
        "ldr qU13A, [%x[uptr0], u_col_stride2]\n"
        "fmla vV11B.4s, vU33B.4s, vW33B.4s\n"
        "ldr qW12A, [%x[wptr0], w_col_stride1]\n"
        "fmla vV12B.4s, vU33B.4s, vW32B.4s\n"
        "ldr qU12A, [%x[uptr0], u_col_stride1]\n"
        "fmla vV21B.4s, vU33B.4s, vW23B.4s\n"
        "ldr qW11A, [%x[wptr0]], #0x10\n"
        "fmla vV22B.4s, vU33B.4s, vW22B.4s\n"
        "ldr qU24A, [uptr1, u_col_stride3]\n"
        "fmla vV11B.4s, vU32B.4s, vW32B.4s\n"
        "ldr qW23A, [wptr1, w_col_stride2]\n"
        "fmla vV12B.4s, vU32B.4s, vW31B.4s\n"
        "str qV12B, [%x[vptr0], %x[v_col_stride]]\n"
        "fmla vV21B.4s, vU32B.4s, vW22B.4s\n"
        "ldr qU23A, [uptr1, u_col_stride2]\n"
        "fmla vV22B.4s, vU32B.4s, vW21B.4s\n"
        "ldr qW22A, [wptr1, w_col_stride1]\n"
        "fmla vV22B.4s, vU44B.4s, vW33B.4s\n"
        "ldr qU22A, [uptr1, u_col_stride1]\n"
        "fmla vV21B.4s, vU43B.4s, vW33B.4s\n"
        "ldr qW21A, [wptr1], #0x10\n"
        "fmla vV22B.4s, vU43B.4s, vW32B.4s\n"
        "ldr qU34A, [uptr2, u_col_stride3]\n"
        "fmla vV21B.4s, vU42B.4s, vW32B.4s\n"
        "ldr qW33A, [wptr2, w_col_stride2]\n"
        "fmla vV22B.4s, vU42B.4s, vW31B.4s\n"
        "str qV22B, [vptr1, %x[v_col_stride]]\n"
        "fmla vV11B.4s, vU11B.4s, vW11B.4s\n"
        "ldr qU33A, [uptr2, u_col_stride2]\n"
        "fmla vV11B.4s, vU21B.4s, vW21B.4s\n"
        "ldr qW32A, [wptr2, w_col_stride1]\n"
        "fmla vV21B.4s, vU21B.4s, vW11B.4s\n"
        "ldr qU32A, [uptr2, u_col_stride1]\n"
        "fmla vV11B.4s, vU31B.4s, vW31B.4s\n"
        "str qV11B, [%x[vptr0]], #0x10\n"
        "fmla vV21B.4s, vU31B.4s, vW21B.4s\n"
        "ldr qW31A, [wptr2], #0x10\n"
        "fmla vV21B.4s, vU41B.4s, vW31B.4s\n"
        "str qV21B, [vptr1], #0x10\n"
        "fmul vV12A.4s, vU14A.4s, vW13A.4s\n"
        "bne 1b\n"  // Loop

      "2:"  // Branch destination for zero loops
        "cbnz %w[odd_tail], 4f\n"

      "3:"  // Even number of iterations
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "ldr qU14B, [%x[uptr0], u_col_stride3]\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "ldr qW13B, [%x[wptr0], w_col_stride2]\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "ldr qU13B, [%x[uptr0], u_col_stride2]\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "ldr qW12B, [%x[wptr0], w_col_stride1]\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "ldr qU12B, [%x[uptr0], u_col_stride1]\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "ldr qW11B, [%x[wptr0]], #0x10\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "ldr qU24B, [uptr1, u_col_stride3]\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "ldr qW23B, [wptr1, w_col_stride2]\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], %x[v_col_stride]]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "ldr qU23B, [uptr1, u_col_stride2]\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "ldr qW22B, [wptr1, w_col_stride1]\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "ldr qU22B, [uptr1, u_col_stride1]\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "ldr qW21B, [wptr1], #0x10\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "ldr qU34B, [uptr2, u_col_stride3]\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "ldr qW33B, [wptr2, w_col_stride2]\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, %x[v_col_stride]]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "ldr qU33B, [uptr2, u_col_stride2]\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "ldr qW32B, [wptr2, w_col_stride1]\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "ldr qU32B, [uptr2, u_col_stride1]\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "ldr qW31B, [wptr2], #0x10\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"

        "fmul vV12B.4s, vU14B.4s, vW13B.4s\n"
        "fmul vV11B.4s, vU13B.4s, vW13B.4s\n"
        "fmla vV12B.4s, vU13B.4s, vW12B.4s\n"
        "fmla vV11B.4s, vU12B.4s, vW12B.4s\n"
        "fmla vV12B.4s, vU12B.4s, vW11B.4s\n"
        "fmla vV12B.4s, vU24B.4s, vW23B.4s\n"
        "fmul vV22B.4s, vU24B.4s, vW13B.4s\n"
        "fmla vV11B.4s, vU23B.4s, vW23B.4s\n"
        "ldr qU44B, [uptr3, u_col_stride3]\n"
        "fmla vV12B.4s, vU23B.4s, vW22B.4s\n"
        "ldr qU43B, [uptr3, u_col_stride2]\n"
        "fmul vV21B.4s, vU23B.4s, vW13B.4s\n"
        "ldr qU42B, [uptr3, u_col_stride1]\n"
        "fmla vV22B.4s, vU23B.4s, vW12B.4s\n"
        "ldr qU11B, [%x[uptr0]], #0x10\n"
        "fmla vV11B.4s, vU22B.4s, vW22B.4s\n"
        "ldr qU21B, [uptr1], #0x10\n"
        "fmla vV12B.4s, vU22B.4s, vW21B.4s\n"
        "ldr qU31B, [uptr2], #0x10\n"
        "fmla vV21B.4s, vU22B.4s, vW12B.4s\n"
        "ldr qU41B, [uptr3], #0x10\n"
        "fmla vV22B.4s, vU22B.4s, vW11B.4s\n"
        "fmla vV12B.4s, vU34B.4s, vW33B.4s\n"
        "fmla vV22B.4s, vU34B.4s, vW23B.4s\n"
        "fmla vV11B.4s, vU33B.4s, vW33B.4s\n"
        "fmla vV12B.4s, vU33B.4s, vW32B.4s\n"
        "fmla vV21B.4s, vU33B.4s, vW23B.4s\n"
        "fmla vV22B.4s, vU33B.4s, vW22B.4s\n"
        "fmla vV11B.4s, vU32B.4s, vW32B.4s\n"
        "fmla vV12B.4s, vU32B.4s, vW31B.4s\n"
        "str qV12B, [%x[vptr0], %x[v_col_stride]]\n"
        "fmla vV21B.4s, vU32B.4s, vW22B.4s\n"
        "fmla vV22B.4s, vU32B.4s, vW21B.4s\n"
        "fmla vV22B.4s, vU44B.4s, vW33B.4s\n"
        "fmla vV21B.4s, vU43B.4s, vW33B.4s\n"
        "fmla vV22B.4s, vU43B.4s, vW32B.4s\n"
        "fmla vV21B.4s, vU42B.4s, vW32B.4s\n"
        "fmla vV22B.4s, vU42B.4s, vW31B.4s\n"
        "str qV22B, [vptr1, %x[v_col_stride]]\n"
        "fmla vV11B.4s, vU11B.4s, vW11B.4s\n"
        "fmla vV11B.4s, vU21B.4s, vW21B.4s\n"
        "fmla vV21B.4s, vU21B.4s, vW11B.4s\n"
        "fmla vV11B.4s, vU31B.4s, vW31B.4s\n"
        "str qV11B, [%x[vptr0]], #0x10\n"
        "fmla vV21B.4s, vU31B.4s, vW21B.4s\n"
        "fmla vV21B.4s, vU41B.4s, vW31B.4s\n"
        "str qV21B, [vptr1], #0x10\n"
        "b 5f\n"

      "4:"  // Odd number of iterations
        "fmul vV11A.4s, vU13A.4s, vW13A.4s\n"
        "fmla vV12A.4s, vU13A.4s, vW12A.4s\n"
        "fmla vV11A.4s, vU12A.4s, vW12A.4s\n"
        "fmla vV12A.4s, vU12A.4s, vW11A.4s\n"
        "fmla vV12A.4s, vU24A.4s, vW23A.4s\n"
        "fmul vV22A.4s, vU24A.4s, vW13A.4s\n"
        "fmla vV11A.4s, vU23A.4s, vW23A.4s\n"
        "ldr qU44A, [uptr3, u_col_stride3]\n"
        "fmla vV12A.4s, vU23A.4s, vW22A.4s\n"
        "ldr qU43A, [uptr3, u_col_stride2]\n"
        "fmul vV21A.4s, vU23A.4s, vW13A.4s\n"
        "ldr qU42A, [uptr3, u_col_stride1]\n"
        "fmla vV22A.4s, vU23A.4s, vW12A.4s\n"
        "ldr qU11A, [%x[uptr0]], #0x10\n"
        "fmla vV11A.4s, vU22A.4s, vW22A.4s\n"
        "ldr qU21A, [uptr1], #0x10\n"
        "fmla vV12A.4s, vU22A.4s, vW21A.4s\n"
        "ldr qU31A, [uptr2], #0x10\n"
        "fmla vV21A.4s, vU22A.4s, vW12A.4s\n"
        "ldr qU41A, [uptr3], #0x10\n"
        "fmla vV22A.4s, vU22A.4s, vW11A.4s\n"
        "fmla vV12A.4s, vU34A.4s, vW33A.4s\n"
        "fmla vV22A.4s, vU34A.4s, vW23A.4s\n"
        "fmla vV11A.4s, vU33A.4s, vW33A.4s\n"
        "fmla vV12A.4s, vU33A.4s, vW32A.4s\n"
        "fmla vV21A.4s, vU33A.4s, vW23A.4s\n"
        "fmla vV22A.4s, vU33A.4s, vW22A.4s\n"
        "fmla vV11A.4s, vU32A.4s, vW32A.4s\n"
        "fmla vV12A.4s, vU32A.4s, vW31A.4s\n"
        "str qV12A, [%x[vptr0], %x[v_col_stride]]\n"
        "fmla vV21A.4s, vU32A.4s, vW22A.4s\n"
        "fmla vV22A.4s, vU32A.4s, vW21A.4s\n"
        "fmla vV22A.4s, vU44A.4s, vW33A.4s\n"
        "fmla vV21A.4s, vU43A.4s, vW33A.4s\n"
        "fmla vV22A.4s, vU43A.4s, vW32A.4s\n"
        "fmla vV21A.4s, vU42A.4s, vW32A.4s\n"
        "fmla vV22A.4s, vU42A.4s, vW31A.4s\n"
        "str qV22A, [vptr1, %x[v_col_stride]]\n"
        "fmla vV11A.4s, vU11A.4s, vW11A.4s\n"
        "fmla vV11A.4s, vU21A.4s, vW21A.4s\n"
        "fmla vV21A.4s, vU21A.4s, vW11A.4s\n"
        "fmla vV11A.4s, vU31A.4s, vW31A.4s\n"
        "str qV11A, [%x[vptr0]], #0x10\n"
        "fmla vV21A.4s, vU31A.4s, vW21A.4s\n"
        "fmla vV21A.4s, vU41A.4s, vW31A.4s\n"
        "str qV21A, [vptr1], #0x10\n"

      "5:"  // End of method

      ".unreq qW11B\n" ".unreq qW33A\n" ".unreq qU32B\n"
      ".unreq qU44B\n" ".unreq qW21A\n" ".unreq qU21B\n" ".unreq qU32A\n"
      ".unreq qU43A\n" ".unreq qV21B\n"
      ".unreq qU24A\n" ".unreq qU44A\n" ".unreq qU33B\n"
      ".unreq qU31A\n" ".unreq qV12B\n" ".unreq qU23A\n"
      ".unreq qW31B\n" ".unreq qV22A\n" ".unreq qV12A\n" ".unreq qW21B\n"
      ".unreq qU22B\n" ".unreq qU34A\n" ".unreq qU13B\n" ".unreq qU13A\n"
      ".unreq qU34B\n" ".unreq qU22A\n" ".unreq qU24B\n" ".unreq qU31B\n"
      ".unreq qW12B\n" ".unreq qW13A\n" ".unreq qV21A\n" ".unreq qV11B\n"
      ".unreq qW32A\n" ".unreq qW32B\n" ".unreq qW31A\n" ".unreq qV22B\n"
      ".unreq qW11A\n" ".unreq qW13B\n" ".unreq qU14A\n"
      ".unreq qU33A\n" ".unreq qW33B\n" ".unreq qW22A\n" ".unreq qU23B\n"
      ".unreq qU12A\n" ".unreq qU42A\n" ".unreq qU41A\n" ".unreq qU42B\n"
      ".unreq qW23A\n" ".unreq qW23B\n" ".unreq qU43B\n" ".unreq qU11A\n"
      ".unreq qU12B\n" ".unreq qW12A\n" ".unreq qU41B\n" ".unreq qV11A\n"
      ".unreq qW22B\n" ".unreq qU11B\n" ".unreq qU14B\n" ".unreq qU21A\n"
      ".unreq vW11B\n" ".unreq vW33A\n" ".unreq vU32B\n"
      ".unreq vU44B\n" ".unreq vW21A\n" ".unreq vU21B\n" ".unreq vU32A\n"
      ".unreq vU43A\n" ".unreq vV21B\n"
      ".unreq vU24A\n" ".unreq vU44A\n" ".unreq vU33B\n"
      ".unreq vU31A\n" ".unreq vV12B\n" ".unreq vU23A\n"
      ".unreq vW31B\n" ".unreq vV22A\n" ".unreq vV12A\n" ".unreq vW21B\n"
      ".unreq vU22B\n" ".unreq vU34A\n" ".unreq vU13B\n" ".unreq vU13A\n"
      ".unreq vU34B\n" ".unreq vU22A\n" ".unreq vU24B\n" ".unreq vU31B\n"
      ".unreq vW12B\n" ".unreq vW13A\n" ".unreq vV21A\n" ".unreq vV11B\n"
      ".unreq vW32A\n" ".unreq vW32B\n" ".unreq vW31A\n" ".unreq vV22B\n"
      ".unreq vW11A\n" ".unreq vW13B\n" ".unreq vU14A\n"
      ".unreq vU33A\n" ".unreq vW33B\n" ".unreq vW22A\n" ".unreq vU23B\n"
      ".unreq vU12A\n" ".unreq vU42A\n" ".unreq vU41A\n" ".unreq vU42B\n"
      ".unreq vW23A\n" ".unreq vW23B\n" ".unreq vU43B\n" ".unreq vU11A\n"
      ".unreq vU12B\n" ".unreq vW12A\n" ".unreq vU41B\n" ".unreq vV11A\n"
      ".unreq vW22B\n" ".unreq vU11B\n" ".unreq vU14B\n" ".unreq vU21A\n"
      ".unreq u_col_stride1\n" ".unreq u_col_stride2\n"
      ".unreq u_col_stride3\n"
      ".unreq uptr1\n" ".unreq uptr2\n" ".unreq uptr3\n"
      ".unreq wptr1\n" ".unreq wptr2\n" ".unreq vptr1\n"
      ".unreq w_col_stride1\n" ".unreq w_col_stride2\n"

      : [uptr0] "+r" (uptr0), [vptr0] "+r" (vptr0), [wptr0] "+r" (wptr0),
        [iters] "+r" (n_iters)
      : [u_row_stride] "r" (in_row_stride * sizeof(float)),
        [u_col_stride] "r" (in_col_stride * sizeof(float)),
        [v_row_stride] "r" (out_row_stride * sizeof(float)),
        [v_col_stride] "r" (out_col_stride * sizeof(float)),
        [w_row_stride] "r" (weight_row_stride * sizeof(float)),
        [w_col_stride] "r" (weight_col_stride * sizeof(float)),
        [odd_tail] "r" (odd_tail)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "cc",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "memory"
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
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 1, 0, 1, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 2, 0, 1, 0>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 3, 0, 1, 0>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_right[n_in_pad_right_fns][n_out_pad_right_fns] = {
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 0, 0, 1>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 1, 0, 1>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 2, 0, 1>,
  },
  {
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 0>,
    ConvImpl::template process_tile<true, 0, 0, 0, 3, 0, 1>,
  },
};

template <>
const Conv::TileFn Conv::tilefn_generic = ConvImpl::template process_tile<false>;

template class DepthwiseConvolution<2, 2, 3, 3, 1, 1, float, float>;
}  // namespace depthwise
