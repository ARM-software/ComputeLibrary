/*
 * Copyright (c) 2017 ARM Limited.
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

#pragma once
#include <cassert>
#include "../utils.hpp"

#ifdef __aarch64__

template <>
inline void BlockedGemm<8, 12, float, float>(
  const float* const a, const float* const b, float *c,
  const int M, const int K, const int N,
  const int a_row_stride,
  const int b_row_stride,
  const int c_row_stride
) {
  const int M_BLOCK = 8;
  const int N_BLOCK = 12;

  const int m_blocks = iceildiv(M, M_BLOCK);
  const int n_blocks = iceildiv(N, N_BLOCK);

  // For each block of output rows
  for (int mblock = 0; mblock < m_blocks; mblock++) {
    // For each block of output columns
    for (int nblock = 0; nblock < n_blocks; nblock++) {
      const float *aptr = a + mblock*M_BLOCK*a_row_stride;
      const float *bptr = b + nblock*N_BLOCK;
      float *cptr = c + mblock*M_BLOCK*c_row_stride + nblock*N_BLOCK;
      int k = K;

      asm volatile (
          // Create an 8x12 block of accumulators
          " A_1 .req v27\n"
          "sA_1 .req s27\n"
          " A_2 .req v28\n"
          "sA_2 .req s28\n"
          " A_3 .req v29\n"
          "sA_3 .req s29\n"
          " A_4 .req v30\n"
          "sA_4 .req s30\n"

          " B_1 .req v24\n" " B_2 .req v25\n" " B_3 .req v26\n"
          "qB_1 .req q24\n" "qB_2 .req q25\n" "qB_3 .req q26\n"

          " C_11 .req  v0\n" " C_12 .req  v1\n" " C_13 .req  v2\n"
          " C_21 .req  v3\n" " C_22 .req  v4\n" " C_23 .req  v5\n"
          " C_31 .req  v6\n" " C_32 .req  v7\n" " C_33 .req  v8\n"
          " C_41 .req  v9\n" " C_42 .req v10\n" " C_43 .req v11\n"
          " C_51 .req v12\n" " C_52 .req v13\n" " C_53 .req v14\n"
          " C_61 .req v15\n" " C_62 .req v16\n" " C_63 .req v17\n"
          " C_71 .req v18\n" " C_72 .req v19\n" " C_73 .req v20\n"
          " C_81 .req v21\n" " C_82 .req v22\n" " C_83 .req v23\n"

          "qC_11 .req  q0\n" "qC_12 .req  q1\n" "qC_13 .req  q2\n"
          "qC_21 .req  q3\n" "qC_22 .req  q4\n" "qC_23 .req  q5\n"
          "qC_31 .req  q6\n" "qC_32 .req  q7\n" "qC_33 .req  q8\n"
          "qC_41 .req  q9\n" "qC_42 .req q10\n" "qC_43 .req q11\n"
          "qC_51 .req q12\n" "qC_52 .req q13\n" "qC_53 .req q14\n"
          "qC_61 .req q15\n" "qC_62 .req q16\n" "qC_63 .req q17\n"
          "qC_71 .req q18\n" "qC_72 .req q19\n" "qC_73 .req q20\n"
          "qC_81 .req q21\n" "qC_82 .req q22\n" "qC_83 .req q23\n"

          "aptr1 .req x17\n"
          "aptr2 .req x18\n"
          "aptr3 .req x19\n"
          "aptr4 .req x20\n"
          "aptr5 .req x21\n"
          "aptr6 .req x22\n"
          "aptr7 .req x23\n"

          // Initialise accumulators with 0
          // Initialise pointers
          "movi C_11.4s, #0\n"
          "add aptr1, %x[aptr], %x[a_row_stride]\n"
          "movi C_12.4s, #0\n"
          "add aptr2,    aptr1, %x[a_row_stride]\n"
          "movi C_13.4s, #0\n"
          "add aptr3,    aptr2, %x[a_row_stride]\n"
          "movi C_21.4s, #0\n"
          "add aptr4,    aptr3, %x[a_row_stride]\n"
          "movi C_22.4s, #0\n"
          "add aptr5,    aptr4, %x[a_row_stride]\n"
          "movi C_23.4s, #0\n"
          "add aptr6,    aptr5, %x[a_row_stride]\n"
          "movi C_31.4s, #0\n"
          "add aptr7,    aptr6, %x[a_row_stride]\n"
          "movi C_32.4s, #0\n"
          "ldr qB_1, [%x[bptr]]\n"
          "movi C_33.4s, #0\n"
          "ldr qB_2, [%x[bptr], #0x10]\n"
          "movi C_41.4s, #0\n"
          "prfm pldl1keep, [%x[bptr], #0x00]\n"
          "movi C_42.4s, #0\n"
          "prfm pldl1keep, [%x[bptr], #0x10]\n"
          "movi C_43.4s, #0\n"
          "prfm pldl1keep, [%x[bptr], #0x20]\n"
          "movi C_51.4s, #0\n"
          "prfm pldl1keep, [%x[aptr], #0x00]\n"
          "movi C_52.4s, #0\n"
          "prfm pldl1keep, [   aptr1, #0x00]\n"
          "movi C_53.4s, #0\n"
          "prfm pldl1keep, [   aptr2, #0x00]\n"
          "movi C_61.4s, #0\n"
          "prfm pldl1keep, [   aptr3, #0x00]\n"
          "movi C_62.4s, #0\n"
          "prfm pldl1keep, [   aptr4, #0x00]\n"
          "movi C_63.4s, #0\n"
          "prfm pldl1keep, [   aptr5, #0x00]\n"
          "movi C_71.4s, #0\n"
          "prfm pldl1keep, [   aptr6, #0x00]\n"
          "movi C_72.4s, #0\n"
          "prfm pldl1keep, [   aptr7, #0x00]\n"
          "movi C_73.4s, #0\n"
          "ldr sA_1, [%x[aptr]], #0x4\n"
          "movi C_81.4s, #0\n"
          "ldr sA_2, [   aptr1], #0x4\n"
          "movi C_82.4s, #0\n"
          "ldr sA_3, [   aptr2], #0x4\n"
          "movi C_83.4s, #0\n"
          "subs %x[k], %x[k], #1\n"
          "beq 2f\n"

          "1:"
            "fmla C_11.4s, B_1.4s, A_1.s[0]\n"
            "ldr qB_3, [%x[bptr], #0x20]\n"
            "fmla C_12.4s, B_2.4s, A_1.s[0]\n"
            "ldr sA_4, [   aptr3], #0x4\n"
            "fmla C_13.4s, B_3.4s, A_1.s[0]\n"
            "ldr sA_1, [   aptr4], #0x04\n"

            "fmla C_21.4s, B_1.4s, A_2.s[0]\n"
            "add %x[bptr], %x[bptr], %x[b_row_stride]\n"
            "fmla C_22.4s, B_2.4s, A_2.s[0]\n"
            "prfm pldl1keep, [   aptr3, #0x10]\n"
            "fmla C_23.4s, B_3.4s, A_2.s[0]\n"
            "ldr sA_2, [   aptr5], #0x04\n"

            "fmla C_31.4s, B_1.4s, A_3.s[0]\n"
            "prfm pldl1keep, [%x[bptr], #0x00]\n"
            "fmla C_32.4s, B_2.4s, A_3.s[0]\n"
            "prfm pldl1keep, [%x[bptr], #0x10]\n"
            "fmla C_33.4s, B_3.4s, A_3.s[0]\n"
            "ldr sA_3, [   aptr6], #0x04\n"

            "fmla C_41.4s, B_1.4s, A_4.s[0]\n"
            "prfm pldl1keep, [%x[bptr], #0x20]\n"
            "fmla C_42.4s, B_2.4s, A_4.s[0]\n"
            "prfm pldl1keep, [   aptr4, #0x10]\n"
            "fmla C_43.4s, B_3.4s, A_4.s[0]\n"
            "ldr sA_4, [   aptr7], #0x04\n"

            "fmla C_51.4s, B_1.4s, A_1.s[0]\n"
            "prfm pldl1keep, [   aptr5, #0x10]\n"
            "fmla C_52.4s, B_2.4s, A_1.s[0]\n"
            "prfm pldl1keep, [   aptr6, #0x10]\n"
            "fmla C_53.4s, B_3.4s, A_1.s[0]\n"
            "ldr sA_1, [%x[aptr]], #0x04\n"

            "fmla C_61.4s, B_1.4s, A_2.s[0]\n"
            "prfm pldl1keep, [   aptr7, #0x10]\n"
            "fmla C_62.4s, B_2.4s, A_2.s[0]\n"
            "subs %x[k], %x[k], #1\n"
            "fmla C_63.4s, B_3.4s, A_2.s[0]\n"
            "ldr sA_2, [   aptr1], #0x04\n"

            "fmla C_71.4s, B_1.4s, A_3.s[0]\n"
            "prfm pldl1keep, [%x[aptr], #0x10]\n"
            "fmla C_72.4s, B_2.4s, A_3.s[0]\n"
            "prfm pldl1keep, [   aptr1, #0x10]\n"
            "fmla C_73.4s, B_3.4s, A_3.s[0]\n"
            "ldr sA_3, [   aptr2], #0x04\n"

            "fmla C_81.4s, B_1.4s, A_4.s[0]\n"
            "prfm pldl1keep, [   aptr2, #0x10]\n"
            "fmla C_82.4s, B_2.4s, A_4.s[0]\n"
            "ldp qB_1, qB_2, [%x[bptr]]\n"
            "fmla C_83.4s, B_3.4s, A_4.s[0]\n"
            "bne 1b\n"

          "2:"
            "fmla C_11.4s, B_1.4s, A_1.s[0]\n"
            "ldr qB_3, [%x[bptr], #0x20]\n"
            "fmla C_12.4s, B_2.4s, A_1.s[0]\n"
            "stp qC_11, qC_12, [%x[cptr]]\n"
            "fmla C_13.4s, B_3.4s, A_1.s[0]\n"
            "str qC_13, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"
            "ldr sA_1, [   aptr4], #0x04\n"

            "fmla C_21.4s, B_1.4s, A_2.s[0]\n"
            "ldr sA_4, [   aptr3], #0x4\n"
            "fmla C_22.4s, B_2.4s, A_2.s[0]\n"
            "stp qC_21, qC_22, [%x[cptr]]\n"
            "fmla C_23.4s, B_3.4s, A_2.s[0]\n"
            "str qC_23, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"
            "ldr sA_2, [   aptr5], #0x04\n"

            "fmla C_31.4s, B_1.4s, A_3.s[0]\n"
            "fmla C_32.4s, B_2.4s, A_3.s[0]\n"
            "stp qC_31, qC_32, [%x[cptr]]\n"
            "fmla C_33.4s, B_3.4s, A_3.s[0]\n"
            "str qC_33, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"
            "ldr sA_3, [   aptr6], #0x04\n"

            "fmla C_41.4s, B_1.4s, A_4.s[0]\n"
            "fmla C_42.4s, B_2.4s, A_4.s[0]\n"
            "stp qC_41, qC_42, [%x[cptr]]\n"
            "fmla C_43.4s, B_3.4s, A_4.s[0]\n"
            "str qC_43, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"
            "ldr sA_4, [   aptr7], #0x04\n"

            "fmla C_51.4s, B_1.4s, A_1.s[0]\n"
            "fmla C_52.4s, B_2.4s, A_1.s[0]\n"
            "stp qC_51, qC_52, [%x[cptr]]\n"
            "fmla C_53.4s, B_3.4s, A_1.s[0]\n"
            "str qC_53, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"

            "fmla C_61.4s, B_1.4s, A_2.s[0]\n"
            "fmla C_62.4s, B_2.4s, A_2.s[0]\n"
            "stp qC_61, qC_62, [%x[cptr]]\n"
            "fmla C_63.4s, B_3.4s, A_2.s[0]\n"
            "str qC_63, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"

            "fmla C_71.4s, B_1.4s, A_3.s[0]\n"
            "fmla C_72.4s, B_2.4s, A_3.s[0]\n"
            "stp qC_71, qC_72, [%x[cptr]]\n"
            "fmla C_73.4s, B_3.4s, A_3.s[0]\n"
            "str qC_73, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"

            "fmla C_81.4s, B_1.4s, A_4.s[0]\n"
            "fmla C_82.4s, B_2.4s, A_4.s[0]\n"
            "stp qC_81, qC_82, [%x[cptr]]\n"
            "fmla C_83.4s, B_3.4s, A_4.s[0]\n"
            "str qC_83, [%x[cptr], #0x20]\n"
            "add %x[cptr], %x[cptr], %x[c_row_stride]\n"

          // Clear aliases
          ".unreq aptr1\n"
          ".unreq aptr2\n"
          ".unreq aptr3\n"
          ".unreq aptr4\n"
          ".unreq aptr5\n"
          ".unreq aptr6\n"
          ".unreq aptr7\n"

          ".unreq  A_1\n" ".unreq  A_2\n" ".unreq  A_3\n" ".unreq  A_4\n"
          ".unreq sA_1\n" ".unreq sA_2\n" ".unreq sA_3\n" ".unreq sA_4\n"

          ".unreq  B_1\n" ".unreq  B_2\n" ".unreq  B_3\n"
          ".unreq qB_1\n" ".unreq qB_2\n" ".unreq qB_3\n"

          ".unreq C_11\n" ".unreq C_12\n" ".unreq C_13\n"
          ".unreq C_21\n" ".unreq C_22\n" ".unreq C_23\n"
          ".unreq C_31\n" ".unreq C_32\n" ".unreq C_33\n"
          ".unreq C_41\n" ".unreq C_42\n" ".unreq C_43\n"
          ".unreq C_51\n" ".unreq C_52\n" ".unreq C_53\n"
          ".unreq C_61\n" ".unreq C_62\n" ".unreq C_63\n"
          ".unreq C_71\n" ".unreq C_72\n" ".unreq C_73\n"
          ".unreq C_81\n" ".unreq C_82\n" ".unreq C_83\n"

          ".unreq qC_11\n" ".unreq qC_12\n" ".unreq qC_13\n"
          ".unreq qC_21\n" ".unreq qC_22\n" ".unreq qC_23\n"
          ".unreq qC_31\n" ".unreq qC_32\n" ".unreq qC_33\n"
          ".unreq qC_41\n" ".unreq qC_42\n" ".unreq qC_43\n"
          ".unreq qC_51\n" ".unreq qC_52\n" ".unreq qC_53\n"
          ".unreq qC_61\n" ".unreq qC_62\n" ".unreq qC_63\n"
          ".unreq qC_71\n" ".unreq qC_72\n" ".unreq qC_73\n"
          ".unreq qC_81\n" ".unreq qC_82\n" ".unreq qC_83\n"
          : [aptr] "+r" (aptr),
            [bptr] "+r" (bptr),
            [cptr] "+r" (cptr),
            [k] "+r" (k)
          : [a_row_stride] "r" (a_row_stride * sizeof(float)),
            [b_row_stride] "r" (b_row_stride * sizeof(float)),
            [c_row_stride] "r" (c_row_stride * sizeof(float))
          : "cc", "memory",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
            "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
            "v29", "v30", "x17", "x18", "x19", "x20", "x21", "x22", "x23"
      );
    }
  }
}

/*****************************************************************************/
/* 4x16 blocked GEMM with specialised tails
 */
#include "a64_sgemm_4x16.hpp"

template <>
inline void BlockedGemm<4, 16, float, float>(
  const float* const a, const float* const b, float *c,
  const int M, const int K, const int N,
  const int a_row_stride,
  const int b_row_stride,
  const int c_row_stride
) {
  // Despatch based on tail of K
  switch (K % 4) {
    case 3:
      sgemm_4x16_impl<3>(
        a, b, c, M, K, N, a_row_stride, b_row_stride, c_row_stride
      );
      break;
    case 2:
      sgemm_4x16_impl<2>(
        a, b, c, M, K, N, a_row_stride, b_row_stride, c_row_stride
      );
      break;
    case 1:
      sgemm_4x16_impl<1>(
        a, b, c, M, K, N, a_row_stride, b_row_stride, c_row_stride
      );
      break;
    case 0:
      sgemm_4x16_impl<0>(
        a, b, c, M, K, N, a_row_stride, b_row_stride, c_row_stride
      );
      break;
    default:
      assert(false);
  }
}

#endif  // __aarch64__
