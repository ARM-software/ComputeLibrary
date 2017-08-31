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

template <unsigned int IntBy, typename TIn, typename TOut>
struct TransposeInterleaveCommon {
  // Override the moveblock_1xY methods to improve performance
  static inline void moveblock_1x1(const TIn *&in0, TOut *out) {
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in0++);
    }
  }

  static inline void moveblock_1x2(const TIn *&in0, const TIn *&in1, TOut *out) {
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in0++);
    }
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in1++);
    }
  }

  static inline void moveblock_1x4(const TIn *&in0, const TIn *&in1, const TIn *&in2, const TIn *&in3, TOut *out) {
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in0++);
    }
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in1++);
    }
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in2++);
    }
    for (unsigned int i = 0; i < IntBy; i++) {
      *out++ = static_cast<TOut>(*in3++);
    }
  }

  static inline void Transform(TOut *out, const TIn *in, const int stride, const int x0, const int xmax, const int k0, const int kmax) {
    const auto ldin = stride;

    TOut *outarray = out;
    const TIn *inarray = in;
    TOut *outptr_base = outarray;
    const TIn *inptr_base = inarray + x0 + (k0 * ldin);
    int ldout = (kmax - k0) * IntBy;

    int k=(kmax-k0);
    for ( ; k>3; k-=4) {
        TOut *outptr = outptr_base;
        const TIn *inptr = inptr_base;
        const TIn *inptr1 = inptr + ldin;
        const TIn *inptr2 = inptr1 + ldin;
        const TIn *inptr3 = inptr2 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);
        prefetch_3x(inptr3);

        outptr_base += IntBy * 4;
        inptr_base += ldin * 4;

        for (int x = (xmax-x0) / IntBy; x > 0 ; x--) {
            moveblock_1x4(inptr, inptr1, inptr2, inptr3, outptr);
            outptr += ldout;
        }
    }

    if (k) {
        TOut *outptr = outptr_base;
        const TIn *inptr = inptr_base;
        const TIn *inptr1 = inptr + ldin;
        const TIn *inptr2 = inptr1 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);

        for (int x = (xmax-x0) / IntBy; x > 0 ; x--) {
            switch(k) {
                case 3:
                    moveblock_1x2(inptr, inptr1, outptr);
                    moveblock_1x1(inptr2, outptr + IntBy * 2);
                    break;

                case 2:
                    moveblock_1x2(inptr, inptr1, outptr);
                    break;

                case 1:
                    moveblock_1x1(inptr, outptr);
                    break;
                default:
                    break;
            }

            outptr  += ldout;
        }
    }

    // Cope with ragged X cases
    const unsigned int overflow = (xmax - x0) % IntBy;
    if (overflow) {
        const TIn *inptr_base = inarray + (xmax - overflow) + (k0 * ldin);
        TOut *outptr = outarray + ((xmax - x0) / IntBy) * ldout;

        for (int k=(kmax-k0); k>0; k--) {
            const TIn *inptr = inptr_base;
            inptr_base += ldin;

            for (unsigned int x=0; x < IntBy; x++) {
                TOut val = (x < overflow) ? static_cast<TOut>(*inptr++) : static_cast<TOut>(0);
                *outptr++ = val;
            }
        }
    }
}
};
