/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "mergeresults.hpp"
#include "transform.hpp"

namespace arm_gemm {

/*
 * Define "standard" transforms for the blocked GEMMs for SVE.
 *
 * This assumes that A is interleaved 'height' ways, B is interleaved
 * 'width'xVL ways and transposed, and that the merge needs to work in
 * 'height' x 'width'xVL blocks.
 *
 * The optional 'block' parameter is for kernels using dot-product type
 * instructions like UDOT and SDOT.
 */
template<typename TOperand, typename TResult, unsigned int height, unsigned int width_vectors, unsigned int block=1, unsigned int mmla=1>
class StdTransformsSVE
{
public:
    template<typename TIn>
    void PrepareA(TOperand *out, const TIn *in, const int stride, const int y0,
                  const int ymax, const int k0, const int kmax, bool transposed) {
        if (transposed) {
            Transform<height, block,  true>(out, in, stride, y0, ymax, k0, kmax);
        } else {
            Transform<height, block, false>(out, in, stride, y0, ymax, k0, kmax);
        }
    }

    template<typename TIn>
    void PrepareB(TOperand *out, const TIn *in, const int stride, const int x0,
                  const int xmax, const int k0, const int kmax, bool transposed) {
        if (transposed) {
            Transform<width_vectors, block, false, true>(out, in, stride, x0, xmax, k0, kmax);
        } else {
            Transform<width_vectors, block,  true, true>(out, in, stride, x0, xmax, k0, kmax);
        }
    }

    template<typename TOut>
    void Merge(TOut *out, const TResult *in, int stride, int y0, int ymax, int x0, int xmax, const TOut *bias, const Activation act, bool append) {
        MergeResults<width_vectors / mmla, height, true>(out, in, stride, y0, ymax, x0, xmax, bias, act, append);
    }
};

} // namespace arm_gemm
