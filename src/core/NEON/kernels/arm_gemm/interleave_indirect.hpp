/*
 * Copyright (c) 2020 Arm Limited.
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

#include <cstddef>
#include <cstdint>

#include "convolution_parameters.hpp"
#include "convolver.hpp"
#include "utils.hpp"

namespace arm_gemm {

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void IndirectInterleave(TOut *out, const TIn * const * const *ptr, unsigned int stringlen, unsigned int rounded_stringlen, unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool, int32_t);

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void ConvolutionInterleave(TOut *out, const TIn *in, size_t in_stride, const convolver<TIn> &conv, const unsigned int rounded_stringlen, const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool, int32_t);

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void Interleave(TOut *out, const TIn *in, size_t in_stride, const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool, int32_t);

} // namespace arm_gemm
