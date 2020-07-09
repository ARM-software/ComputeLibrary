/*
 * Copyright (c) 2019 Arm Limited.
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

#include "impl_dilated.hpp"

template class depthwise::DilatedDepthwiseConvolution<2, 2, 3, 3, 1, 1, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<2, 2, 3, 3, 2, 2, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 1, 1, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<4, 4, 3, 3, 2, 2, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<4, 4, 5, 5, 1, 1, float, float, float>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float, float, float>;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 1, 1, float16_t, float16_t, float16_t>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float16_t, float16_t, float16_t>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 1, 1, float16_t, float16_t, float16_t>;
template class depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float16_t, float16_t, float16_t>;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

