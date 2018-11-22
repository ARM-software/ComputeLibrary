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
#include "impl_u8_s32.hpp"

namespace depthwise
{
using Conv = DepthwiseConvolution<4, 4, 3, 3, 2, 2, uint8_t, int32_t>;
using ConvImpl = DepthwiseConvolutionImpl<4, 4, 3, 3, 2, 2, uint8_t, int32_t>;

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
        {
                ConvImpl::template process_tile<true, 0, 0, 6, 0, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 6, 0, 1, 0>,
                ConvImpl::template process_tile<true, 0, 0, 6, 0, 2, 0>,
                ConvImpl::template process_tile<true, 0, 0, 6, 0, 3, 0>,
        },
        {
                ConvImpl::template process_tile<true, 0, 0, 7, 0, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 7, 0, 1, 0>,
                ConvImpl::template process_tile<true, 0, 0, 7, 0, 2, 0>,
                ConvImpl::template process_tile<true, 0, 0, 7, 0, 3, 0>,
        },
        {
                ConvImpl::template process_tile<true, 0, 0, 8, 0, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 8, 0, 1, 0>,
                ConvImpl::template process_tile<true, 0, 0, 8, 0, 2, 0>,
                ConvImpl::template process_tile<true, 0, 0, 8, 0, 3, 0>,
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
        {
                ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 1>,
                ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 2>,
                ConvImpl::template process_tile<true, 0, 0, 0, 6, 0, 3>,
        },
        {
                ConvImpl::template process_tile<true, 0, 0, 0, 7, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 0, 7, 0, 1>,
                ConvImpl::template process_tile<true, 0, 0, 0, 7, 0, 2>,
                ConvImpl::template process_tile<true, 0, 0, 0, 7, 0, 3>,
        },
        {
                ConvImpl::template process_tile<true, 0, 0, 0, 8, 0, 0>,
                ConvImpl::template process_tile<true, 0, 0, 0, 8, 0, 1>,
                ConvImpl::template process_tile<true, 0, 0, 0, 8, 0, 2>,
                ConvImpl::template process_tile<true, 0, 0, 0, 8, 0, 3>,
        },
};

template <>
const Conv::TileFn Conv::tilefn_generic = ConvImpl::template process_tile<false>;

template class DepthwiseConvolution<4, 4, 3, 3, 2, 2, uint8_t, int32_t>;
}  // namespace depthwise
