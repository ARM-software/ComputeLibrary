/*
 * Copyright (c) 2020-2021 Arm Limited.
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

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Window.h"

#include "ndrange.hpp"

#include <cassert>

/* This file contains mapping between integral types used in arm_compute and arm_gemm
 * These two codebases both require a degree of separation for the sake of modularity
 * so maintain their own types which represent similar information.
 */

namespace arm_gemm
{
//we want to unify the maximum number of dimensions used beween arm_gemm and arm compute library
constexpr std::size_t ndrange_max =
    arm_compute::Dimensions<unsigned int>::num_max_dimensions;

using ndrange_t = NDRange<ndrange_max>;
using ndcoord_t = NDCoordinate<ndrange_max>;

/* Converts an `arm_gemm::ndrange_t` to a `arm_compute::Window`
 *
 * As `NDRange<T>` does not not encode start positions, we specify
 * the start to be zero in the produced `arm_compute::Window`
 *
 * @param [ndr] the `arm_gemm::ndrange_t` we wish to convert into a `arm_compute::Window`
 * @returns an `arm_compute::Window` representing the same dimensional ranges as `ndr`
 */
inline arm_compute::Window to_window(const ndrange_t &ndr)
{
    arm_compute::Window win;

    for(unsigned int i = 0; i != ndrange_max; ++i)
    {
        //populate the window with the dimensions of the NDRange
        win.set(i, arm_compute::Window::Dimension(0, ndr.get_size(i)));
    }

    return win;
}

/*
 * Converts an `arm_gemm::ndcoord_t` to a `arm_compute::Window`
 *
 * @param [ndc] the `arm_gemm::ndcoord_t` we wish to convert into a `arm_compute::Window`
 * @returns an `arm_compute::Window` representing the same dimensional ranges as `ndc`
 */
inline arm_compute::Window to_window(const ndcoord_t &ndc)
{
    arm_compute::Window win;

    for(unsigned int i = 0; i != ndrange_max; ++i)
    {
        const auto start = ndc.get_position(i);
        const auto size  = ndc.get_size(i);
        const auto stop  = start + size;

        //populate the window with the dimensions of the NDRange
        win.set(i, arm_compute::Window::Dimension(start, stop));
    }

    return win;
}

/** Convert an `arm_compute::Window` to an `arm_gemm::NDRange` of the same max dimensions
 *
 * It should be noted that `arm_compute::Window` specifies a `start()` and an `end()`
 * where as `arm_gemm::ndrange_t` only has a size, as a result we store the delta between the range
 *
 * @param [win] the `arm_compute::Window` we want to convert to `arm_gemm::ndrange_t`
 * @return the resultant ndrange_t
 */
inline ndrange_t to_ndrange(const arm_compute::Window &win)
{
    return
    {
        static_cast<unsigned int>(win[0].end() - win[0].start()),
        static_cast<unsigned int>(win[1].end() - win[1].start()),
        static_cast<unsigned int>(win[2].end() - win[2].start()),
        static_cast<unsigned int>(win[3].end() - win[3].start()),
        static_cast<unsigned int>(win[4].end() - win[4].start()),
        static_cast<unsigned int>(win[5].end() - win[5].start())
    };
}

/** Convert an `arm_compute::Window` to an `arm_gemm::NDCoord` of the same max dimensions
 *
 * @param [win] the `arm_compute::Window` we want to convert to `arm_gemm::ndcoord_t`
 * @return the resultant ndcoord_t
 */
inline ndcoord_t to_ndcoord(const arm_compute::Window &win)
{
    return
    {
        { static_cast<unsigned int>(win[0].start()), static_cast<unsigned int>(win[0].end() - win[0].start()) },
        { static_cast<unsigned int>(win[1].start()), static_cast<unsigned int>(win[1].end() - win[1].start()) },
        { static_cast<unsigned int>(win[2].start()), static_cast<unsigned int>(win[2].end() - win[2].start()) },
        { static_cast<unsigned int>(win[3].start()), static_cast<unsigned int>(win[3].end() - win[3].start()) },
        { static_cast<unsigned int>(win[4].start()), static_cast<unsigned int>(win[4].end() - win[4].start()) },
        { static_cast<unsigned int>(win[5].start()), static_cast<unsigned int>(win[5].end() - win[5].start()) }
    };
}

} //namespace arm_gemm
