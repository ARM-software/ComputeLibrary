/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/Helpers.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/IKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <cstdint>

using namespace arm_compute;

Window arm_compute::calculate_max_window(const TensorInfo &info, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if(!skip_border)
    {
        border_size = BorderSize(0);
    }

    const Coordinates &anchor = info.valid_region().anchor;
    const TensorShape &shape  = info.valid_region().shape;

    Window window;

    window.set(0, Window::Dimension(
                   // Skip the border left of the image
                   anchor[0] + border_size.left,
                   // Skip the border right of the image
                   // Make sure the window width is a multiple of the step size
                   anchor[0] + border_size.left + ceil_to_multiple(shape[0] - border_size.left - border_size.right, steps[0]),
                   steps[0]));

    size_t             n            = 1;
    const TensorShape &tensor_shape = info.tensor_shape();

    if(tensor_shape.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                       // Skip the border above the image
                       anchor[1] + border_size.top,
                       // Skip the border below the image
                       anchor[1] + border_size.top + ceil_to_multiple(shape[1] - border_size.top - border_size.bottom, steps[1]),
                       steps[1]));

        ++n;
    }

    for(; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, std::max<size_t>(1, tensor_shape[n])));
    }

    return window;
}

Window arm_compute::calculate_max_window_horizontal(const TensorInfo &info, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if(skip_border)
    {
        border_size.top    = 0;
        border_size.bottom = 0;
    }
    else
    {
        border_size.left  = 0;
        border_size.right = 0;
    }

    const Coordinates &anchor = info.valid_region().anchor;
    const TensorShape &shape  = info.valid_region().shape;

    Window window;

    window.set(0, Window::Dimension(
                   // Skip the border left of the image
                   anchor[0] + border_size.left,
                   // Skip the border right of the image
                   // Make sure the window width is a multiple of the step size
                   anchor[0] + border_size.left + ceil_to_multiple(shape[0] - border_size.left - border_size.right, steps[0]),
                   steps[0]));

    size_t             n            = 1;
    const TensorShape &tensor_shape = info.tensor_shape();

    if(tensor_shape.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                       // Skip the border above the image
                       anchor[1] - border_size.top,
                       // Skip the border below the image
                       anchor[1] + shape[1] + border_size.bottom,
                       1));

        ++n;
    }

    for(; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, std::max<size_t>(1, tensor_shape[n])));
    }

    return window;
}
