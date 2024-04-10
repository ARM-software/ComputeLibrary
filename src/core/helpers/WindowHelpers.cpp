/*
* Copyright (c) 2020-2022 Arm Limited.
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
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
Window
calculate_max_window(const ValidRegion &valid_region, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if (!skip_border)
    {
        border_size = BorderSize(0);
    }

    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                      // Skip the border left of the image
                      anchor[0] + border_size.left,
                      // Skip the border right of the image
                      // Make sure the window width is a multiple of the step size
                      anchor[0] + border_size.left +
                          ceil_to_multiple(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) -
                                                           static_cast<int>(border_size.right)),
                                           steps[0]),
                      steps[0]));

    size_t n = 1;

    if (anchor.num_dimensions() > 1)
    {
        window.set(1,
                   Window::Dimension(
                       // Skip the border above the image
                       anchor[1] + border_size.top,
                       // Skip the border below the image
                       anchor[1] + border_size.top +
                           ceil_to_multiple(std::max(0, static_cast<int>(shape[1]) - static_cast<int>(border_size.top) -
                                                            static_cast<int>(border_size.bottom)),
                                            steps[1]),
                       steps[1]));

        ++n;
    }

    if (anchor.num_dimensions() > 2)
    {
        window.set(2, Window::Dimension(anchor[2], std::max<size_t>(1, shape[2]), steps[2]));

        ++n;
    }

    for (; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for (; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

Window calculate_max_window(const TensorShape &shape, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if (!skip_border)
    {
        border_size = BorderSize(0);
    }

    Window window;

    window.set(0, Window::Dimension(
                      // Skip the border left of the image
                      border_size.left,
                      // Skip the border right of the image
                      // Make sure the window width is a multiple of the step size
                      border_size.left +
                          ceil_to_multiple(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) -
                                                           static_cast<int>(border_size.right)),
                                           steps[0]),
                      steps[0]));

    size_t n = 1;

    if (shape.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                          // Skip the border above the image
                          border_size.top,
                          // Skip the border below the image
                          border_size.top + ceil_to_multiple(std::max(0, static_cast<int>(shape[1]) -
                                                                             static_cast<int>(border_size.top) -
                                                                             static_cast<int>(border_size.bottom)),
                                                             steps[1]),
                          steps[1]));

        ++n;
    }

    if (shape.num_dimensions() > 2)
    {
        window.set(2, Window::Dimension(0, std::max<size_t>(1, shape[2]), steps[2]));

        ++n;
    }

    for (; n < shape.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(0, std::max<size_t>(1, shape[n])));
    }

    for (; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

Window calculate_max_enlarged_window(const ValidRegion &valid_region, const Steps &steps, BorderSize border_size)
{
    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                      // move the anchor to the start from the border
                      anchor[0] - border_size.left,
                      // move the anchor to include the right end border
                      // Make sure the window width is a multiple of the step size
                      anchor[0] - border_size.left +
                          ceil_to_multiple(shape[0] + border_size.left + border_size.right, steps[0]),
                      steps[0]));

    size_t n = 1;

    if (anchor.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                          // Include the border above the image
                          anchor[1] - border_size.top,
                          // Include the border below the image
                          anchor[1] - border_size.top +
                              ceil_to_multiple(shape[1] + border_size.top + border_size.bottom, steps[1]),
                          steps[1]));

        ++n;
    }

    if (anchor.num_dimensions() > 2)
    {
        window.set(2, Window::Dimension(0, std::max<size_t>(1, shape[n]), steps[2]));

        ++n;
    }

    for (; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for (; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

Window calculate_max_window_horizontal(const ValidRegion &valid_region,
                                       const Steps       &steps,
                                       bool               skip_border,
                                       BorderSize         border_size)
{
    if (skip_border)
    {
        border_size.top    = 0;
        border_size.bottom = 0;
    }
    else
    {
        border_size.left  = 0;
        border_size.right = 0;
    }

    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                      // Skip the border left of the image
                      anchor[0] + border_size.left,
                      // Skip the border right of the image
                      // Make sure the window width is a multiple of the step size
                      anchor[0] + border_size.left +
                          ceil_to_multiple(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) -
                                                           static_cast<int>(border_size.right)),
                                           steps[0]),
                      steps[0]));

    size_t n = 1;

    if (anchor.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                          // Skip the border above the image
                          anchor[1] - border_size.top,
                          // Skip the border below the image
                          anchor[1] + shape[1] + border_size.bottom, 1));

        ++n;
    }

    for (; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for (; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

std::pair<Window, size_t> calculate_squashed_or_max_window(const ITensorInfo &src0, const ITensorInfo &src1)
{
    const auto &shape0         = src0.tensor_shape();
    const auto &shape1         = src1.tensor_shape();
    const auto &strides0       = src0.strides_in_bytes();
    const auto &strides1       = src1.strides_in_bytes();
    const auto  num_dimensions = std::max(src0.num_dimensions(), src1.num_dimensions());

    Window win;
    size_t split_dimension = Window::DimY;
    size_t dim             = 0;

    size_t squashed_bytes = src0.element_size();

    // Try to squash the low dimensions together.
    for (; dim < num_dimensions; ++dim)
    {
        if (shape0[dim] != shape1[dim] || strides0[dim] != squashed_bytes || strides1[dim] != squashed_bytes)
        {
            break;
        }

        squashed_bytes *= shape0[dim];
    }

    if (dim == num_dimensions)
    {
        auto squashed_elements = squashed_bytes / src0.element_size();

        split_dimension = Window::DimX;

        // The input tensors can be interpreted as 1D array.
        win.set(0, Window::Dimension(0, squashed_elements, 1));

        for (dim = 1; dim < Coordinates::num_max_dimensions; ++dim)
        {
            win.set(dim, Window::Dimension(0, 1, 1));
        }
    }
    else
    {
        // Generates the max window.
        for (dim = 0; dim < Coordinates::num_max_dimensions; ++dim)
        {
            win.set(dim, Window::Dimension(0, std::max(shape0[dim], shape1[dim]), 1));
        }
    }

    return std::make_pair(win, split_dimension);
}

std::pair<Window, size_t> calculate_squashed_or_max_window(const ITensorInfo &src)
{
    const auto &shape          = src.tensor_shape();
    const auto &strides        = src.strides_in_bytes();
    const auto  num_dimensions = src.num_dimensions();

    Window win;
    size_t split_dimension = Window::DimY;
    size_t dim             = 0;
    size_t squashed_bytes  = src.element_size();

    // Try to squash the low dimensions together.
    for (; dim < num_dimensions; ++dim)
    {
        if (strides[dim] != squashed_bytes)
        {
            break;
        }
        squashed_bytes *= shape[dim];
    }
    if (dim == num_dimensions)
    {
        const auto squashed_elements = squashed_bytes / src.element_size();
        split_dimension              = Window::DimX;
        // The input tensor can be interpreted as 1D array.
        win.set(0, Window::Dimension(0, squashed_elements, 1));
        for (dim = 1; dim < Coordinates::num_max_dimensions; ++dim)
        {
            win.set(dim, Window::Dimension(0, 1, 1));
        }
    }
    else
    {
        // Generate the max window.
        for (dim = 0; dim < Coordinates::num_max_dimensions; ++dim)
        {
            win.set(dim, Window::Dimension(0, shape[dim], 1));
        }
    }
    return std::make_pair(win, split_dimension);
}

} // namespace arm_compute
