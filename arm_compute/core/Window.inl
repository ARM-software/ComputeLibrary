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
namespace arm_compute
{
inline Window::Window(const Window &src)
    : _dims()
{
    for(size_t i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        set(i, src[i]);
    }
}

inline constexpr const Window::Dimension &Window::operator[](size_t dimension) const
{
    // Precondition: dimension < Coordinates::num_max_dimensions
    return _dims.at(dimension);
}
inline void Window::set(size_t dimension, const Window::Dimension &dim)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);
    _dims[dimension] = dim;
}

inline Window Window::collapse_if_possible(const Window &full_window, size_t first) const
{
    bool   is_collapsable = false;
    Window collapsed;
    for(size_t d = 0; d < Coordinates::num_max_dimensions; ++d)
    {
        if(is_collapsable)
        {
            collapsed.set(first, Window::Dimension(collapsed[first].end() * _dims[d].start(), collapsed[first].end() * _dims[d].end()));
        }
        else
        {
            collapsed.set(d, _dims[d]);
        }

        if(is_collapsable || d == first) // Try to start collapsing from this dimension
        {
            // The _dims's dimension must match the full _dims dimension to be collapsable:
            is_collapsable = _dims[d].start() == 0 && _dims[d].start() == full_window[d].start()
                             && full_window[d].end() == _dims[d].end();
        }
        else
        {
            is_collapsable = false;
        }
    }
    return collapsed;
}

inline Window Window::collapse(const Window &full_window, size_t first) const
{
    Window collapsed = collapse_if_possible(full_window, first);
    // Make sure that the window has collapsed
    int end   = _dims[first].end();
    int start = 0;
    ARM_COMPUTE_UNUSED(start);
    for(size_t d = first + 1; d < Coordinates::num_max_dimensions; ++d)
    {
        start = end * _dims[d].start();
        end *= _dims[d].end();
    }
    ARM_COMPUTE_ERROR_ON((collapsed[first].end() != end) || (collapsed[first].start() != start));
    return collapsed;
}

inline void Window::shift(size_t dimension, int shift_value)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);
    Window::Dimension &d = _dims[dimension];
    d                    = Window::Dimension(d.start() + shift_value, d.end() + shift_value, d.step());
}

inline void Window::adjust(size_t dimension, int adjust_value, bool is_at_start)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);
    Window::Dimension &d = _dims[dimension];

    if(is_at_start)
    {
        d = Window::Dimension(d.start() + adjust_value, d.end(), d.step());
    }
    else
    {
        d = Window::Dimension(d.start(), d.end() + adjust_value, d.step());
    }
}

inline void Window::scale(size_t dimension, float scale_value)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);
    Window::Dimension &d           = _dims[dimension];
    const int          scaled_step = d.step() * scale_value;
    const int          scaled_end  = ceil_to_multiple(d.end() * scale_value, scaled_step);
    d                              = Window::Dimension(d.start() * scale_value, scaled_end, scaled_step);
}

inline void Window::set_dimension_step(size_t dimension, int step)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);
    _dims[dimension].set_step(step);
}

inline void Window::validate() const
{
    for(size_t i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(_dims[i].step() == 0);
        ARM_COMPUTE_ERROR_ON(_dims[i].end() < _dims[i].start());
        ARM_COMPUTE_ERROR_ON((_dims[i].end() - _dims[i].start()) % _dims[i].step());
    }
}

inline constexpr size_t Window::num_iterations(size_t dimension) const
{
    // Precondition: dimension < Coordinates::num_max_dimensions
    // Precondition: (end - start) % step == 0
    return (_dims.at(dimension).end() - _dims.at(dimension).start()) / _dims.at(dimension).step();
}

inline Window Window::split_window(size_t dimension, size_t id, size_t total) const
{
    ARM_COMPUTE_ERROR_ON(id >= total);
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);

    Window out;

    for(size_t d = 0; d < Coordinates::num_max_dimensions; ++d)
    {
        if(d == dimension)
        {
            int start          = _dims[d].start();
            int end            = _dims[d].end();
            int per_sub_window = (num_iterations(d) / total) * _dims[d].step();

            start += id * per_sub_window;

            if(id != total - 1)
            {
                end = start + per_sub_window;
            }

            out.set(d, Dimension(start, end, _dims[d].step()));
        }
        else
        {
            out.set(d, _dims[d]);
        }
    }

    return out;
}

template <unsigned int window_dimension>
inline bool Window::slide_window_slice(Window &slice) const
{
    for(unsigned int n = window_dimension; n < Coordinates::num_max_dimensions; ++n)
    {
        // Did we reach the end of this dimension?
        const int v = slice._dims[n].start() + 1;

        if(v < _dims[n].end())
        {
            // No: increment
            slice._dims[n] = Dimension(v, v + 1, 1);

            // Reset lower dimensions:
            for(unsigned int lower = window_dimension; lower < n; ++lower)
            {
                slice._dims[lower] = Dimension(_dims[lower].start(), _dims[lower].start() + 1, 1);
            }
            return true;
        }
    }

    // It was the last slice
    return false; // Iteration over
}

template <unsigned int window_dimension>
inline Window          Window::first_slice_window() const
{
    Window slice;

    std::copy_n(_dims.begin(), window_dimension, slice._dims.begin());

    //Initialise higher dimensions to be the first slice.
    for(unsigned int n = window_dimension; n < Coordinates::num_max_dimensions; ++n)
    {
        slice._dims[n] = Dimension(_dims[n].start(), _dims[n].start() + 1, 1);
    }

    return slice;
}

inline void Window::use_tensor_dimensions(const TensorShape &shape, size_t first_dimension)
{
    for(unsigned int n = first_dimension; n < shape.num_dimensions(); ++n)
    {
        set(n, Window::Dimension(0, std::max(shape[n], static_cast<size_t>(1))));
    }
}
}
