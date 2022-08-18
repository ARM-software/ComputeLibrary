/*
 * Copyright (c) 2016-2020, 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_WINDOW_H
#define ARM_COMPUTE_WINDOW_H

#include <algorithm>
#include <array>
#include <cstddef>

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Utils.h"

namespace arm_compute
{
/** Describe a multidimensional execution window. */
class Window
{
public:
    /** Alias for dimension 0 also known as X dimension */
    static constexpr size_t DimX = 0;
    /** Alias for dimension 1 also known as Y dimension */
    static constexpr size_t DimY = 1;
    /** Alias for dimension 2 also known as Z dimension */
    static constexpr size_t DimZ = 2;
    /** Alias for dimension 3 also known as W dimension */
    static constexpr size_t DimW = 3;
    /** Alias for dimension 4 also known as V dimension */
    static constexpr size_t DimV = 4;

    /** Default constructor: create a window containing a single element. */
    constexpr Window()
        : _dims(), _is_broadcasted(utility::generate_array<bool, Coordinates::num_max_dimensions, false>::value)
    {
    }
    /** Copy constructor
     *
     * @param[in] src Copy the values from src to a new object
     */
    Window(const Window &src);
    /** Copy assignment operator
     *
     * @param[in] rhs Copy the values from rhs to the current object
     *
     * @return Reference to the updated object
     */
    Window &operator=(const Window &rhs);

    /** Describe one of the image's dimensions with a start, end and step.
     *
     * Iteration through the elements of the dimension is done like this:
     * for(int v = start(); v < end(); v += step())
     * {
     *   ...
     * }
     */
    class Dimension
    {
    public:
        /** Constructor, by default creates a dimension of 1.
         *
         * @param[in] start Start of the dimension
         * @param[in] end   End of the dimension
         * @param[in] step  Step between two elements of the dimension when iterating.
         *
         */
        constexpr Dimension(int start = 0, int end = 1, int step = 1)
            : _start(start), _end(end), _step(step)
        {
        }
        Dimension(const Dimension &d) = default;
        /** Default assignment operator to allow dimensions to be copied */
        Dimension &operator=(const Dimension &d) = default;
        /** Return the start of the dimension */
        constexpr int start() const
        {
            return _start;
        }
        /** Return the end of the dimension */
        constexpr int end() const
        {
            return _end;
        }
        /** Return the step of the dimension */
        constexpr int step() const
        {
            return _step;
        }
        /** Set the dimension's step
         *
         * @param[in] step The new step
         */
        void set_step(int step)
        {
            _step = step;
        }
        /** Set the dimension's end
         *
         * @param[in] end The new end
         */
        void set_end(int end)
        {
            _end = end;
        }
        /** Check whether two Dimensions are equal.
         *
         * @param[in] lhs LHS Dimensions
         * @param[in] rhs RHS Dimensions
         *
         * @return True if the Dimensions are the same.
         */
        friend bool operator==(const Dimension &lhs, const Dimension &rhs)
        {
            return (lhs._start == rhs._start) && (lhs._end == rhs._end) && (lhs._step == rhs._step);
        }

    private:
        int _start; /**< Start of the dimension */
        int _end;   /**< End of the dimension */
        int _step;
    };

    /** Read only access to a given dimension of the window
     *
     * @note Precondition: dimension < Coordinates::num_max_dimensions
     *
     * @param[in] dimension The dimension to access
     *
     * @return The requested dimension
     */
    constexpr const Dimension &operator[](size_t dimension) const;

    /** Alias to access the first dimension of the window
     *
     * @return First dimension of the window
     */
    constexpr const Dimension &x() const
    {
        return _dims.at(Window::DimX);
    }

    /** Alias to access the second dimension of the window
     *
     * @return Second dimension of the window
     */
    constexpr const Dimension &y() const
    {
        return _dims.at(Window::DimY);
    }

    /** Alias to access the third dimension of the window
     *
     * @return Third dimension of the window
     */
    constexpr const Dimension &z() const
    {
        return _dims.at(Window::DimZ);
    }

    /** Set the values of a given dimension
     *
     * @param[in] dimension The dimension to set
     * @param[in] dim       The values to set the dimension to
     */
    void set(size_t dimension, const Dimension &dim);

    /** Set the dimension as broadcasted dimension
     *
     * @param[in] dimension The dimension to set
     */
    void set_broadcasted(size_t dimension);

    /** Return whether a dimension has been broadcasted
     *
     * @param[in] dimension The requested dimension
     *
     * @return true if the dimension has been broadcasted
     */
    bool is_broadcasted(size_t dimension) const;

    /** Use the tensor's dimensions to fill the window dimensions.
     *
     * @param[in] shape           @ref TensorShape to copy the dimensions from.
     * @param[in] first_dimension Only copy dimensions which are greater or equal to this value.
     */
    void use_tensor_dimensions(const TensorShape &shape, size_t first_dimension = Window::DimX);

    /** Shift the values of a given dimension by the given shift_value
     *
     * @param[in] dimension   The dimension to shift
     * @param[in] shift_value Value to shift the start and end values of.
     */
    void shift(size_t dimension, int shift_value);

    /** Shift down all the dimensions of a window
     *
     * i.e new_dims[n] = old_dims[n+shift_value].
     *
     * @param[in] shift_value Number of dimensions to shift the window by.
     *
     * @return The window with the shifted dimensions.
     */
    Window shift_dimensions(unsigned int shift_value) const;

    /** Adjust the start or end of a given dimension by the given value
     *
     * @param[in] dimension    The dimension to adjust
     * @param[in] adjust_value The adjusted value.
     * @param[in] is_at_start  The flag to indicate whether adjust the start or end of the dimension.
     */
    void adjust(size_t dimension, int adjust_value, bool is_at_start);

    /** Scale the values of a given dimension by the given scale_value
     *
     * @note The end of the window is rounded up to be a multiple of step after the scaling.
     *
     * @param[in] dimension   The dimension to scale
     * @param[in] scale_value Value to scale the start, end and step values of.
     */
    void scale(size_t dimension, float scale_value);

    /** Set the step of a given dimension.
     *
     * @param[in] dimension Dimension to update
     * @param[in] step      The new dimension's step value
     */
    void set_dimension_step(size_t dimension, int step);

    /** Will validate all the window's dimensions' values when asserts are enabled
     *
     * No-op when asserts are disabled
     */
    void validate() const;

    /** Return the number of iterations needed to iterate through a given dimension
     *
     * @param[in] dimension The requested dimension
     *
     * @return The number of iterations
     */
    constexpr size_t num_iterations(size_t dimension) const;
    /** Return the total number of iterations needed to iterate through the entire window
     *
     * @return Number of total iterations
     */
    size_t num_iterations_total() const;
    /** Return the shape of the window in number of steps */
    TensorShape shape() const;
    /** Split a window into a set of sub windows along a given dimension
     *
     * For example to split a window into 3 sub-windows along the Y axis, you would have to do:<br/>
     * Window sub0 = window.split_window( 1, 0, 3);<br/>
     * Window sub1 = window.split_window( 1, 1, 3);<br/>
     * Window sub2 = window.split_window( 1, 2, 3);<br/>
     *
     * @param[in] dimension Dimension along which the split will be performed
     * @param[in] id        Id of the sub-window to return. Must be in the range (0, total-1)
     * @param[in] total     Total number of sub-windows the window will be split into.
     *
     * @return The subwindow "id" out of "total"
     */
    Window split_window(size_t dimension, size_t id, size_t total) const;
    /** First 1D slice of the window
     *
     * @return The first slice of the window.
     */
    Window first_slice_window_1D() const
    {
        return first_slice_window<1>();
    };
    /** First 2D slice of the window
     *
     * @return The first slice of the window.
     */
    Window first_slice_window_2D() const
    {
        return first_slice_window<2>();
    };
    /** First 3D slice of the window
     *
     * @return The first slice of the window.
     */
    Window first_slice_window_3D() const
    {
        return first_slice_window<3>();
    };
    /** First 4D slice of the window
     *
     * @return The first slice of the window.
     */
    Window first_slice_window_4D() const
    {
        return first_slice_window<4>();
    };
    /** Slide the passed 1D window slice.
     *
     * If slice contains the last slice then it will remain unchanged and false will be returned.
     *
     * @param[in,out] slice Current slice, to be updated to the next slice.
     *
     * @return true if slice contains a new slice, false if slice already contained the last slice
     */
    bool slide_window_slice_1D(Window &slice) const
    {
        return slide_window_slice<1>(slice);
    }
    /** Slide the passed 2D window slice.
     *
     * If slice contains the last slice then it will remain unchanged and false will be returned.
     *
     * @param[in,out] slice Current slice, to be updated to the next slice.
     *
     * @return true if slice contains a new slice, false if slice already contained the last slice
     */
    bool slide_window_slice_2D(Window &slice) const
    {
        return slide_window_slice<2>(slice);
    }
    /** Slide the passed 3D window slice.
     *
     * If slice contains the last slice then it will remain unchanged and false will be returned.
     *
     * @param[in,out] slice Current slice, to be updated to the next slice.
     *
     * @return true if slice contains a new slice, false if slice already contained the last slice
     */
    bool slide_window_slice_3D(Window &slice) const
    {
        return slide_window_slice<3>(slice);
    }
    /** Slide the passed 4D window slice.
     *
     * If slice contains the last slice then it will remain unchanged and false will be returned.
     *
     * @param[in,out] slice Current slice, to be updated to the next slice.
     *
     * @return true if slice contains a new slice, false if slice already contained the last slice
     */
    bool slide_window_slice_4D(Window &slice) const
    {
        return slide_window_slice<4>(slice);
    }
    /** Collapse the dimensions between @p first and @p last if possible.
     *
     * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
     *
     * @param[in]  full_window   Full window @p window has been created from.
     * @param[in]  first         Start dimension into which the following are collapsed.
     * @param[in]  last          End (exclusive) dimension to collapse.
     * @param[out] has_collapsed (Optional) Whether the window was collapsed.
     *
     * @return Collapsed window.
     */
    Window collapse_if_possible(const Window &full_window, size_t first, size_t last, bool *has_collapsed = nullptr) const;

    /** Collapse the dimensions higher than @p first if possible.
     *
     * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
     *
     * @param[in]  full_window   Full window @p window has been created from.
     * @param[in]  first         Start dimension into which the following are collapsed.
     * @param[out] has_collapsed (Optional) Whether the window was collapsed.
     *
     * @return Collapsed window.
     */
    Window collapse_if_possible(const Window &full_window, size_t first, bool *has_collapsed = nullptr) const
    {
        return collapse_if_possible(full_window, first, Coordinates::num_max_dimensions, has_collapsed);
    }

    /** Collapse the dimensions between @p first and @p last.
     *
     * A dimension is collapsable if it starts from 0 and matches the corresponding dimension in the full_window
     *
     * @param[in] full_window Full window @p window has been created from.
     * @param[in] first       Start dimension into which the following are collapsed.
     * @param[in] last        End (exclusive) dimension to collapse.
     *
     * @return Collapsed window if successful.
     */
    Window collapse(const Window &full_window, size_t first, size_t last = Coordinates::num_max_dimensions) const;

    /** Don't advance in the dimension where @p shape is less equal to 1.
     *
     * @param[in] shape A TensorShape.
     *
     * @return Broadcast window.
     */
    Window broadcast_if_dimension_le_one(const TensorShape &shape) const;

    /** Don't advance in the dimension where shape of @p info is less equal to 1.
     *
     * @param[in] info An ITensorInfo.
     *
     * @return Broadcast window.
     */
    Window broadcast_if_dimension_le_one(const ITensorInfo &info) const
    {
        return broadcast_if_dimension_le_one(info.tensor_shape());
    }
    /** Friend function that swaps the contents of two windows
     *
     * @param[in] lhs First window to swap.
     * @param[in] rhs Second window to swap.
     */
    friend void swap(Window &lhs, Window &rhs);
    /** Check whether two Windows are equal.
     *
     * @param[in] lhs LHS window
     * @param[in] rhs RHS window
     *
     * @return True if the given windows are the same.
     */
    friend bool operator==(const Window &lhs, const Window &rhs);

private:
    /** First slice of the window
     *
     * @return The first slice of the window.
     */
    template <unsigned int window_dimension>
    Window                 first_slice_window() const;

    /** Slide the passed window slice.
     *
     * If slice contains the last slice then it will remain unchanged and false will be returned.
     *
     * @param[in,out] slice Current slice, to be updated to the next slice.
     *
     * @return true if slice contains a new slice, false if slice already contained the last slice
     */
    template <unsigned int window_dimension>
    bool slide_window_slice(Window &slice) const;

private:
    std::array<Dimension, Coordinates::num_max_dimensions> _dims;
    std::array<bool, Coordinates::num_max_dimensions>      _is_broadcasted;
};
} // namespace arm_compute
#include "Window.inl"
#endif /*ARM_COMPUTE_WINDOW_H */
