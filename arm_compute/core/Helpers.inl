/*
 * Copyright (c) 2016-2021, 2023 Arm Limited.
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
#include "arm_compute/core/Error.h"

#include <cmath>
#include <numeric>

namespace arm_compute
{
template <size_t dimension>
struct IncrementIterators
{
    template <typename T, typename... Ts>
    static void unroll(T &&it, Ts &&... iterators)
    {
        auto increment = [](T && it)
        {
            it.increment(dimension);
        };
        utility::for_each(increment, std::forward<T>(it), std::forward<Ts>(iterators)...);
    }
    static void unroll()
    {
        // End of recursion
    }
};

template <size_t dim>
struct ForEachDimension
{
    template <typename L, typename... Ts>
    static void unroll(const Window &w, Coordinates &id, L &&lambda_function, Ts &&... iterators)
    {
        const auto &d = w[dim - 1];

        for(auto v = d.start(); v < d.end(); v += d.step(), IncrementIterators < dim - 1 >::unroll(iterators...))
        {
            id.set(dim - 1, v);
            ForEachDimension < dim - 1 >::unroll(w, id, lambda_function, iterators...);
        }
    }
};

template <>
struct ForEachDimension<0>
{
    template <typename L, typename... Ts>
    static void unroll(const Window &w, Coordinates &id, L &&lambda_function, Ts &&... iterators)
    {
        ARM_COMPUTE_UNUSED(w, iterators...);
        lambda_function(id);
    }
};

template <typename L, typename... Ts>
inline void execute_window_loop(const Window &w, L &&lambda_function, Ts &&... iterators)
{
    w.validate();

    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(w[i].step() == 0);
    }

    Coordinates id;
    ForEachDimension<Coordinates::num_max_dimensions>::unroll(w, id, std::forward<L>(lambda_function), std::forward<Ts>(iterators)...);
}

inline constexpr Iterator::Iterator()
    : _ptr(nullptr), _dims()
{
}

inline Iterator::Iterator(const ITensor *tensor, const Window &win)
    : Iterator()
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);
    ARM_COMPUTE_ERROR_ON(tensor->info() == nullptr);

    initialize(tensor->info()->num_dimensions(), tensor->info()->strides_in_bytes(), tensor->buffer(), tensor->info()->offset_first_element_in_bytes(), win);
}

inline Iterator::Iterator(size_t num_dims, const Strides &strides, uint8_t *buffer, size_t offset, const Window &win)
    : Iterator()
{
    initialize(num_dims, strides, buffer, offset, win);
}

inline void Iterator::initialize(size_t num_dims, const Strides &strides, uint8_t *buffer, size_t offset, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(buffer == nullptr);

    _ptr = buffer + offset;

    //Initialize the stride for each dimension and calculate the position of the first element of the iteration:
    for(unsigned int n = 0; n < num_dims; ++n)
    {
        _dims[n]._stride = win[n].step() * strides[n];
        std::get<0>(_dims)._dim_start += static_cast<size_t>(strides[n]) * win[n].start();
    }

    //Copy the starting point to all the dimensions:
    for(unsigned int n = 1; n < Coordinates::num_max_dimensions; ++n)
    {
        _dims[n]._dim_start = std::get<0>(_dims)._dim_start;
    }

    ARM_COMPUTE_ERROR_ON_WINDOW_DIMENSIONS_GTE(win, num_dims);
}

inline void Iterator::increment(const size_t dimension)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);

    _dims[dimension]._dim_start += _dims[dimension]._stride;

    for(unsigned int n = 0; n < dimension; ++n)
    {
        _dims[n]._dim_start = _dims[dimension]._dim_start;
    }
}

inline constexpr size_t Iterator::offset() const
{
    return _dims.at(0)._dim_start;
}

inline constexpr uint8_t *Iterator::ptr() const
{
    return _ptr + _dims.at(0)._dim_start;
}

inline void Iterator::reset(const size_t dimension)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions - 1);

    _dims[dimension]._dim_start = _dims[dimension + 1]._dim_start;

    for(unsigned int n = 0; n < dimension; ++n)
    {
        _dims[n]._dim_start = _dims[dimension]._dim_start;
    }
}

inline Coordinates index2coords(const TensorShape &shape, int index)
{
    int num_elements = shape.total_size();

    ARM_COMPUTE_ERROR_ON_MSG(index < 0 || index >= num_elements, "Index has to be in [0, num_elements]!");
    ARM_COMPUTE_ERROR_ON_MSG(num_elements == 0, "Cannot create coordinate from empty shape!");

    Coordinates coord{ 0 };

    for(int d = shape.num_dimensions() - 1; d >= 0; --d)
    {
        num_elements /= shape[d];
        coord.set(d, index / num_elements);
        index %= num_elements;
    }

    return coord;
}

inline int coords2index(const TensorShape &shape, const Coordinates &coord)
{
    int num_elements = shape.total_size();
    ARM_COMPUTE_UNUSED(num_elements);
    ARM_COMPUTE_ERROR_ON_MSG(num_elements == 0, "Cannot create linear index from empty shape!");

    int index  = 0;
    int stride = 1;

    for(unsigned int d = 0; d < coord.num_dimensions(); ++d)
    {
        index += coord[d] * stride;
        stride *= shape[d];
    }

    return index;
}

inline size_t get_data_layout_dimension_index(const DataLayout &data_layout, const DataLayoutDimension &data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(data_layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");
    const auto &dims = get_layout_map().at(data_layout);
    const auto &it   = std::find(dims.cbegin(), dims.cend(), data_layout_dimension);
    ARM_COMPUTE_ERROR_ON_MSG(it == dims.cend(), "Invalid dimension for the given layout.");
    return it - dims.cbegin();
}

inline DataLayoutDimension get_index_data_layout_dimension(const DataLayout &data_layout, const size_t index)
{
    ARM_COMPUTE_ERROR_ON_MSG(data_layout == DataLayout::UNKNOWN, "Cannot retrieve the layout dimension for an unknown layout!");
    const auto &dims = get_layout_map().at(data_layout);
    ARM_COMPUTE_ERROR_ON_MSG(index >= dims.size(), "Invalid index for the given layout.");
    return dims[index];
}
} // namespace arm_compute
