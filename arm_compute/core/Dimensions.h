/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_DIMENSIONS_H__
#define __ARM_COMPUTE_DIMENSIONS_H__

#include "arm_compute/core/Error.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

namespace arm_compute
{
/** Constant value used to indicate maximum dimensions of a Window, TensorShape and Coordinates */
constexpr size_t MAX_DIMS = 6;

/** Dimensions with dimensionality */
template <typename T>
class Dimensions
{
public:
    /** Number of dimensions the tensor has */
    static constexpr size_t num_max_dimensions = MAX_DIMS;

    /** Constructor to initialize the tensor shape.
     *
     * @param[in] dims Values to initialize the dimensions.
     */
    template <typename... Ts>
    explicit Dimensions(Ts... dims)
        : _id{ { static_cast<T>(dims)... } }, _num_dimensions{ sizeof...(dims) }
    {
    }

    /** Allow instances of this class to be copy constructed */
    Dimensions(const Dimensions &) = default;

    /** Allow instances of this class to be copied */
    Dimensions &operator=(const Dimensions &) = default;

    /** Allow instances of this class to be move constructed */
    Dimensions(Dimensions &&) = default;

    /** Allow instances of this class to be moved */
    Dimensions &operator=(Dimensions &&) = default;

    /** Accessor to set the value of one of the dimensions.
     *
     * @param[in] dimension Dimension for which the value is set.
     * @param[in] value     Value to be set for the dimension.
     */
    void set(size_t dimension, T value)
    {
        ARM_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
        _id[dimension]  = value;
        _num_dimensions = std::max(_num_dimensions, dimension + 1);
    }
    /** Alias to access the size of the first dimension */
    T x() const
    {
        return _id[0];
    }
    /** Alias to access the size of the second dimension */
    T y() const
    {
        return _id[1];
    }
    /** Alias to access the size of the third dimension */
    T z() const
    {
        return _id[2];
    }
    /** Generic accessor to get the size of any dimension
     *
     * @note Precondition: dimension < Dimensions::num_max_dimensions
     *
     * @param[in] dimension Dimension of the wanted size
     *
     * @return The size of the requested dimension.
     */
    const T &operator[](size_t dimension) const
    {
        ARM_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
        return _id[dimension];
    }
    /** Generic accessor to get the size of any dimension
     *
     * @note Precondition: dimension < Dimensions::num_max_dimensions
     *
     * @param[in] dimension Dimension of the wanted size
     *
     * @return The size of the requested dimension.
     */
    T &operator[](size_t dimension)
    {
        ARM_COMPUTE_ERROR_ON(dimension >= num_max_dimensions);
        return _id[dimension];
    }
    /** Returns the effective dimensionality of the tensor */
    unsigned int num_dimensions() const
    {
        return _num_dimensions;
    }

    /** Set number of dimensions */
    void set_num_dimensions(size_t num_dimensions)
    {
        _num_dimensions = num_dimensions;
    }

    /** Collapse dimensions.
     *
     * @param[in] n     Number of dimensions to collapse into @p first.
     * @param[in] first Dimensions into which the following @p n are collapsed.
     */
    void collapse(const size_t n, const size_t first = 0)
    {
        ARM_COMPUTE_ERROR_ON(first + n > _id.size());

        const size_t last = std::min(_num_dimensions, first + n);

        if(last > (first + 1))
        {
            // Collapse dimensions into the first
            _id[first] = std::accumulate(&_id[first], &_id[last], 1, std::multiplies<T>());
            // Shift the remaining dimensions down
            std::copy(&_id[last], &_id[_num_dimensions], &_id[first + 1]);
            // Reduce the number of dimensions
            const size_t old_num_dimensions = _num_dimensions;
            _num_dimensions -= last - first - 1;
            // Fill the now empty dimensions with zero
            std::fill(&_id[_num_dimensions], &_id[old_num_dimensions], 0);
        }
    }

    /** Collapse dimensions starting from a given point
     *
     * @param[in] start Starting point of collapsing dimensions
     */
    void collapse_from(size_t start)
    {
        ARM_COMPUTE_ERROR_ON(start > num_dimensions());

        collapse(num_dimensions() - start, start);
    }

    /** Remove dimension of a given index
     *
     * @note If index is greater than the number of dimensions no operation is performed
     *
     * @param[in] idx Dimension index to remove
     */
    void remove(size_t idx)
    {
        ARM_COMPUTE_ERROR_ON(_num_dimensions < 1);
        if(idx >= _num_dimensions)
        {
            return;
        }

        std::copy(_id.begin() + idx + 1, _id.end(), _id.begin() + idx);
        _num_dimensions--;

        // Make sure all empty dimensions are filled with 0
        std::fill(_id.begin() + _num_dimensions, _id.end(), 0);
    }

    /** Returns a read/write iterator that points to the first element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::iterator begin()
    {
        return _id.begin();
    }
    /** Returns a read-only (constant) iterator that points to the first element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::const_iterator begin() const
    {
        return _id.begin();
    }
    /** Returns a read-only (constant) iterator that points to the first element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::const_iterator cbegin() const
    {
        return begin();
    }
    /** Returns a read/write iterator that points one past the last element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::iterator end()
    {
        return _id.end();
    }
    /** Returns a read-only (constant) iterator that points one past the last element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::const_iterator end() const
    {
        return _id.end();
    }
    /** Returns a read-only (constant) iterator that points one past the last element in the dimension array.
     *
     * @return an iterator.
     */
    typename std::array<T, num_max_dimensions>::const_iterator cend() const
    {
        return end();
    }

protected:
    /** Protected destructor. */
    ~Dimensions() = default;

    std::array<T, num_max_dimensions> _id;
    size_t _num_dimensions{ 0 };
};

/** Check that given dimensions are equal.
 *
 * @param[in] lhs Left-hand side Dimensions.
 * @param[in] rhs Right-hand side Dimensions.
 *
 * @return True if the given dimensions are equal.
 */
template <typename T>
inline bool operator==(const Dimensions<T> &lhs, const Dimensions<T> &rhs)
{
    return ((lhs.num_dimensions() == rhs.num_dimensions()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
}
/** Check that given dimensions are not equal.
 *
 * @param[in] lhs Left-hand side Dimensions.
 * @param[in] rhs Right-hand side Dimensions.
 *
 * @return True if the given dimensions are not equal.
 */
template <typename T>
inline bool operator!=(const Dimensions<T> &lhs, const Dimensions<T> &rhs)
{
    return !(lhs == rhs);
}
}
#endif /*__ARM_COMPUTE_DIMENSIONS_H__*/
