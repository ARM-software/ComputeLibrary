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
#ifndef __ARM_COMPUTE_TENSORSHAPE_H__
#define __ARM_COMPUTE_TENSORSHAPE_H__

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

namespace arm_compute
{
/** Shape of a tensor */
class TensorShape : public Dimensions<size_t>
{
public:
    /** Constructor to initialize the tensor shape.
     *
     * @param[in] dims Values to initialize the dimensions.
     */
    template <typename... Ts>
    TensorShape(Ts... dims)
        : Dimensions{ dims... }
    {
        // Initialize unspecified dimensions to 1
        if(_num_dimensions > 0)
        {
            std::fill(_id.begin() + _num_dimensions, _id.end(), 1);
        }

        // Correct number dimensions to ignore trailing dimensions of size 1
        apply_dimension_correction();
    }
    /** Allow instances of this class to be copy constructed */
    TensorShape(const TensorShape &) = default;
    /** Allow instances of this class to be copied */
    TensorShape &operator=(const TensorShape &) = default;
    /** Allow instances of this class to be move constructed */
    TensorShape(TensorShape &&) = default;
    /** Allow instances of this class to be moved */
    TensorShape &operator=(TensorShape &&) = default;
    /** Default destructor */
    ~TensorShape() = default;

    /** Accessor to set the value of one of the dimensions.
     *
     * @param[in] dimension Dimension for which the value is set.
     * @param[in] value     Value to be set for the dimension.
     */
    void set(size_t dimension, size_t value)
    {
        // Clear entire shape if one dimension is zero
        if(value == 0)
        {
            _num_dimensions = 0;
            std::fill(_id.begin(), _id.end(), 0);
            return;
        }

        // Make sure all empty dimensions are filled with 1
        std::fill(_id.begin() + _num_dimensions, _id.end(), 1);

        // Set the specified dimension and increase the number of dimensions if
        // necessary
        Dimensions::set(dimension, value);

        // Correct number dimensions to ignore trailing dimensions of size 1
        apply_dimension_correction();
    }

    /** Accessor to remove the dimension n from the tensor shape.
     *
     * @note The upper dimensions of the tensor shape will be shifted down by 1
     *
     * @param[in] n Dimension to remove
     */
    void remove_dimension(size_t n)
    {
        ARM_COMPUTE_ERROR_ON(_num_dimensions < 1);
        ARM_COMPUTE_ERROR_ON(n >= _num_dimensions);

        std::copy(_id.begin() + n + 1, _id.end(), _id.begin() + n);

        // Reduce number of dimensions
        _num_dimensions--;

        // Make sure all empty dimensions are filled with 1
        std::fill(_id.begin() + _num_dimensions, _id.end(), 1);

        // Correct number dimensions to ignore trailing dimensions of size 1
        apply_dimension_correction();
    }

    /** Collapse the first n dimensions.
     *
     * @param[in] n     Number of dimensions to collapse into @p first
     * @param[in] first Dimensions into which the following @p n are collapsed.
     */
    void collapse(size_t n, size_t first = 0)
    {
        Dimensions::collapse(n, first);

        // Make sure all empty dimensions are filled with 1
        std::fill(_id.begin() + _num_dimensions, _id.end(), 1);
    }

    /** Collapses all dimensions to a single linear total size.
     *
     * @return The total tensor size in terms of elements.
     */
    size_t total_size() const
    {
        return std::accumulate(_id.begin(), _id.end(), 1, std::multiplies<size_t>());
    }
    /** Collapses given dimension and above.
     *
     * @param[in] dimension Size of the wanted dimension
     *
     * @return The linear size of the collapsed dimensions
     */
    size_t total_size_upper(size_t dimension) const
    {
        ARM_COMPUTE_ERROR_ON(dimension >= TensorShape::num_max_dimensions);
        return std::accumulate(_id.begin() + dimension, _id.end(), 1, std::multiplies<size_t>());
    }

    /** Compute size of dimensions lower than the given one.
     *
     * @param[in] dimension Upper boundary.
     *
     * @return The linear size of the collapsed dimensions.
     */
    size_t total_size_lower(size_t dimension) const
    {
        ARM_COMPUTE_ERROR_ON(dimension > TensorShape::num_max_dimensions);
        return std::accumulate(_id.begin(), _id.begin() + dimension, 1, std::multiplies<size_t>());
    }

private:
    /** Remove trailing dimensions of size 1 from the reported number of dimensions. */
    void apply_dimension_correction()
    {
        for(int i = static_cast<int>(_num_dimensions) - 1; i > 0; --i)
        {
            if(_id[i] == 1)
            {
                --_num_dimensions;
            }
            else
            {
                break;
            }
        }
    }
};
}
#endif /*__ARM_COMPUTE_TENSORSHAPE_H__*/
