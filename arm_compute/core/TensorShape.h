/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TENSORSHAPE_H
#define ARM_COMPUTE_TENSORSHAPE_H

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/Utility.h"

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
     * @param[in] dimension            Dimension for which the value is set.
     * @param[in] value                Value to be set for the dimension.
     * @param[in] apply_dim_correction (Optional) Flag to state whether apply dimension correction after setting one dimension. E.g. when permuting NCHW -> NHWC, 1x1x2 would become 2x1x1, but _num_dimensions should be 3 rather than 1.
     * @param[in] increase_dim_unit    (Optional) Set to true if new unit dimensions increase the number of dimensions of the shape.
     *
     * @return *this.
     */
    TensorShape &set(size_t dimension, size_t value, bool apply_dim_correction = true, bool increase_dim_unit = true)
    {
        // Clear entire shape if one dimension is zero
        if(value == 0)
        {
            _num_dimensions = 0;
            std::fill(_id.begin(), _id.end(), 0);
        }
        else
        {
            // Make sure all empty dimensions are filled with 1
            std::fill(_id.begin() + _num_dimensions, _id.end(), 1);

            // Set the specified dimension and increase the number of dimensions if
            // necessary
            Dimensions::set(dimension, value, increase_dim_unit);

            // Correct number dimensions to ignore trailing dimensions of size 1
            if(apply_dim_correction)
            {
                apply_dimension_correction();
            }
        }
        return *this;
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
    /** Shifts right the tensor shape increasing its dimensions
     *
     * @param[in] step Rotation step
     */
    void shift_right(size_t step)
    {
        ARM_COMPUTE_ERROR_ON(step > TensorShape::num_max_dimensions - num_dimensions());

        std::rotate(begin(), begin() + TensorShape::num_max_dimensions - step, end());
        _num_dimensions += step;

        // Correct number dimensions to ignore trailing dimensions of size 1
        apply_dimension_correction();
    }

    /** Return a copy with collapsed dimensions starting from a given point.
     *
     * @param[in] start Starting point of collapsing dimensions.
     *
     * @return A copy with collapse dimensions starting from start.
     */
    TensorShape collapsed_from(size_t start) const
    {
        TensorShape copy(*this);
        copy.collapse(num_dimensions() - start, start);
        return copy;
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

    /** If shapes are broadcast compatible, return the broadcasted shape.
     *
     * Two tensor shapes are broadcast compatible if for each dimension, they're equal or one of them is 1.
     *
     * If two shapes are compatible, each dimension in the broadcasted shape is the max of the original dimensions.
     *
     * @param[in] shapes Tensor shapes.
     *
     * @return The broadcasted shape or an empty shape if the shapes are not broadcast compatible.
     */
    template <typename... Shapes>
    static TensorShape broadcast_shape(const Shapes &... shapes)
    {
        TensorShape bc_shape;

        auto broadcast = [&bc_shape](const TensorShape & other)
        {
            if(bc_shape.num_dimensions() == 0)
            {
                bc_shape = other;
            }
            else if(other.num_dimensions() != 0)
            {
                for(size_t d = 0; d < TensorShape::num_max_dimensions; ++d)
                {
                    const size_t dim_min = std::min(bc_shape[d], other[d]);
                    const size_t dim_max = std::max(bc_shape[d], other[d]);

                    if((dim_min != 1) && (dim_min != dim_max))
                    {
                        bc_shape = TensorShape{ 0U };
                        break;
                    }

                    bc_shape.set(d, dim_max);
                }
            }
        };

        utility::for_each(broadcast, shapes...);

        return bc_shape;
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
#endif /*ARM_COMPUTE_TENSORSHAPE_H*/
