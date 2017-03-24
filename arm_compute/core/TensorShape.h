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
#ifndef DOXYGEN_SKIP_THIS /* Doxygen gets confused by the templates and can't match the implementation to the declaration */
    /** Constructor to initialize the tensor shape.
     *
     * @param[in] dims Values to initialize the dimensions.
     */
    template <typename... Ts>
    TensorShape(Ts... dims)
        : Dimensions{ dims... }
    {
        // Initialize empty dimensions to 1
        std::fill(_id.begin() + _num_dimensions, _id.end(), 1);
    }
#endif
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
    /** Collapses all dimensions to a single linear total size.
     *
     * @return The total tensor size in terms of elements.
     */
    size_t total_size() const
    {
        const size_t size = std::accumulate(_id.begin(), _id.end(), 1, std::multiplies<size_t>());
        ARM_COMPUTE_ERROR_ON(0 == size);
        return size;
    }
    /** Collapses given dimension and above.
     *
     * @note Precondition: dimension < TensorShape::num_max_dimensions
     *
     * @param[in] dimension Size of the wanted dimension
     *
     * @return The linear size of the collapsed dimensions
     */
    size_t total_size_upper(size_t dimension) const
    {
        const size_t size = std::accumulate(_id.begin() + dimension, _id.end(), 1, std::multiplies<size_t>());
        ARM_COMPUTE_ERROR_ON(0 == size);
        return size;
    }
};
}
#endif /*__ARM_COMPUTE_TENSORSHAPE_H__*/
