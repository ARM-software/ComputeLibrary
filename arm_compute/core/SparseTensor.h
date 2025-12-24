/*
 * Copyright (c) 2025 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_SPARSETENSOR_H
#define ACL_ARM_COMPUTE_RUNTIME_SPARSETENSOR_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"

#include <functional>
#include <stdexcept>

typedef std::function<bool(const void *)> predicate_t;

namespace arm_compute
{
/** Common base class for all sparse tensors */
class SparseTensor : public ITensor
{
public:
    /** Prevent instances of this class to be constructed by the default constructor */
    SparseTensor() = delete;
    /** Prevent instances of this class to be copy constructed */
    SparseTensor(const SparseTensor&) = delete;
    /** Prevent instances of this class to be copied */
    SparseTensor& operator=(const SparseTensor&) = delete;
    ~SparseTensor() = default;

    /** Returns the number of sparse dimensions */
    size_t sparse_dim() const;
    /** Returns the number of dense dimensions */
    size_t dense_dim() const;
    /** Returns the (total) number of dimensions */
    size_t dim() const;
    /** Returns true if the tensor is hybrid (contains both sparse and dense dimensions)
     *  
     *  @note A sparse tensor is hybrid if it has at least one dense dimension.
     */
    bool is_hybrid() const;
    /** Returns the ratio of zero-valued elements to the total number of elements */
    float sparsity() const;
    /** Returns the ratio of non-zero elements to the total number of elements */
    float density() const;
    /** Returns the dense volume */
    uint32_t dense_volume(size_t sparse_dim) const;
    /** Returns the number of non zero elements */
    virtual size_t nnz() const = 0;
    /** Converts the sparse tensor to a dense tensor */
    virtual std::unique_ptr<ITensor> to_dense() = 0;
    /** Returns the coordinates of the n-th (non-zero) element.
     *
     *  @param nth The *zero-base* index of the element
     *
     *  @return The coordinates of the element
     */
    virtual Coordinates get_coordinates(size_t nth) const = 0;
    /** Returns a pointer to the n-th (non-zero) element. If the element specified by
     *  the coordinates is zero, nullptr is returned.
     *
     *  @param nth The *zero-base* index of the element
     *
     *  @return The value of the element
     *
     *  @note The value has size dense_volume(sparse_dim()).
     */
    virtual const uint8_t *get_value(Coordinates coords) const = 0;

protected:
    SparseTensor(size_t dim, size_t sparse_dim);

    std::function<bool(const void *)> make_is_nonzero_predicate(DataType dt) const;
    bool has_non_zero_elements(uint8_t *arr, size_t len, size_t element_size, predicate_t is_non_zero) const;
    void print_values(std::ostream &os, const uint8_t *data, size_t offset, size_t count) const;

private:
    size_t _total_dim;
    size_t _sparse_dim;
};
}

#endif // ACL_ARM_COMPUTE_RUNTIME_SPARSETENSOR_H
