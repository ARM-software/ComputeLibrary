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

#ifndef ACL_ARM_COMPUTE_RUNTIME_CSRTENSOR_H
#define ACL_ARM_COMPUTE_RUNTIME_CSRTENSOR_H

#include "arm_compute/core/SparseTensor.h"
#include "arm_compute/runtime/SparseTensorAllocator.h"

#include <vector>

namespace arm_compute
{
class CSRTensor final : public SparseTensor, public IMemoryManageable
{
public:
    /** Prevent instances of this class to be move constructed */
    CSRTensor(CSRTensor &&) = delete;
    /** Prevent instances of this class to be moved */
    CSRTensor &operator=(CSRTensor &&) = delete;

    /** Print the internal state of the CSRTensor instance
    *
    * @param[in] os the output stream; std::cout set as default.
    *
    * @note It prints (on os stream) the two vectors of the indices with the format
    *       [row_idx_0, row_idx_1, ...] and [col_idx_0, col_idx_1, ...]
    * @note This print function should overlap the one defined for ITensor.
    */
    void print(std::ostream &os = std::cout) const;
    
    // Inherited methods overridden:
    ITensorInfo *info() const override;
    ITensorInfo *info() override;
    uint8_t *buffer() const override;
    size_t nnz() const override;
    std::unique_ptr<ITensor> to_dense() override;
    Coordinates get_coordinates(size_t nth) const override;
    const uint8_t *get_value(Coordinates coords) const override;
    void associate_memory_group(IMemoryGroup *memory_group) override;

private:
    /** The size of each index element */
    static constexpr size_t index_size = sizeof(int32_t);

    /** Convert a dense tensor to sparse tensor with specified sparse dimensions using COO format.
     *
     *  @param[in] tensor
     *  @param[in] sparse_dim  It should belong to [1, tensor->info->num_dimensions()]
     */
    CSRTensor(const ITensor *tensor, size_t sparse_dim);
    /** Convert a dense tensor to a *fully* sparse tensor.
     *
     *  @note sparse_dim = tensor->info->num_dimensions().
     *        If tensor->info->num_dimensions() > 2 an error is raised.
     */
    CSRTensor(const ITensor *tensor);

    size_t _crow_bytes; /**< Row index size in bytes */
    size_t _col_bytes;  /**< Column index size in bytes */
    // In the SparseTensorAllocator buffer, the memory is stored that way
    // +---------------+---------------+----------...
    // |  Row offsets  |  Col indices  |  Values  ...
    // +---------------+---------------+----------...
    mutable SparseTensorAllocator _allocator; /**< Instance of the basic CPU allocator.*/

friend class Tensor;
};
}

#endif // ACL_ARM_COMPUTE_RUNTIME_CSRTENSOR_H
