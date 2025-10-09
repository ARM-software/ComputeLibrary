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

#ifndef ACL_ARM_COMPUTE_RUNTIME_COOTENSOR_H
#define ACL_ARM_COMPUTE_RUNTIME_COOTENSOR_H

#include "arm_compute/core/SparseTensor.h"
#include "arm_compute/runtime/SparseTensorAllocator.h"

#include <vector>

namespace arm_compute
{
class COOTensor final : public SparseTensor, public IMemoryManageable
{
public:
    /** Prevent instances of this class to be move constructed */
    COOTensor(COOTensor &&) = delete;
    /** Prevent instances of this class to be moved */
    COOTensor &operator=(COOTensor &&) = delete;

    /** Print the internal state of the COOTensor instance
    *
    * @param[in] os the output stream; std::cout set as default.
    *
    * @note It prints (on os stream) the vector of the indices with the format
    *       index: [idx_0, idx_1, ...], and the corresponding values with the format
    *       value: [val_0, val_1, ...].
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
    /** Convert a dense tensor to sparse tensor with specified sparse dimensions using COO format.
     *
     *  @param[in] tensor
     *  @param[in] sparse_dim   Belongs to [1, tensor->info->num_dimensions()]
     */
    COOTensor(const ITensor *tensor, size_t sparse_dim);
    /** Convert a dense tensor to a *fully* sparse COOTensor.
     *
     *  @param[in] tensor
     *
     *  @note sparse_dim = tensor->info->num_dimensions()
     */
    COOTensor(const ITensor *tensor);

    std::vector<Coordinates> _indices;
    mutable SparseTensorAllocator _allocator; /**< Instance of the basic CPU allocator.*/

friend class Tensor;
};
}

#endif // ACL_ARM_COMPUTE_RUNTIME_COOTENSOR_H
