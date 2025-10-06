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

#ifndef ACL_ARM_COMPUTE_CORE_IREDUCIBLETENSOR_H
#define ACL_ARM_COMPUTE_CORE_IREDUCIBLETENSOR_H

#include "arm_compute/core/SparseTensor.h"

namespace arm_compute
{
/** Forward declaration of COOTensor and CSRTensor class */
class COOTensor;
class CSRTensor;

/** Interface for all reducible tensors, i.e. all tensors that can be
 *  converted to a sparse representation.
 */
class IReducibleTensor
{
public:
    virtual ~IReducibleTensor() = default;
    /** Convert a dense tensor to sparse tensor with specified sparse dimensions using the default
     *  sparse tensor representation: COO format.
     *
     * @param[in] dim sparse dimension
     *
     * @return A unique pointer to a SparseTensor object.
     */
    virtual std::unique_ptr<SparseTensor> to_sparse(size_t dim) const = 0;
    /** Convert a dense tensor to COO sparse tensor with specified sparse dimensions.
     *
     * @param[in] dim sparse dimension
     *
     * @return A unique pointer to a COOTensor object.
     */
    virtual std::unique_ptr<COOTensor> to_coo_sparse(size_t dim) const = 0;
    /** Convert a dense tensor to COO sparse tensor with the default sparse dimension.
     *  For COO format, the number of sparse dimensions is equal to the total number of dimensions.
     *
     * @return A unique pointer to a COOTensor object.
     */
    virtual std::unique_ptr<COOTensor> to_coo_sparse() const = 0;
    /** Convert a dense tensor to CSR sparse tensor with specified sparse dimensions.
     *
     * @param[in] dim sparse dimension
     *
     * @return A unique pointer to a CSRTensor object.
     */
    virtual std::unique_ptr<CSRTensor> to_csr_sparse(size_t dim) const = 0;
    /** Convert a dense tensor to CSR sparse tensor with the default sparse dimension.
     *  For CSR format, the number of sparse dimensions is equal to 2.
     *
     * @return A unique pointer to a CSRTensor object.
     */
    virtual std::unique_ptr<CSRTensor> to_csr_sparse() const = 0;
};
}

#endif // ACL_ARM_COMPUTE_CORE_IREDUCIBLETENSOR_H
