/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ACLUTILS_H_
#define ARM_COMPUTE_ACLUTILS_H_

#include "arm_compute/AclTypes.h"

#ifdef __cplusplus
extern "C" {
#endif /** __cplusplus */

/** Get the size of the existing tensor in byte
 *
 * @note The size isn't based on allocated memory, but based on information in its descriptor (dimensions, data type, etc.).
 *
 * @param[in]  tensor A tensor in interest
 * @param[out] size   The size of the tensor
 *
 * @return Status code
 *
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclGetTensorSize(AclTensor tensor, uint64_t *size);

/** Get the descriptor of this tensor
 *
 * @param[in]  tensor A tensor in interest
 * @param[out] desc   The descriptor of the tensor
 *
 * @return Status code
 *
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclGetTensorDescriptor(AclTensor tensor, AclTensorDescriptor *desc);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* ARM_COMPUTE_ACLUTILS_H_ */
