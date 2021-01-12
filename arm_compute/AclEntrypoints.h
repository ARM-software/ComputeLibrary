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
#ifndef ARM_COMPUTE_ACLENTRYPOINTS_H_
#define ARM_COMPUTE_ACLENTRYPOINTS_H_

#include "arm_compute/AclTypes.h"

#ifdef __cplusplus
extern "C" {
#endif /** __cplusplus */

/** Create a context object
 *
 * Context is responsible for retaining internal information and work as an aggregate service mechanism
 *
 * @param[in, out] ctx     A valid non-zero context object if no failure occurs
 * @param[in]      target  Target to create the context for
 * @param[in]      options Context options to be used for all the kernels that are created under the context
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclCreateContext(AclContext              *ctx,
                           AclTarget                target,
                           const AclContextOptions *options);

/** Destroy a given context object
 *
 * @param[in] ctx A valid context object to destroy
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided context is invalid
 */
AclStatus AclDestroyContext(AclContext ctx);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACLENTRYPOINTS_H_ */
