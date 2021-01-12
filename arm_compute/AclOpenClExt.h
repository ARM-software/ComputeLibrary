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
#ifndef ARM_COMPUTE_ACLOPENCLEXT_H_
#define ARM_COMPUTE_ACLOPENCLEXT_H_

#include "arm_compute/AclTypes.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif /* CL_TARGET_OPENCL_VERSION */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include "include/CL/cl.h"
#pragma GCC diagnostic pop

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Extract the underlying OpenCL context used by a given Compute Library context object
 *
 * @note @ref AclContext should be of an OpenCL backend target
 * @note @ref AclContext refcount should be 0, meaning not used by other objects
 *
 * @param[in]  ctx            A valid non-zero context
 * @param[out] opencl_context Underlying OpenCL context used
 *
 * @return Status code
 */
AclStatus AclGetClContext(AclContext ctx, cl_context *opencl_context);

/** Set the underlying OpenCL context used by a given Compute Library context object
 *
 * @note @ref AclContext should be of an OpenCL backend target
 *
 * @param[in]  ctx            A valid non-zero context object
 * @param[out] opencl_context Underlying OpenCL context to be used
 *
 * @return Status code
 */
AclStatus AclSetClContext(AclContext ctx, cl_context opencl_context);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACLOPENCLEXT_H_ */
