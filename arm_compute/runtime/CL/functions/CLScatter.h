/*
 * Copyright (c) 2024 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLSCATTER_H
#define ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLSCATTER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;
class ITensorInfo;
struct ScatterInfo;
class CLCompileContext;

/** Function to compute ScatterND Layer */
class CLScatter : public IFunction
{
public:
    /** Default Constructor */
    CLScatter();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScatter(const CLScatter &) = delete;
    /** Default move constructor */
    CLScatter(CLScatter &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScatter &operator=(const CLScatter &) = delete;
    /** Default move assignment operator */
    CLScatter &operator=(CLScatter &&);
    /** Default destructor */
    ~CLScatter();
    /** Initialise the kernel's inputs and outputs
     *
     * @note Negative indices are treated as out of bounds.
     *
     * Valid data layouts:
     * - All
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor. Values used to fill output. Can be nullptr when zero initialization is true.
     * @param[in]  updates         Tensor containing values used to update output tensor. Data types supported: same as @p src
     * @param[in]  indices         Tensor containing Indices to change in the output Tensor. Data types supported : S32
     * @param[out] output          Destination tensor. Data types supported: same as @p src.
     * @param[in]  info            Scatter info object.
     */
    void configure(const CLCompileContext &compile_context,
                   const ICLTensor        *src,
                   const ICLTensor        *updates,
                   const ICLTensor        *indices,
                   ICLTensor              *output,
                   const ScatterInfo      &info);
    /** Initialise the kernel's inputs and output
     *
     * Similar to @ref CLScatter::configure()
     */
    void configure(const ICLTensor   *src,
                   const ICLTensor   *updates,
                   const ICLTensor   *indices,
                   ICLTensor         *output,
                   const ScatterInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLScatter
     *
     * @param[in] src     Source tensor.
     * @param[in] updates Tensor containing values used for updating the output Tensor. Data types supported : same as @p src
     * @param[in] indices Tensor containing Indices to change in the output Tensor. Data types supported : S32
     * @param[in] output  Destination tensor. Data types supported: same as @p src.
     * @param[in] info    Scatter info containing type of scatter.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *output,
                           const ScatterInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLSCATTER_H
