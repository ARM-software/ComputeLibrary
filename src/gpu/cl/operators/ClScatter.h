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

#ifndef ACL_SRC_GPU_CL_OPERATORS_CLSCATTER_H
#define ACL_SRC_GPU_CL_OPERATORS_CLSCATTER_H

#include "arm_compute/function_info/ScatterInfo.h"

#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
// Forward declaration
class ClFillKernel;
class ClScatterKernel;
class ClCopyKernel;

/** Basic operator to execute Scatter on OpenCL. This operator calls the following OpenCL kernels:
 *
 *  -# @ref kernels::ClScatterKernel
 */
class ClScatter : public IClOperator
{
public:
    /** Constructor */
    ClScatter();
    /** Default destructor */
    ~ClScatter() = default;
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * @note indices must always be S32.
     * @note Negative indices are treated as out of bounds.
     * @note src, updates and dst tensors must be same datatype.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source input tensor info. Can be nullptr when using "Add" Scatter Function with zero initialization.
     * @param[in]  updates         Tensor info for tensor storing update values to use for scatter function. Data types supported: same as @p src.
     * @param[in]  indices         Tensor info for tensor storing indices to use for scatter function. Data types supported: S32 only.
     * @param[out] dst             Output tensor to store the result of the Scatter Function. Data types supported: same as @p src and @p updates.
     * @param[in]  Scatter_info    Contains Scatter operation information described in @ref ScatterInfo.
     */
    void configure(const CLCompileContext &compile_context,
                   const ITensorInfo      *src,
                   const ITensorInfo      *updates,
                   const ITensorInfo      *indices,
                   ITensorInfo            *dst,
                   const ScatterInfo      &Scatter_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClScatter::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *dst,
                           const ScatterInfo &Scatter_info);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<opencl::IClKernel> _scatter_kernel{nullptr};
    std::unique_ptr<opencl::IClKernel> _fill_kernel{nullptr};
    std::unique_ptr<opencl::IClKernel> _copy_kernel{nullptr};
    bool                               _fill_zero{false};
    bool                               _run_copy{false};
};
} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_OPERATORS_CLSCATTER_H
