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
#ifndef ACL_SRC_CPU_OPERATORS_CPUSCATTER_H
#define ACL_SRC_CPU_OPERATORS_CPUSCATTER_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/function_info/ScatterInfo.h"

#include "src/cpu/ICpuKernel.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to execute Scatter in Neon ™ */
class CpuScatter : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * @note indices must always be U32
     * @note src, updates and dst tensors must be same datatype.
     *
     * @param[in]  src          Source input tensor info. Can be nullptr when using "Add" Scatter Function with zero initialization.
     * @param[in]  updates      Tensor info for tensor storing update values to use for scatter function. Data types supported: same as @p src.
     * @param[in]  indices      Tensor info for tensor storing indices to use for scatter function. Data types supported: U32 only.
     * @param[out] dst          Output tensor to store the result of the Scatter Function. Data types supported: same as @p src and @p updates.
     * @param[in]  Scatter_info Contains Scatter operation information described in @ref ScatterInfo.
     */
    void configure(const ITensorInfo *src,
                   const ITensorInfo *updates,
                   const ITensorInfo *indices,
                   ITensorInfo       *dst,
                   const ScatterInfo &Scatter_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuScatter::configure()
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
    std::unique_ptr<ICPPKernel> _scatter_kernel{nullptr};
    std::unique_ptr<ICPPKernel> _fill_kernel{nullptr};
};
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_OPERATORS_CPUSCATTER_H
