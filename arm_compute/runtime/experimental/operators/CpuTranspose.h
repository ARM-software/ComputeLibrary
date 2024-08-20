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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUTRANSPOSE_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUTRANSPOSE_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
/** Wrapper class for CpuTranspose. For information on the functions,
 * see "src/cpu/operators/CpuTranspose.h"
*/
class CpuTranspose : public INEOperator
{
public:
    /** Constructor **/
    CpuTranspose();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuTranspose(const CpuTranspose &) = delete;
    /** Prevent copy assignment */
    CpuTranspose &operator=(const CpuTranspose &) = delete;
    /** Default move constructor */
    CpuTranspose(CpuTranspose &&) = default;
    /** Default move assignment */
    CpuTranspose &operator=(CpuTranspose &&) = default;
    /** Default destructor */
    ~CpuTranspose() override;

    /** Configure kernel for a given list of arguments
     *
     * @param[in]  src Source tensor to permute. Data types supported: All
     * @param[out] dst Destination tensor. Data types supported: Same as @p src
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuTransposeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUTRANSPOSE_H
