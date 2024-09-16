/*
 * Copyright (c) 2017-2021, 2023-2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMLOWP_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMLOWP_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/function_info/GEMMInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

namespace experimental
{
namespace op
{
/*
 * A shallow wrapper for arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore.
 * Any new features should be added to arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore and
 * arm_compute::experimental::op::CpuGEMMLowp should remain a shallow wrapper.
*/
class CpuGEMMLowp : public INEOperator
{
public:
    /** Constructor */
    CpuGEMMLowp();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGEMMLowp(const CpuGEMMLowp &) = delete;
    /** Default move constructor */
    CpuGEMMLowp(CpuGEMMLowp &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGEMMLowp &operator=(const CpuGEMMLowp &) = delete;
    /** Default move assignment operator */
    CpuGEMMLowp &operator=(CpuGEMMLowp &&) = default;
    /** Default destructor */
    ~CpuGEMMLowp();
    /** Initialise the kernel's inputs, output
     *
     *valid configurations can be referenced in @ref arm_compute::NEGEMMLowpMatrixMultiplyCore.
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *output,
                   const GEMMInfo    &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGEMMLowp
     *
     * Similar to @ref CpuGEMMLowp::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           const ITensorInfo *output,
                           const GEMMInfo    &gemm_info = GEMMInfo());

    // Inherited methods overridden
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;

    experimental::MemoryRequirements workspace() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMLOWP_H
