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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUACTIVATION_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUACTIVATION_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
/** Wrapper class for CpuActivation. For information on the functions,
 * see "src/cpu/operators/CpuActivation.h"
*/
class CpuActivation : public INEOperator
{
public:
    /** Constructor **/
    CpuActivation();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuActivation(const CpuActivation &) = delete;
    /** Prevent copy assignment */
    CpuActivation &operator=(const CpuActivation &) = delete;
    /** Default move constructor */
    CpuActivation(CpuActivation &&) = default;
    /** Default move assignment */
    CpuActivation &operator=(CpuActivation &&) = default;
    /** Default destructor */
    ~CpuActivation() override;

    /** Configure operator for a given list of arguments
     *
     * @param[in]  src      Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out] dst      Destination tensor info. Data type supported: same as @p src
     * @param[in]  act_info Activation layer parameters.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const ActivationLayerInfo &act_info);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuActivation::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info);

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUACTIVATION_H
