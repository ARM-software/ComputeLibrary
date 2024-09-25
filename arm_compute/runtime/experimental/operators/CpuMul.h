/*
 * Copyright (c) 2016-2021, 2023-2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMUL_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMUL_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{
/** Wrapper class for CpuMul. For information on the functions,
 * see "src/cpu/operators/CpuMul.h"
*/
class CpuMul : public INEOperator
{
public:
    /** Constructor */
    CpuMul();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuMul(const CpuMul &) = delete;
    /** Default move constructor */
    CpuMul(CpuMul &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuMul &operator=(const CpuMul &) = delete;
    /** Default move assignment operator */
    CpuMul &operator=(CpuMul &&) = default;
    /** Default destructor */
    ~CpuMul() override;
    /** Initialise the kernel's inputs, dst and convertion policy.
     *
     * Similar to @ref NEPixelWiseMultiplication::configure()
     */
    void configure(ITensorInfo               *src1,
                   ITensorInfo               *src2,
                   ITensorInfo               *dst,
                   float                      scale,
                   ConvertPolicy              overflow_policy,
                   RoundingPolicy             rounding_policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref NEPixelWiseMultiplication::validate()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *src1,
                           const ITensorInfo         *src2,
                           const ITensorInfo         *dst,
                           float                      scale,
                           ConvertPolicy              overflow_policy,
                           RoundingPolicy             rounding_policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMUL_H
