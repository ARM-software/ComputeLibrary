/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEQUANTIZE_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEQUANTIZE_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

/*
 * A shallow wrapper for arm_compute::cpu::CpuDequantize.
 * Any new features should be added to arm_compute::cpu::CpuDequantize and
 * arm_compute::experimental::op::CpuDequantize should remain a shallow wrapper.
*/
namespace arm_compute
{
namespace experimental
{
namespace op
{
/** A simple wrapper class which runs cpu::CpuDequantize that dequantizes an input tensor */
class CpuDequantize : public INEOperator
{
public:
    /** Default Constructor */
    CpuDequantize();
    /** Default Destructor */
    ~CpuDequantize();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuDequantize(const CpuDequantize &) = delete;
    /** Default move constructor */
    CpuDequantize(CpuDequantize &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuDequantize &operator=(const CpuDequantize &) = delete;
    /** Default move assignment operator */
    CpuDequantize &operator=(CpuDequantize &&) = default;
    /** Configure the kernel.
     *
     * Valid configurations and data layouts can be referenced in @ref arm_compute::NEDequantizationLayer.
     */
    void configure(const ITensorInfo *input, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuDequantize
     *
     * Similar to @ref CpuDequantize::configure
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEQUANTIZE_H
