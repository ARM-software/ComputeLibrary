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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUQUANTIZE_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUQUANTIZE_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include "src/cpu/ICpuOperator.h"

#include <memory>

/*
 * A shallow wrapper for arm_compute::cpu::CpuQuantize.
 * Any new features should be added to arm_compute::cpu::CpuQuantize and
 * arm_compute::experimental::op::CpuQuantize should remain a shallow wrapper.
*/
namespace arm_compute
{
namespace experimental
{
namespace op
{

/** A simple wrapper class which runs cpu::CpuQuantize */
class CpuQuantize : public arm_compute::experimental::INEOperator
{
public:
    CpuQuantize();
    /** Default Destructor */
    ~CpuQuantize();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuQuantize(const CpuQuantize &) = delete;
    /** Default move constructor */
    CpuQuantize(CpuQuantize &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuQuantize &operator=(const CpuQuantize &) = delete;
    /** Default move assignment operator */
    CpuQuantize &operator=(CpuQuantize &&) = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src                |dst                                    |
     * |:------------------|:--------------------------------------|
     * |QASYMM8            |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
     * |QASYMM8_SIGNED     |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
     * |F16                |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
     * |F32                |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
     *
     * @param[in]  input  Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
     * @param[out] output Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16
     */
    void configure(const ITensorInfo *input, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuQuantize
     *
     * Similar to @ref CpuQuantize::configure()
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
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUQUANTIZE_H
