/*
 * Copyright (c) 2025 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H

/** @file
 * @publicapi
 */

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
class CpuMeanStdDevNormalization : public INEOperator
{
public:
    /** Default Constructor */
    CpuMeanStdDevNormalization();
    /** Default Destructor */
    ~CpuMeanStdDevNormalization();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuMeanStdDevNormalization(const CpuMeanStdDevNormalization &) = delete;
    /** Default move constructor */
    CpuMeanStdDevNormalization(CpuMeanStdDevNormalization &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuMeanStdDevNormalization &operator=(const CpuMeanStdDevNormalization &) = delete;
    /** Default move assignment operator */
    CpuMeanStdDevNormalization &operator=(CpuMeanStdDevNormalization &&) = default;
    /** Configure the kernel.
     *
     * Valid configurations and data layouts can be referenced in @ref arm_compute::NEMeanStdDevNormalizationLayer.
     */
    void configure(ITensorInfo *input, ITensorInfo *output = nullptr, float epsilon = 1e-8f);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuMeanStdDevNormalization
     *
     * Similar to @ref CpuMeanStdDevNormalization::configure
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output = nullptr, float epsilon = 1e-8f);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H
