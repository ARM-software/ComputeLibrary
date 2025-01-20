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
#ifndef ACL_SRC_CPU_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H
#define ACL_SRC_CPU_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/ITensorPack.h"

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
class CpuMeanStdDevNormalization : public ICpuOperator
{
public:
    /** Initialise the kernel's input and outputs.
     *
     * @note If the output tensor info is a nullptr, the normalization will be performed in-place.
     *
     * @param[in, out] input   Source tensor info with 2 dimensions. In case of @p output tensor info = nullptr,
     *                         this tensor will store the result of the normalization. Data types supported: F16/F32.
     * @param[out]     output  (Optional) Destination tensor info. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in]      epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     */
    void configure(ITensorInfo *input, ITensorInfo *output = nullptr, float epsilon = 1e-8f);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuMeanStdDevNormalizationKernel
     *
     * @param[in] input   Source tensor info with 2 dimensions. In case of @p output tensor info = nullptr,
     *                    this tensor will store the result of the normalization. Data types supported: F16/F32.
     * @param[in] output  (Optional) Destination tensor info. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in] epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output = nullptr, float epsilon = 1e-8f);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::string _name{};
};
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_OPERATORS_CPUMEANSTDDEVNORMALIZATION_H
