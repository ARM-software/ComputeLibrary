/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_OPERATORS_CPUFLATTEN_H
#define ACL_SRC_CPU_OPERATORS_CPUFLATTEN_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
class CpuReshape;
/** Basic function to flatten a given input */
class CpuFlatten : public ICpuOperator
{
public:
    /** Constructor */
    CpuFlatten();
    /** Destructor */
    ~CpuFlatten();
    /** Configure operator for a given list of arguments
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in] src Source tensor to flatten with at least 3 dimensions.
     *                The dimensions above the third will be interpreted as batches. Data types supported: All
     * @param[in] dst Destination tensor with shape [w*h*d, input_batches] where:
     *                            w = width input tensor, h = height input tensor and d = depth input tensor.
     *                            Data type supported: same as @p src
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuFlatten::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<CpuReshape> _reshape;
};
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_OPERATORS_CPUFLATTEN_H
