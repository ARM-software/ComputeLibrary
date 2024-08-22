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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMDIRECTCONV2D_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMDIRECTCONV2D_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/IOperator.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
/*
 * A shallow wrapper for arm_compute::cpu::CpuGemmDirectConv2d.
 * Any new features should be added to arm_compute::cpu::CpuGemmDirectConv2d and
 * arm_compute::experimental::op::CpuGemmDirectConv2d should remain a shallow wrapper.
*/
class CpuGemmDirectConv2d : public IOperator
{
public:
    /** Constructor **/
    CpuGemmDirectConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmDirectConv2d(const CpuGemmDirectConv2d &) = delete;
    /** Prevent copy assignment */
    CpuGemmDirectConv2d &operator=(const CpuGemmDirectConv2d &) = delete;
    /** Default move constructor */
    CpuGemmDirectConv2d(CpuGemmDirectConv2d &&) = default;
    /** Default move assignment */
    CpuGemmDirectConv2d &operator=(CpuGemmDirectConv2d &&) = default;
    /** Default destructor */
    ~CpuGemmDirectConv2d() override;

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |src2           |dst            |
     * |:--------------|:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |S32            |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |S32            |QASYMM8_SIGNED |
     * |F16            |F16            |F16            |F16            |
     * |F32            |F32            |F32            |F32            |
     * |BFLOAT16       |BFLOAT16       |BFLOAT16       |BFLOAT16       |
     *
     * @param[in] src     Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                    while every optional dimension from 4 and above represent a batch of inputs.
     *                    Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                    Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in] biases  Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                    Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] dst     Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                    Data types supported: Same as @p input.
     * @param[in] info    Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ITensorInfo *src,
                   const ITensorInfo *weights,
                   const ITensorInfo *biases,
                   ITensorInfo       *dst,
                   const Conv2dInfo  &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmDirectConv2d
     *
     * Similar to CpuGemmDirectConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src,
                           const ITensorInfo *weights,
                           const ITensorInfo *biases,
                           const ITensorInfo *dst,
                           const Conv2dInfo  &info);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMDIRECTCONV2D_H
