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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEPTHWISECONV2D_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEPTHWISECONV2D_H

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
 * A shallow wrapper for arm_compute::cpu::CpuDepthwiseConv2d.
 * Any new features should be added to arm_compute::cpu::CpuDepthwiseConv2d and
 * arm_compute::experimental::op::CpuDepthwiseConv2d should remain a shallow wrapper.
*/
class CpuDepthwiseConv2d : public IOperator
{
public:
    /** Constructor **/
    CpuDepthwiseConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuDepthwiseConv2d(const CpuDepthwiseConv2d &) = delete;
    /** Prevent copy assignment */
    CpuDepthwiseConv2d &operator=(const CpuDepthwiseConv2d &) = delete;
    /** Default move constructor */
    CpuDepthwiseConv2d(CpuDepthwiseConv2d &&) = default;
    /** Default move assignment */
    CpuDepthwiseConv2d &operator=(CpuDepthwiseConv2d &&) = default;
    /** Default destructor */
    ~CpuDepthwiseConv2d() override;

    /** Initialize the function's source, destination, weights and convolution information.
     *
     * Valid data layouts:
     * - NHWC
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(ITensorInfo               *input,
                   const ITensorInfo         *weights,
                   const ITensorInfo         *biases,
                   ITensorInfo               *output,
                   const PadStrideInfo       &conv_info,
                   unsigned int               depth_multiplier = 1,
                   const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                   const Size2D              &dilation         = Size2D(1, 1));

    /** Static function to check if given info will lead to a valid configuration for @ref CpuDepthwiseConv2d::configure()
     *
     * Similar to @ref CpuDepthwiseConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *input,
                           const ITensorInfo         *weights,
                           const ITensorInfo         *biases,
                           const ITensorInfo         *output,
                           const PadStrideInfo       &conv_info,
                           unsigned int               depth_multiplier = 1,
                           const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                           const Size2D              &dilation         = Size2D(1U, 1U));

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

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUDEPTHWISECONV2D_H
