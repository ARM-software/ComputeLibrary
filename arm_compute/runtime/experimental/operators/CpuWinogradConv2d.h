/*
 * Copyright (c) 2021-2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUWINOGRADCONV2D_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUWINOGRADCONV2D_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/runtime/IOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{
/*
 * A shallow wrapper for arm_compute::cpu::CpuWinogradConv2d.
 * Any new features should be added to arm_compute::cpu::CpuWinogradConv2d and
 * arm_compute::experimental::op::CpuWinogradConv2d should remain a shallow wrapper.
*/
class CpuWinogradConv2d : public IOperator
{
public:
    /** Constructors */
    CpuWinogradConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2d(const CpuWinogradConv2d &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2d &operator=(const CpuWinogradConv2d &) = delete;
    /** Default move constructor */
    CpuWinogradConv2d(CpuWinogradConv2d &&) = default;
    /** Default move assignment */
    CpuWinogradConv2d &operator=(CpuWinogradConv2d &&) = default;

    /** Destructor */
    ~CpuWinogradConv2d() override;

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1           |src2   |dst            |
     * |:--------------|:--------------|:------|:--------------|
     * |F16            |F16            |F16    |F16            |
     * |F32            |F32            |F32    |F32            |
     *
     * @param[in]  src              Source tensor Info. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor Info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     *                              For supported kernel sizes, see @ref arm_compute::NEWinogradConvolutionLayer
     * @param[in]  biases           Biases tensor Info. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] dst              Destination tensor Info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(const ITensorInfo         *src,
                   const ITensorInfo         *weights,
                   const ITensorInfo         *biases,
                   ITensorInfo               *dst,
                   const PadStrideInfo       &conv_info,
                   const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                   bool                       enable_fast_math = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2d
     *
     * Similar to CpuWinogradConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *src,
                           const ITensorInfo         *weights,
                           const ITensorInfo         *biases,
                           const ITensorInfo         *dst,
                           const PadStrideInfo       &conv_info,
                           const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                           bool                       enable_fast_math = false);

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

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUWINOGRADCONV2D_H
