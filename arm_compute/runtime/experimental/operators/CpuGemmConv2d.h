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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMCONV2D_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMCONV2D_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{
/*
 * A shallow wrapper for arm_compute::cpu::CpuGemmConv2d.
 * Any new features should be added to arm_compute::cpu::CpuGemmConv2d and
 * arm_compute::experimental::op::CpuGemmConv2d should remain a shallow wrapper.
*/
class CpuGemmConv2d : public IOperator
{
public:
    /** Constructor */
    CpuGemmConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmConv2d(const CpuGemmConv2d &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CpuGemmConv2d(CpuGemmConv2d &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmConv2d &operator=(const CpuGemmConv2d &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CpuGemmConv2d &operator=(CpuGemmConv2d &&) = delete;
    /** Destructor */
    ~CpuGemmConv2d();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2     |dst            |
     * |:--------------|:------------------|:--------|:--------------|
     * |F16            |F16                |F16      |F16            |
     * |F32            |F32                |F32      |F32            |
     * |BFLOAT16       |BFLOAT16           |BFLOAT16 |BFLOAT16       |
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QASYMM8_SIGNED     |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     *
     * @param[in]  src              Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in]  weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in]  biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] dst              Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with CpuWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                              tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensorInfo         *src,
                   const ITensorInfo         *weights,
                   const ITensorInfo         *biases,
                   ITensorInfo               *dst,
                   const PadStrideInfo       &conv_info,
                   const WeightsInfo         &weights_info     = WeightsInfo(),
                   const Size2D              &dilation         = Size2D(1U, 1U),
                   const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                   bool                       enable_fast_math = false,
                   unsigned int               num_groups       = 1);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmConvolution::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *src,
                           const ITensorInfo         *weights,
                           const ITensorInfo         *biases,
                           const ITensorInfo         *output,
                           const PadStrideInfo       &conv_info,
                           const WeightsInfo         &weights_info     = WeightsInfo(),
                           const Size2D              &dilation         = Size2D(1U, 1U),
                           const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                           bool                       enable_fast_math = false,
                           unsigned int               num_groups       = 1);

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * The parameter list is the same as @ref NEGEMMConvolutionLayer::has_opt_impl
     *
     * @return a status.
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                               const ITensorInfo         *src,
                               const ITensorInfo         *weights,
                               const ITensorInfo         *biases,
                               const ITensorInfo         *output,
                               const PadStrideInfo       &conv_info,
                               const WeightsInfo         &weights_info     = WeightsInfo(),
                               const Size2D              &dilation         = Size2D(1U, 1U),
                               const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                               bool                       enable_fast_math = false);

    /** Update of quantization information at the run stage for convolution so that the quantization multipliers can be properly calculated.
     * Please @ref NEGEMMConvolutionLayer for a more in-depth explanation and example.
     *
     * @param[in] tensors Vector that contains the tensors to operate on.
     */
    void update_quantization_parameters(ITensorPack &tensors);

    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMMCONV2D_H
