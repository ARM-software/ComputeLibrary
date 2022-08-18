/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMCONVOLUTIONLAYER_H
#define ARM_COMPUTE_NEGEMMCONVOLUTIONLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to compute the convolution layer. This function calls the following kernels/functions:
 *
 * -# @ref cpu::CpuGemmConv2d
 *
 */
class NEGEMMConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEGEMMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConvolutionLayer(const NEGEMMConvolutionLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMConvolutionLayer(NEGEMMConvolutionLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConvolutionLayer &operator=(const NEGEMMConvolutionLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMConvolutionLayer &operator=(NEGEMMConvolutionLayer &&) = delete;
    /** Default destructor */
    ~NEGEMMConvolutionLayer();
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
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                              tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer
     *
     * @param[in] input            Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in] biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] output           Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                           bool enable_fast_math = false, unsigned int num_groups = 1);

    /** Static function to check if there is an optimized version of
     * GEMM available for the input parameters.
     *
     * The method is intended to be used to find out the optimal
     * memory layout to be used for the weights tensor when running
     * variable weights execution.
     *
     * The user can query the database of optimised kernels in
     * arm_gemm by specifying one of the enumerations of
     * arm_compute::WeightFormat in the weight_format field of the input
     * parameter weights_info. In case of success, the method
     * writes the expected format in the output parameter
     * expected_weight_format. The expected_weight_format can than be
     * used in the configure method of the class for retrieving the
     * best optimal kernel.
     *
     * Use case one - query for a specific format:
     *
     *     WeightInfo weights_info(..., arm_compute::WeightFormat::OHWIo4, ...); // Set the value of the input query.
     *     if (NEGEMMConvolutionlayer::has_opt_impl(WeightFormat(), ...., weights_info, ...))
     *     {
     *       auto conv = std::unique_ptr<NEGEMMConvolutionlayer>();
     *       conv->configure(..., weights_info, ...);  // uses the same WeightFormat the user wanted originally, OHWYo4.
     *       conv->run(...);
     *     }
     *
     * Use case two - query for any format that would be optimal for the GEMM to execute:
     *
     *     WeightInfo weights_info(..., arm_compute::WeightFormat::ANY, ...); // Set the value of the input query.
     *     arm_compute::WeightFormat expected_wf;
     *     if (NEGEMMConvolutionlayer::has_opt_impl(expected_wf, ...., weights_info, ...))
     *     {
     *       auto conv = std::unique_ptr<NEGEMMConvolutionlayer>();
     *       // ... code to convert the layout of the weights tensor to the layout returned by has_opt_impl
     *       WeightInfo new_weights_info(..., expected_wf, ...); // Set the value of the WeightFormat returned by has_opt_impl.
     *       conv->configure(..., new_weights_info, ...);
     *       conv->run(...);
     *     }
     *
     * Notice that a GEMM configured with a WeightFormat other than
     * UNSPECIFIED will run GEMM with variable weights mode.
     *
     * @param[out] expected_weight_format The arm_compute::WeightFormat expected by the kernel.
     * @param[in]  src                    Source tensor info.
     * @param[in]  weights                Weights tensor info.
     * @param[in]  biases                 Biases tensor info. Shared biases supported.
     * @param[in]  dst                    Destination tensor info.
     * @param[in]  conv_info              Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info           (optional) Specifies additional configuration parameters for the weights of the GEMM computation.
     * @param[in]  dilation               (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info               (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported. And no activation (i.e. Linear) which is the default value.
     * @param[in]  enable_fast_math       (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *
     * @return a Status
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format, const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                               const PadStrideInfo &conv_info,
                               const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                               bool enable_fast_math = false);
    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEGEMMCONVOLUTIONLAYER_H */
