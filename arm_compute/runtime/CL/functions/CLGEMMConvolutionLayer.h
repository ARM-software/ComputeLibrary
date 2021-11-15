/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H

#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref opencl::ClGemmConv2d
 */
class CLGEMMConvolutionLayer : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] memory_manager  (Optional) Memory manager.
     * @param[in] weights_manager (Optional) Weights manager.
     */
    CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer(const CLGEMMConvolutionLayer &) = delete;
    /** Default move constructor */
    CLGEMMConvolutionLayer(CLGEMMConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer &operator=(const CLGEMMConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLGEMMConvolutionLayer &operator=(CLGEMMConvolutionLayer &&) = default;
    /**Default destructor */
    ~CLGEMMConvolutionLayer();
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
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  post_ops     (Optional) A sequence of post operations that are performed after the main operation.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1,
                   const experimental::PostOpList<ICLTensor *> &post_ops = experimental::PostOpList<ICLTensor *> {});
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output          Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info    Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  post_ops        (Optional) A sequence of post operations that are performed after the main operation.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1,
                   const experimental::PostOpList<ICLTensor *> &post_ops = experimental::PostOpList<ICLTensor *> {});
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  post_ops     (Optional) A sequence of post operations that are performed after the main operation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1,
                           const experimental::PostOpList<ITensorInfo *> &post_ops = experimental::PostOpList<ITensorInfo *> {});

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H */
