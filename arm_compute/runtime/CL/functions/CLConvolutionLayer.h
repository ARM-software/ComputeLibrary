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
#ifndef ARM_COMPUTE_CLCONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLCONVOLUTIONLAYER_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref opencl::ClGemmConv2d
 * -# @ref opencl::ClWinogradConv2d
 * -# @ref opencl::ClDirectConv2d
 * -# @ref CLFFTConvolutionLayer
 *
 * The function selects one of the algorithms mentioned above based on:
 *      - The size of the kernel
 *      - Number of input/output feature maps
 *      - Amount of memory needed
 *
 * Generally GEMM-based convolution is executed when neither Winograd nor FFT nor Direct convolution can be performed.
 *
 * FP32 Algorithm| Filter Size                                                 |   Input/Output feature maps               |
 * --------------|-------------------------------------------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5(fast maths) 7x1 1x7                 |  Input channels is greater than 3         |
 * FFT           | Squared kernels and greater than 9x9                        |  Input feature maps > Output feature maps |
 * DirectConv    | 9x9                                                         |                                           |
 * GEMM          | Any size                                                    |                                           |
 *
 * Winograd 5x5 requires fast maths enabled.
 *
 * FP16 Algorithm| Filter Size                |   Input/Output feature maps               |
 * --------------|----------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5    |  Input channels is greater than 3         |
 * FFT           | Not supported              |                                           |
 * DirectConv    | 9x9                        |                                           |
 * GEMM          | Any size                   |                                           |
 *
 * Winograd FP16 requires fast maths enabled.
 *
 */
class CLConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Default Destructor */
    ~CLConvolutionLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvolutionLayer(const CLConvolutionLayer &) = delete;
    /** Default move constructor */
    CLConvolutionLayer(CLConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvolutionLayer &operator=(const CLConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLConvolutionLayer &operator=(CLConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
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
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Same as @p input, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  post_ops         (Optional) A sequence of post operations that are performed after the main operation.
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1,
                   const experimental::PostOpList<ICLTensor *> &post_ops = experimental::PostOpList<ICLTensor *> {});
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context  The compile context to be used.
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Same as @p input, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  post_ops         (Optional) A sequence of post operations that are performed after the main operation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false,
                   unsigned int num_groups = 1, const experimental::PostOpList<ICLTensor *> &post_ops = experimental::PostOpList<ICLTensor *> {});
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Same as @p input, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in] post_ops         (Optional) A sequence of post operations that are performed after the main operation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false,
                           unsigned int num_groups = 1, const experimental::PostOpList<ITensorInfo *> &post_ops = experimental::PostOpList<ITensorInfo *> {});
    /** Static function to check if given info will return the convolution called by @ref CLConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] gpu_target       Specifies the @p GPUTarget.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     *
     * @return the Convolution Method Hint
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                    const WeightsInfo &weights_info, const ActivationLayerInfo &act_info, const GPUTarget gpu_target, const Size2D &dilation = Size2D(1U, 1U), bool enable_fast_math = false);
    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ARM_COMPUTE_CLCONVOLUTIONLAYER_H */
