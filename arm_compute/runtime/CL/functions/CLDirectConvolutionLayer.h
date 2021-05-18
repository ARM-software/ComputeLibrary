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
#ifndef ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute direct convolution function:
 */
class CLDirectConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    CLDirectConvolutionLayer();
    /** Destructor */
    ~CLDirectConvolutionLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectConvolutionLayer(const CLDirectConvolutionLayer &) = delete;
    /** Default move constructor */
    CLDirectConvolutionLayer(CLDirectConvolutionLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectConvolutionLayer &operator=(const CLDirectConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLDirectConvolutionLayer &operator=(CLDirectConvolutionLayer &&);
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
     * |QASYMM8        |QASYMM8        |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                       Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info  (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output          Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLDirectConvolutionLayer
     *
     * @param[in] input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs.
     *                      Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in] weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                      Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                      Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] act_info  (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYER_H */
