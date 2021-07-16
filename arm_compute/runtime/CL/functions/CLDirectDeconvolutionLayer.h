/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDIRECTDECONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLDIRECTDECONVOLUTIONLAYER_H

#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayerUpsample.h"
#include "arm_compute/runtime/CL/functions/CLReverse.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;
/** Function to run the deconvolution layer.
 *
 * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perform a 1x1
 * convolution pass. Input stride defines how many zeroes we should put between each element of the input and pad is the amount of padding.
 *
 *  The relation between input to output is as follows:
 *  \f[
 *       width\_output = (width\_input - 1) \cdot stride\_x - 2 \cdot padding\_x + kernel\_x
 *  \f]
 *  \f[
 *       height\_output = (height\_input - 1) \cdot stride\_y - 2 \cdot padding\_y + kernel\_y
 *  \f]
 *
 *  where:
 *      width_input is the size of the first input dimension.
 *      height_input is the size of the second input dimension.
 *      width_output is the size of the first output dimension.
 *      height_output is the size of the second output dimension.
 *      kernel_x and kernel_y are the convolution sizes in x and y.
 *      stride_x and stride_y is the input stride of the first and second dimension.
 *
 * The weights used by Deconvolution are supposed to be the same as the ones used for Convolution. Therefore, it will be necessary to use the weights in the
 * reverse order to perform an actual convolution. This is achieved by using @ref CLReverse.
 *
 * This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLDeconvolutionLayerUpsample
 * -# @ref CLConvolutionLayer
 *
 * And the following CPP kernels:
 * -# @ref CLReverse
 *
 */
class CLDirectDeconvolutionLayer : public IFunction
{
public:
    /** Constructor */
    CLDirectDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectDeconvolutionLayer(const CLDirectDeconvolutionLayer &) = delete;
    /** Default move constructor */
    CLDirectDeconvolutionLayer(CLDirectDeconvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectDeconvolutionLayer &operator=(const CLDirectDeconvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLDirectDeconvolutionLayer &operator=(CLDirectDeconvolutionLayer &&) = default;
    /** Set the input, weights, biases and output tensors.
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
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * @param[in,out] input        Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
     *                             Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]     weights      The 4d weights with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input or QSYMM8_PER_CHANNEL if @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]     bias         (Optional) The biases have one dimension.
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type
     * @param[out]    output       Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info         Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     * @param[in]     weights_info (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref opencl::kernels::ClWeightsReshapeKernel.
     *
     */
    void configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info, const WeightsInfo &weights_info = WeightsInfo());
    /** Set the input, weights, biases and output tensors.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] input           Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
     *                                Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]     weights         The 4d weights with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input or QSYMM8_PER_CHANNEL if @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]     bias            (Optional) The biases have one dimension.
     *                                Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type
     * @param[out]    output          Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info            Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     * @param[in]     weights_info    (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref opencl::kernels::ClWeightsReshapeKernel.
     *
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info,
                   const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLDirectDeconvolutionLayer
     *
     * @param[in] input        Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
     *                         Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in] weights      The 4d weights info with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input or QSYMM8_PER_CHANNEL if @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] bias         (Optional) The biases have one dimension.
     *                         Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type
     * @param[in] output       Output tensor info. The output has the same number of dimensions as the @p input.
     * @param[in] info         Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     * @param[in] weights_info (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref opencl::kernels::ClWeightsReshapeKernel.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &info,
                           const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                  _memory_group;
    CLDeconvolutionLayerUpsample _scale_f;
    CLConvolutionLayer           _conv_f;
    CLReverse                    _flip_weights;

    CLTensor   _scaled_output;
    ICLTensor *_original_weights;
    CLTensor   _weights_flipped;
    CLTensor   _flip_axis;

    bool _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLDECONVOLUTIONLAYER_H */
