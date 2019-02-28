/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLDECONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLDECONVOLUTIONLAYER_H__

#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayerUpsample.h"

#include "arm_compute/core/CPP/kernels/CPPFlipWeightsKernel.h"

#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;
/** Function to run the deconvolution layer.
 *
 * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perform a 1x1
 * convolution pass. Input stride defines how many zeroes we should put between each element of the input, pad is the amount of padding and finally a is a user
 * specified value where a < stride - 1, that increases the padding top and right of the input image.
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
 * reverse order to perform an actual convolution. This is achieved by using the @ref CPPFlipWeightsKernel.
 *
 * This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLDeconvolutionLayerUpsample
 * -# @ref CLConvolutionLayer
 *
 */
class CLDeconvolutionLayer : public IFunction
{
public:
    /** Constructor */
    CLDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayer(const CLDeconvolutionLayer &) = delete;
    /** Default move constructor */
    CLDeconvolutionLayer(CLDeconvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayer &operator=(const CLDeconvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLDeconvolutionLayer &operator=(CLDeconvolutionLayer &&) = default;
    /** Set the input, weights, biases and output tensors.
     *
     * @deprecated This method is deprecated and will be removed in release 19.05
     *
     * @param[in,out] input              Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: QASYMM8/F16/F32.
     * @param[in]     weights            The 4d weights with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]     bias               (Optional) The biases have one dimension. Data type supported: Same as @p input.
     * @param[out]    output             Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info               Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in]     inner_border_right The number of zeros added to right edge of the input.
     * @param[in]     inner_border_top   The number of zeros added to top edge of the input.
     * @param[in]     weights_info       (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref CLWeightsReshapeKernel.
     *
     */
    void configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info,
                   unsigned int inner_border_right, unsigned int inner_border_top, const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayer
     *
     * @deprecated This method is deprecated and will be removed in release 19.05
     *
     * @param[in] input              Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: QASYMM8/F16/F32.
     * @param[in] weights            The 4d weights info with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in] bias               (Optional) The biases have one dimension. Data type supported: Same as @p input.
     * @param[in] output             Output tensor info. The output has the same number of dimensions as the @p input.
     * @param[in] info               Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in] inner_border_right The number of zeros added to right edge of the input.
     * @param[in] inner_border_top   The number of zeros added to top edge of the input.
     * @param[in] weights_info       (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref CLWeightsReshapeKernel.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &info,
                           unsigned int inner_border_right, unsigned int inner_border_top, const WeightsInfo &weights_info = WeightsInfo());

    /** Set the input, weights, biases and output tensors.
     *
     * @param[in,out] input        Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: QASYMM8/F16/F32.
     * @param[in]     weights      The 4d weights with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]     bias         (Optional) The biases have one dimension. Data type supported: Same as @p input.
     * @param[out]    output       Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info         Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in]     weights_info (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref CLWeightsReshapeKernel.
     *
     */
    void configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info, const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayer
     *
     * @param[in] input        Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: QASYMM8/F16/F32.
     * @param[in] weights      The 4d weights info with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in] bias         (Optional) The biases have one dimension. Data type supported: Same as @p input.
     * @param[in] output       Output tensor info. The output has the same number of dimensions as the @p input.
     * @param[in] info         Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in] weights_info (Optional) Weights information needed for @ref CLConvolutionLayer, specifies if the weights tensor has been reshaped with @ref CLWeightsReshapeKernel.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &info, const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLMemoryGroup                _memory_group;
    CLDeconvolutionLayerUpsample _scale_f;
    CLConvolutionLayer           _conv_f;
    CPPFlipWeightsKernel         _flip_weights;
    CLTensor                     _scaled_output;
    ICLTensor                   *_original_weights;
    CLTensor                     _weights_flipped;
    bool                         _is_prepared;
};
}
#endif /* __ARM_COMPUTE_CLDECONVOLUTIONLAYER_H__ */
