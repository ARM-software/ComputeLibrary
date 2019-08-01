/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLWINOGRADCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLWINOGRADCONVOLUTIONLAYER_H__

#include "arm_compute/core/CL/kernels/CLWinogradFilterTransformKernel.h"
#include "arm_compute/core/CL/kernels/CLWinogradOutputTransformKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLWinogradInputTransform.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute Winograd-based convolution on OpenCL. This function calls the following OpenCL functions/kernels:
 *
 *  -# @ref CLWinogradInputTransform
 *  -# @ref CLWinogradFilterTransformKernel (only once)
 *  -# @ref CLGEMM
 *  -# @ref CLWinogradOutputTransformKernel
 *
 */
class CLWinogradConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLWinogradConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradConvolutionLayer(const CLWinogradConvolutionLayer &) = delete;
    /** Default move constructor */
    CLWinogradConvolutionLayer(CLWinogradConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradConvolutionLayer &operator=(const CLWinogradConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLWinogradConvolutionLayer &operator=(CLWinogradConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @note: This function only works with 3x3,3x1,1x3,5x5,5x1,1x5,7x1 and 1x7 kernels along with unit strides for both NCHW and NHWC data layout
     * @note  Some Winograd configurations (i.e. F(4x4, 5x5)) are supported only with enable_fast_math = true
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CLWinogradConvolutionLayer
     *
     * @note: This function only works with 3x3,3x1,1x3,5x5,5x1 and 1x5 kernels along with unit strides for both NCHW and NHWC data layout
     * @note  Some Winograd configurations (i.e. F(4x4, 5x5)) are supported only with enable_fast_math = true
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLMemoryGroup                   _memory_group;
    CLGEMM                          _batched_mm;
    CLWinogradInputTransform        _input_transform;
    CLWinogradFilterTransformKernel _filter_transform;
    CLWinogradOutputTransformKernel _output_transform;
    CLTensor                        _input0;
    CLTensor                        _input1;
    CLTensor                        _batched_mm_output;
    const ICLTensor                *_original_weights;
    bool                            _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLWINOGRADCONVOLUTIONLAYER_H__ */
