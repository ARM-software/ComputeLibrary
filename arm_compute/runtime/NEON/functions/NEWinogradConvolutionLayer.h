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
#ifndef ARM_COMPUTE_NEWINOGRADCONVOLUTIONLAYER_H
#define ARM_COMPUTE_NEWINOGRADCONVOLUTIONLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/functions/CPPPermute.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ICPPKernel;

/** Basic function to simulate a convolution layer. This function calls the following Neon kernels:
 * -# @ref NEWinogradLayerTransformWeightsKernel (executed only once in the first call to the run() method )
 * -# @ref NEWinogradLayerTransformInputKernel
 * -# @ref NEWinogradLayerTransformOutputKernel
 * -# @ref NEGEMMAssemblyDispatch
 * -# @ref CPPPermute (three times: weights, input and output)
 *
 * @note  Some Winograd configurations (i.e. F(2x2, 5x5), F(4x4, 5x5)) are supported only with enable_fast_math = true
 */
class NEWinogradConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEWinogradConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager = nullptr);
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEWinogradConvolutionLayer(NEWinogradConvolutionLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEWinogradConvolutionLayer &operator=(NEWinogradConvolutionLayer &&) = delete;
    /** Default destructor */
    ~NEWinogradConvolutionLayer() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     *                              Currently only 3x3 and 5x5 kernels are supported.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                   bool enable_fast_math = false);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     *                             Currently only 3x3 and 5x5 kernels are supported.
     * @param[in] biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradConvolutionLayer(const NEWinogradConvolutionLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradConvolutionLayer &operator=(const NEWinogradConvolutionLayer &) = delete;

private:
    MemoryGroup                 _memory_group;
    NEGEMM                      _gemm_function;
    std::unique_ptr<ICPPKernel> _transform_input_kernel;
    std::unique_ptr<ICPPKernel> _transform_output_kernel;
    std::unique_ptr<ICPPKernel> _transform_weights_kernel;
    NEActivationLayer           _activationlayer_function;

    CPPPermute     _permute_input;
    CPPPermute     _permute_weights;
    CPPPermute     _permute_output;
    Tensor         _input_transformed;
    Tensor         _output_transformed;
    Tensor         _input_workspace;
    Tensor         _output_workspace;
    Tensor         _kernel_storage;
    Tensor         _input_nhwc;
    Tensor         _output_nhwc;
    Tensor         _weights_hwio;
    const ITensor *_input;
    const ITensor *_weights;
    ITensor       *_output;
    bool           _is_prepared;
    bool           _is_activationlayer_enabled;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEWINOGRADCONVOLUTIONLAYER_H */
