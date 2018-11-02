/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEWINOGRADLAYER_H__
#define __ARM_COMPUTE_NEWINOGRADLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/functions/CPPPermute.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;
/** Basic function to simulate a convolution layer. This function calls the following NEON kernels:
 * -# @ref NEWinogradLayerTransformWeightsKernel (executed only once in the first call to the run() method )
 * -# @ref NEWinogradLayerTransformInputKernel
 * -# @ref NEWinogradLayerTransformOutputKernel
 * -# @ref NEWinogradLayerBatchedGEMMKernel
 * -# @ref CPPPermute (three times: weights, input and output)
 */
class NEWinogradLayer : public IFunction
{
public:
    /** Constructor */
    NEWinogradLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     *                       Currently only 3x3 kernels are supported.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run() override;

    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer
     *
     * @param[in] input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs.
     *                      Data types supported: F32.
     * @param[in] weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     *                      Currently only 3x3 kernels are supported.
     * @param[in] biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                      Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info);

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayer(const NEWinogradLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayer &operator=(const NEWinogradLayer &) = delete;

private:
    MemoryGroup                _memory_group;
    std::unique_ptr<INEKernel> _batched_gemm_kernel;
    std::unique_ptr<INEKernel> _transform_input_kernel;
    std::unique_ptr<INEKernel> _transform_output_kernel;
    std::unique_ptr<INEKernel> _transform_weights_kernel;

    CPPPermute     _permute_input;
    CPPPermute     _permute_weights;
    CPPPermute     _permute_output;
    Tensor         _input_workspace;
    Tensor         _output_workspace;
    Tensor         _kernel_storage;
    Tensor         _input_nhwc;
    Tensor         _output_nhwc;
    Tensor         _weights_hwio;
    const ITensor *_input;
    const ITensor *_weights;
    ITensor       *_output;
    bool           _reshaped_kernel;
};
}
#endif /* __ARM_COMPUTE_NEWINOGRADLAYER_H__ */
