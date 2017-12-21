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
#ifndef __ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_H__
#define __ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_H__

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/SubTensor.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Convolution layer node */
class DepthwiseConvolutionLayer final : public INode
{
public:
    /** Default constructor
     *
     * @param[in] conv_width  Convolution width
     * @param[in] conv_height Convolution height
     * @param[in] weights     Weights values tensor
     * @param[in] biases      Biases values tensor
     * @param[in] conv_info   Convolution info
     * @param[in] opt3x3      (Optional) If true executes DepthwiseConvolutionLayer3x3
     * @param[in] quant_info  (Optional) Quantization info used for weights
     */
    template <typename AccessorType>
    DepthwiseConvolutionLayer(unsigned int conv_width, unsigned int conv_height, AccessorType &&weights, AccessorType &&biases, const PadStrideInfo conv_info, bool opt3x3 = true,
                              const QuantizationInfo quant_info = QuantizationInfo())
        : _conv_width(conv_width), _conv_height(conv_height), _weights(std::move(weights)), _biases(std::move(biases)), _conv_info(conv_info), _opt3x3(opt3x3), _quant_info(std::move(quant_info))
    {
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output) override;

private:
    unsigned int           _conv_width;
    unsigned int           _conv_height;
    Tensor                 _weights;
    Tensor                 _biases;
    const PadStrideInfo    _conv_info;
    bool                   _opt3x3;
    const QuantizationInfo _quant_info;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_H__ */
