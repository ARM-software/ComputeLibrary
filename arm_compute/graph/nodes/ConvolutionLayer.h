/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__
#define __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** Convolution layer node */
class ConvolutionLayer : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] conv_width   Convolution width
     * @param[in] conv_height  Convolution height
     * @param[in] ofm          Output feature map
     * @param[in] weights      Weights of the convolution layer
     * @param[in] biases       Bias of the convolution layer
     * @param[in] conv_info    Convolution information
     * @param[in] weights_info Weights information
     */
    template <typename AccessorTypeWeights, typename AccessorTypeBiases>
    ConvolutionLayer(unsigned int conv_width, unsigned int conv_height, unsigned int ofm, AccessorTypeWeights &&weights,
                     AccessorTypeBiases &&biases, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo())
        : _conv_width(conv_width), _conv_height(conv_height), _ofm(ofm), _weights(std::move(weights)), _biases(std::move(biases)), _conv_info(conv_info), _weights_info(weights_info)
    {
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(Hint hint, ITensor *input, ITensor *output) override;
    void print_info() override;

private:
    unsigned int         _conv_width;   /**< Convolution width */
    unsigned int         _conv_height;  /**< Convolution height */
    unsigned int         _ofm;          /**< Output feature maps */
    Tensor               _weights;      /**< Weights tensor */
    Tensor               _biases;       /**< Biases tensor */
    const PadStrideInfo &_conv_info;    /**< Convolution layer information */
    const WeightsInfo   &_weights_info; /**< Convolution layer weights information */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__ */
