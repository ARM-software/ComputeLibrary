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
#ifndef __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__
#define __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__

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
class ConvolutionLayer final : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] conv_width         Convolution width
     * @param[in] conv_height        Convolution height
     * @param[in] ofm                Output feature map
     * @param[in] weights            Weights of the convolution layer
     * @param[in] biases             Bias of the convolution layer
     * @param[in] conv_info          Convolution information
     * @param[in] num_groups         (Optional) Number of groups, default = 1
     * @param[in] weights_info       (Optional) Weights information
     * @param[in] weights_quant_info (Optional) Weights quantization information
     * @param[in] out_quant_info     (Optional) Output quantization info
     */
    template <typename AccessorTypeWeights, typename AccessorTypeBiases>
    ConvolutionLayer(unsigned int           conv_width,
                     unsigned int           conv_height,
                     unsigned int           ofm,
                     AccessorTypeWeights &&weights,
                     AccessorTypeBiases   &&biases,
                     const PadStrideInfo    conv_info,
                     unsigned int           num_groups         = 1,
                     const WeightsInfo      weights_info       = WeightsInfo(),
                     const QuantizationInfo weights_quant_info = QuantizationInfo(),
                     const QuantizationInfo out_quant_info     = QuantizationInfo())
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _ofm(ofm),
          _weights(std::move(weights)),
          _biases(std::move(biases)),
          _conv_info(std::move(conv_info)),
          _num_groups(num_groups),
          _weights_info(std::move(weights_info)),
          _weights_quant_info(std::move(weights_quant_info)),
          _out_quant_info(std::move(out_quant_info)),
          _is(nullptr),
          _os(nullptr),
          _ws(nullptr),
          _bs(nullptr)
    {
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output) override;

private:
    /** Instantiates a non-grouped convolution
     *
     * @param[in] input            Input tensor
     * @param[in] output           Output tensor
     * @param[in] conv_method_hint Hint that specifies which convolution layer method to use
     *
     * @return Convolution function
     */
    std::unique_ptr<arm_compute::IFunction> instantiate_convolution(ITensor *input, ITensor *output, ConvolutionMethodHint conv_method_hint);
    /** Instantiates a grouped convolution
     *
     * @param[in] input            Input tensor
     * @param[in] output           Output tensor
     * @param[in] conv_method_hint Hint that specifies which convolution layer method to use
     *
     * @return Grouped Convolution function
     */
    std::unique_ptr<arm_compute::IFunction> instantiate_grouped_convolution(ITensor *input, ITensor *output, ConvolutionMethodHint conv_method_hint);

private:
    unsigned int           _conv_width;         /**< Convolution width */
    unsigned int           _conv_height;        /**< Convolution height */
    unsigned int           _ofm;                /**< Output feature maps */
    Tensor                 _weights;            /**< Weights tensor */
    Tensor                 _biases;             /**< Biases tensor */
    const PadStrideInfo    _conv_info;          /**< Convolution layer information */
    unsigned int           _num_groups;         /**< Number of groups */
    const WeightsInfo      _weights_info;       /**< Convolution layer weights information */
    const QuantizationInfo _weights_quant_info; /**< Output quantization information */
    const QuantizationInfo _out_quant_info;     /**< Output quantization information */

    std::unique_ptr<SubTensor[]> _is; /**< Input tensor sub-tensors used for grouped convolution */
    std::unique_ptr<SubTensor[]> _os; /**< Output tensor sub-tensors used for grouped convolution */
    std::unique_ptr<SubTensor[]> _ws; /**< Weights tensor sub-tensors used for grouped convolution */
    std::unique_ptr<SubTensor[]> _bs; /**< Biases tensor sub-tensors used for grouped convolution */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_H__ */
