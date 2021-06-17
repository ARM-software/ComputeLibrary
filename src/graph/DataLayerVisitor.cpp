/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/graph/DataLayerVisitor.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/nodes/Nodes.h"

namespace arm_compute
{
namespace graph
{
namespace
{
template <typename T>
void add_convolution_layer_data(DataLayerVisitor::LayerData &layer_data, T &node)
{
    PadStrideInfo ps_info = node.convolution_info();
    DataLayout    layout  = node.output(0)->desc().layout;
    // Add data layout
    layer_data["data_layout"] = to_string(layout);
    // Add padding info
    std::ostringstream padding;
    padding << "[" << to_string(ps_info.pad_left()) << ","
            << to_string(ps_info.pad_top()) << ","
            << to_string(ps_info.pad_bottom()) << ","
            << to_string(ps_info.pad_right()) << "]";

    layer_data["pad"] = padding.str();

    // Add stride info
    std::ostringstream stride;
    stride << "[" << to_string(ps_info.stride().first) << ","
           << to_string(ps_info.stride().second) << "]";

    layer_data["stride"] = stride.str();

    // Add dilation info
    // graph api does not support dilation > 1
    layer_data["dilation"] = "[1,1]";

    // Add bias enabled?
    // Assumes three inputs (input, weights, bias)
    std::string bias_enabled   = node.input(2) == nullptr ? "0" : "1";
    layer_data["bias_enabled"] = bias_enabled;

    // Change input names for weights / bias (if applicable)
    // Assumes input(1) is weights and input(2) is bias
    if(layer_data.count("input_shape1"))
    {
        layer_data["weights_shape"] = layer_data["input_shape1"];
        layer_data.erase("input_shape1");
    }
    if(layer_data.count("input_shape2"))
    {
        layer_data["bias_shape"] = layer_data["input_shape2"];
        layer_data.erase("input_shape2");
    }
}

template <typename T>
void add_convolution_layer_method(DataLayerVisitor::LayerData &layer_data, T &node)
{
    std::ostringstream method;
    method << node.convolution_method();
    layer_data["convolution_method"] = method.str();
}

template <typename T>
void add_generic_layer_data(DataLayerVisitor::LayerData &layer_data, T &node)
{
    // Add layer name
    layer_data["layer_name"] = node.name();
    // Loop over each input tensor
    for(size_t tensor_no = 0; tensor_no < node.num_inputs(); ++tensor_no)
    {
        // Add input tensor shapes
        if(node.input(tensor_no) != nullptr)
        {
            layer_data["input_shape" + to_string(tensor_no)] = "[" + to_string(node.input(tensor_no)->desc().shape) + "]";
        }
    }
    // Add output tensor shape
    if(node.output(0) != nullptr)
    {
        layer_data["output_shape0"] = "[" + to_string(node.output(0)->desc().shape) + "]";
    }
}
} // namespace

void DataLayerVisitor::visit(ConvolutionLayerNode &n)
{
    _layer_data.clear();
    add_generic_layer_data<ConvolutionLayerNode>(_layer_data, n);
    add_convolution_layer_data<ConvolutionLayerNode>(_layer_data, n);
    add_convolution_layer_method<ConvolutionLayerNode>(_layer_data, n);
}

void DataLayerVisitor::visit(DepthwiseConvolutionLayerNode &n)
{
    _layer_data.clear();
    add_generic_layer_data<DepthwiseConvolutionLayerNode>(_layer_data, n);
    add_convolution_layer_data<DepthwiseConvolutionLayerNode>(_layer_data, n);
}

void DataLayerVisitor::visit(FusedConvolutionBatchNormalizationNode &n)
{
    _layer_data.clear();
    add_generic_layer_data<FusedConvolutionBatchNormalizationNode>(_layer_data, n);
    add_convolution_layer_data<FusedConvolutionBatchNormalizationNode>(_layer_data, n);
    add_convolution_layer_method<FusedConvolutionBatchNormalizationNode>(_layer_data, n);
}

void DataLayerVisitor::visit(FusedDepthwiseConvolutionBatchNormalizationNode &n)
{
    _layer_data.clear();
    add_generic_layer_data<FusedDepthwiseConvolutionBatchNormalizationNode>(_layer_data, n);
    add_convolution_layer_data<FusedDepthwiseConvolutionBatchNormalizationNode>(_layer_data, n);
}

void DataLayerVisitor::visit(OutputNode &n)
{
    _layer_data.clear();
    ARM_COMPUTE_UNUSED(n);
}

void DataLayerVisitor::default_visit(INode &n)
{
    _layer_data.clear();
    add_generic_layer_data<INode>(_layer_data, n);
}

const DataLayerVisitor::LayerData &DataLayerVisitor::layer_data() const
{
    return _layer_data;
}
} // namespace graph
} // namespace arm_compute
