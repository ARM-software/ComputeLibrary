/*
 * Copyright (c) 2020 Arm Limited.
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

#ifndef ARM_COMPUTE_GRAPH_EDSR_H
#define ARM_COMPUTE_GRAPH_EDSR_H

#include "arm_compute/graph.h"

#include "utils/GraphUtils.h"

class GraphEdsr
{
public:
    GraphEdsr()
        : _graph(0, "EDSR")
    {
    }

    bool setup(const arm_compute::utils::CommonGraphParams &common_params, const arm_compute::utils::SimpleOption<std::string> &expected_output_filename)
    {
        using namespace arm_compute;
        using namespace arm_compute::graph;
        using namespace arm_compute::utils;
        using namespace arm_compute::graph_utils;

        const auto &data_path = common_params.data_path;
        const auto &target    = common_params.target;

        NodeID id_upscale_net_FakeQuantWithMinMaxVars_transposed = _graph.add_node<ConstNode>(
                                                                       TensorDescriptor
        {
            TensorShape{ 12, 2, 2, 3 },
            DataType::QASYMM8,
            QuantizationInfo(0.00393533194437623, 1),
            DataLayout::NHWC });
        INode *node_upscale_net_FakeQuantWithMinMaxVars_transposed = _graph.node(id_upscale_net_FakeQuantWithMinMaxVars_transposed);
        node_upscale_net_FakeQuantWithMinMaxVars_transposed->set_common_node_parameters(NodeParams{ "upscale_net_FakeQuantWithMinMaxVars_transposed", target });
        node_upscale_net_FakeQuantWithMinMaxVars_transposed->output(0)->set_accessor(get_weights_accessor(data_path,
                                                                                                          "/cnn_data/edsr_model/upscale_net_FakeQuantWithMinMaxVars_transposed.npy", DataLayout::NHWC));

        NodeID id_pre_upscale_Conv2D_bias = _graph.add_node<ConstNode>(
                                                TensorDescriptor
        {
            TensorShape{ 12 },
            DataType::S32,
            QuantizationInfo(2.9644968435604824e-06),
            DataLayout::NHWC });
        INode *node_pre_upscale_Conv2D_bias = _graph.node(id_pre_upscale_Conv2D_bias);
        node_pre_upscale_Conv2D_bias->set_common_node_parameters(NodeParams{ "pre_upscale_Conv2D_bias", target });
        node_pre_upscale_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/pre_upscale_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_pre_upscale_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                            TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 12 },
            DataType::QASYMM8,
            QuantizationInfo(0.000455576169770211, 128),
            DataLayout::NHWC });
        INode *node_pre_upscale_FakeQuantWithMinMaxVars = _graph.node(id_pre_upscale_FakeQuantWithMinMaxVars);
        node_pre_upscale_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "pre_upscale_FakeQuantWithMinMaxVars", target });
        node_pre_upscale_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/pre_upscale_FakeQuantWithMinMaxVars.npy",
                                                                                               DataLayout::NHWC));

        NodeID id_post_residual_Conv2D_bias = _graph.add_node<ConstNode>(
                                                  TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.2760000345224398e-06),
            DataLayout::NHWC });
        INode *node_post_residual_Conv2D_bias = _graph.node(id_post_residual_Conv2D_bias);
        node_post_residual_Conv2D_bias->set_common_node_parameters(NodeParams{ "post_residual_Conv2D_bias", target });
        node_post_residual_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/post_residual_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_post_residual_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                              TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00036424631252884865, 129),
            DataLayout::NHWC });
        INode *node_post_residual_FakeQuantWithMinMaxVars = _graph.node(id_post_residual_FakeQuantWithMinMaxVars);
        node_post_residual_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "post_residual_FakeQuantWithMinMaxVars", target });
        node_post_residual_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/post_residual_FakeQuantWithMinMaxVars.npy",
                                                                                                 DataLayout::NHWC));

        TensorShape scalar_4d_shape{};

        scalar_4d_shape.set(0, 1, false).set(1, 1, false).set(2, 1, false).set(3, 1, false);

        NodeID id_mul_15_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_15_y = _graph.node(id_mul_15_y);
        node_mul_15_y->set_common_node_parameters(NodeParams{ "mul_15_y", target });
        node_mul_15_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_15_y.npy", DataLayout::NHWC));

        NodeID id_block_15_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.2441644230420934e-06),
            DataLayout::NHWC });
        INode *node_block_15_1_Conv2D_bias = _graph.node(id_block_15_1_Conv2D_bias);
        node_block_15_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_15_1_Conv2D_bias", target });
        node_block_15_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_15_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_15_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00037038681330159307, 125),
            DataLayout::NHWC });
        INode *node_block_15_1_FakeQuantWithMinMaxVars = _graph.node(id_block_15_1_FakeQuantWithMinMaxVars);
        node_block_15_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_15_1_FakeQuantWithMinMaxVars", target });
        node_block_15_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_15_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_14_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_14_y = _graph.node(id_mul_14_y);
        node_mul_14_y->set_common_node_parameters(NodeParams{ "mul_14_y", target });
        node_mul_14_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_14_y.npy", DataLayout::NHWC));

        NodeID id_block_14_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.3417260333881131e-06),
            DataLayout::NHWC });
        INode *node_block_14_1_Conv2D_bias = _graph.node(id_block_14_1_Conv2D_bias);
        node_block_14_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_14_1_Conv2D_bias", target });
        node_block_14_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_14_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_14_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00040307495510205626, 127),
            DataLayout::NHWC });
        INode *node_block_14_1_FakeQuantWithMinMaxVars = _graph.node(id_block_14_1_FakeQuantWithMinMaxVars);
        node_block_14_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_14_1_FakeQuantWithMinMaxVars", target });
        node_block_14_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_14_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_13_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_13_y = _graph.node(id_mul_13_y);
        node_mul_13_y->set_common_node_parameters(NodeParams{ "mul_13_y", target });
        node_mul_13_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_13_y.npy", DataLayout::NHWC));

        NodeID id_block_13_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.2636977544389083e-06),
            DataLayout::NHWC });
        INode *node_block_13_1_Conv2D_bias = _graph.node(id_block_13_1_Conv2D_bias);
        node_block_13_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_13_1_Conv2D_bias", target });
        node_block_13_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_13_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_13_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003858553245663643, 131),
            DataLayout::NHWC });
        INode *node_block_13_1_FakeQuantWithMinMaxVars = _graph.node(id_block_13_1_FakeQuantWithMinMaxVars);
        node_block_13_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_13_1_FakeQuantWithMinMaxVars", target });
        node_block_13_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_13_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_12_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_12_y = _graph.node(id_mul_12_y);
        node_mul_12_y->set_common_node_parameters(NodeParams{ "mul_12_y", target });
        node_mul_12_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_12_y.npy", DataLayout::NHWC));

        NodeID id_block_12_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.3479783547154511e-06),
            DataLayout::NHWC });
        INode *node_block_12_1_Conv2D_bias = _graph.node(id_block_12_1_Conv2D_bias);
        node_block_12_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_12_1_Conv2D_bias", target });
        node_block_12_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_12_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_12_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00041212860378436744, 130),
            DataLayout::NHWC });
        INode *node_block_12_1_FakeQuantWithMinMaxVars = _graph.node(id_block_12_1_FakeQuantWithMinMaxVars);
        node_block_12_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_12_1_FakeQuantWithMinMaxVars", target });
        node_block_12_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_12_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_11_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_11_y = _graph.node(id_mul_11_y);
        node_mul_11_y->set_common_node_parameters(NodeParams{ "mul_11_y", target });
        node_mul_11_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_11_y.npy", DataLayout::NHWC));

        NodeID id_block_11_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.2847248171965475e-06),
            DataLayout::NHWC });
        INode *node_block_11_1_Conv2D_bias = _graph.node(id_block_11_1_Conv2D_bias);
        node_block_11_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_11_1_Conv2D_bias", target });
        node_block_11_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_11_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_11_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00040296532097272575, 131),
            DataLayout::NHWC });
        INode *node_block_11_1_FakeQuantWithMinMaxVars = _graph.node(id_block_11_1_FakeQuantWithMinMaxVars);
        node_block_11_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_11_1_FakeQuantWithMinMaxVars", target });
        node_block_11_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_11_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_10_y = _graph.add_node<ConstNode>(
                                 TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_10_y = _graph.node(id_mul_10_y);
        node_mul_10_y->set_common_node_parameters(NodeParams{ "mul_10_y", target });
        node_mul_10_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_10_y.npy", DataLayout::NHWC));

        NodeID id_block_10_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.1997129831797793e-06),
            DataLayout::NHWC });
        INode *node_block_10_1_Conv2D_bias = _graph.node(id_block_10_1_Conv2D_bias);
        node_block_10_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_10_1_Conv2D_bias", target });
        node_block_10_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_10_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_10_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                           TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00036640543839894235, 129),
            DataLayout::NHWC });
        INode *node_block_10_1_FakeQuantWithMinMaxVars = _graph.node(id_block_10_1_FakeQuantWithMinMaxVars);
        node_block_10_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_10_1_FakeQuantWithMinMaxVars", target });
        node_block_10_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_10_1_FakeQuantWithMinMaxVars.npy",
                                                                                              DataLayout::NHWC));

        NodeID id_mul_9_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_9_y = _graph.node(id_mul_9_y);
        node_mul_9_y->set_common_node_parameters(NodeParams{ "mul_9_y", target });
        node_mul_9_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_9_y.npy", DataLayout::NHWC));

        NodeID id_block_9_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.1920226370421005e-06),
            DataLayout::NHWC });
        INode *node_block_9_1_Conv2D_bias = _graph.node(id_block_9_1_Conv2D_bias);
        node_block_9_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_9_1_Conv2D_bias", target });
        node_block_9_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_9_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_9_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003706997958943248, 129),
            DataLayout::NHWC });
        INode *node_block_9_1_FakeQuantWithMinMaxVars = _graph.node(id_block_9_1_FakeQuantWithMinMaxVars);
        node_block_9_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_9_1_FakeQuantWithMinMaxVars", target });
        node_block_9_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_9_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_8_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_8_y = _graph.node(id_mul_8_y);
        node_mul_8_y->set_common_node_parameters(NodeParams{ "mul_8_y", target });
        node_mul_8_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_8_y.npy", DataLayout::NHWC));

        NodeID id_block_8_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.218903321387188e-06),
            DataLayout::NHWC });
        INode *node_block_8_1_Conv2D_bias = _graph.node(id_block_8_1_Conv2D_bias);
        node_block_8_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_8_1_Conv2D_bias", target });
        node_block_8_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_8_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_8_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00038377835880964994, 127),
            DataLayout::NHWC });
        INode *node_block_8_1_FakeQuantWithMinMaxVars = _graph.node(id_block_8_1_FakeQuantWithMinMaxVars);
        node_block_8_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_8_1_FakeQuantWithMinMaxVars", target });
        node_block_8_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_8_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_7_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_7_y = _graph.node(id_mul_7_y);
        node_mul_7_y->set_common_node_parameters(NodeParams{ "mul_7_y", target });
        node_mul_7_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_7_y.npy", DataLayout::NHWC));

        NodeID id_block_7_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.257252392861119e-06),
            DataLayout::NHWC });
        INode *node_block_7_1_Conv2D_bias = _graph.node(id_block_7_1_Conv2D_bias);
        node_block_7_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_7_1_Conv2D_bias", target });
        node_block_7_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_7_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_7_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00039844686398282647, 129),
            DataLayout::NHWC });
        INode *node_block_7_1_FakeQuantWithMinMaxVars = _graph.node(id_block_7_1_FakeQuantWithMinMaxVars);
        node_block_7_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_7_1_FakeQuantWithMinMaxVars", target });
        node_block_7_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_7_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_6_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_6_y = _graph.node(id_mul_6_y);
        node_mul_6_y->set_common_node_parameters(NodeParams{ "mul_6_y", target });
        node_mul_6_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_6_y.npy", DataLayout::NHWC));

        NodeID id_block_6_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.244850636794581e-06),
            DataLayout::NHWC });
        INode *node_block_6_1_Conv2D_bias = _graph.node(id_block_6_1_Conv2D_bias);
        node_block_6_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_6_1_Conv2D_bias", target });
        node_block_6_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_6_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_6_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00040187727427110076, 132),
            DataLayout::NHWC });
        INode *node_block_6_1_FakeQuantWithMinMaxVars = _graph.node(id_block_6_1_FakeQuantWithMinMaxVars);
        node_block_6_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_6_1_FakeQuantWithMinMaxVars", target });
        node_block_6_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_6_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_5_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_5_y = _graph.node(id_mul_5_y);
        node_mul_5_y->set_common_node_parameters(NodeParams{ "mul_5_y", target });
        node_mul_5_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_5_y.npy", DataLayout::NHWC));

        NodeID id_block_5_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.241092718373693e-06),
            DataLayout::NHWC });
        INode *node_block_5_1_Conv2D_bias = _graph.node(id_block_5_1_Conv2D_bias);
        node_block_5_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_5_1_Conv2D_bias", target });
        node_block_5_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_5_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_5_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003938926674891263, 129),
            DataLayout::NHWC });
        INode *node_block_5_1_FakeQuantWithMinMaxVars = _graph.node(id_block_5_1_FakeQuantWithMinMaxVars);
        node_block_5_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_5_1_FakeQuantWithMinMaxVars", target });
        node_block_5_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_5_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_4_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_4_y = _graph.node(id_mul_4_y);
        node_mul_4_y->set_common_node_parameters(NodeParams{ "mul_4_y", target });
        node_mul_4_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_4_y.npy", DataLayout::NHWC));

        NodeID id_block_4_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.1748390988941537e-06),
            DataLayout::NHWC });
        INode *node_block_4_1_Conv2D_bias = _graph.node(id_block_4_1_Conv2D_bias);
        node_block_4_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_4_1_Conv2D_bias", target });
        node_block_4_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_4_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_4_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003788181929849088, 129),
            DataLayout::NHWC });
        INode *node_block_4_1_FakeQuantWithMinMaxVars = _graph.node(id_block_4_1_FakeQuantWithMinMaxVars);
        node_block_4_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_4_1_FakeQuantWithMinMaxVars", target });
        node_block_4_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_4_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_3_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_3_y = _graph.node(id_mul_3_y);
        node_mul_3_y->set_common_node_parameters(NodeParams{ "mul_3_y", target });
        node_mul_3_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_3_y.npy", DataLayout::NHWC));

        NodeID id_block_3_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.1937011095142225e-06),
            DataLayout::NHWC });
        INode *node_block_3_1_Conv2D_bias = _graph.node(id_block_3_1_Conv2D_bias);
        node_block_3_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_3_1_Conv2D_bias", target });
        node_block_3_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_3_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_3_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003944312920793891, 129),
            DataLayout::NHWC });
        INode *node_block_3_1_FakeQuantWithMinMaxVars = _graph.node(id_block_3_1_FakeQuantWithMinMaxVars);
        node_block_3_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_3_1_FakeQuantWithMinMaxVars", target });
        node_block_3_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_3_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_2_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_2_y = _graph.node(id_mul_2_y);
        node_mul_2_y->set_common_node_parameters(NodeParams{ "mul_2_y", target });
        node_mul_2_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_2_y.npy", DataLayout::NHWC));

        NodeID id_block_2_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.1634580232566805e-06),
            DataLayout::NHWC });
        INode *node_block_2_1_Conv2D_bias = _graph.node(id_block_2_1_Conv2D_bias);
        node_block_2_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_2_1_Conv2D_bias", target });
        node_block_2_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_2_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_2_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0003789655165746808, 132),
            DataLayout::NHWC });
        INode *node_block_2_1_FakeQuantWithMinMaxVars = _graph.node(id_block_2_1_FakeQuantWithMinMaxVars);
        node_block_2_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_2_1_FakeQuantWithMinMaxVars", target });
        node_block_2_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_2_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_1_y = _graph.add_node<ConstNode>(
                                TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_1_y = _graph.node(id_mul_1_y);
        node_mul_1_y->set_common_node_parameters(NodeParams{ "mul_1_y", target });
        node_mul_1_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_1_y.npy", DataLayout::NHWC));

        NodeID id_block_1_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.197920255435747e-06),
            DataLayout::NHWC });
        INode *node_block_1_1_Conv2D_bias = _graph.node(id_block_1_1_Conv2D_bias);
        node_block_1_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_1_1_Conv2D_bias", target });
        node_block_1_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_1_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_1_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00038527738070115447, 132),
            DataLayout::NHWC });
        INode *node_block_1_1_FakeQuantWithMinMaxVars = _graph.node(id_block_1_1_FakeQuantWithMinMaxVars);
        node_block_1_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_1_1_FakeQuantWithMinMaxVars", target });
        node_block_1_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_1_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_mul_y = _graph.add_node<ConstNode>(
                              TensorDescriptor
        {
            scalar_4d_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.0003921568568330258),
            DataLayout::NHWC });
        INode *node_mul_y = _graph.node(id_mul_y);
        node_mul_y->set_common_node_parameters(NodeParams{ "mul_y", target });
        node_mul_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/mul_y.npy", DataLayout::NHWC));

        NodeID id_block_0_1_Conv2D_bias = _graph.add_node<ConstNode>(
                                              TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.315485519626236e-06),
            DataLayout::NHWC });
        INode *node_block_0_1_Conv2D_bias = _graph.node(id_block_0_1_Conv2D_bias);
        node_block_0_1_Conv2D_bias->set_common_node_parameters(NodeParams{ "block_0_1_Conv2D_bias", target });
        node_block_0_1_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_0_1_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_block_0_1_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                          TensorDescriptor
        {
            TensorShape{ 256, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.00039420535904355347, 129),
            DataLayout::NHWC });
        INode *node_block_0_1_FakeQuantWithMinMaxVars = _graph.node(id_block_0_1_FakeQuantWithMinMaxVars);
        node_block_0_1_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "block_0_1_FakeQuantWithMinMaxVars", target });
        node_block_0_1_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/block_0_1_FakeQuantWithMinMaxVars.npy",
                                                                                             DataLayout::NHWC));

        NodeID id_pre_residual_Conv2D_bias = _graph.add_node<ConstNode>(
                                                 TensorDescriptor
        {
            TensorShape{ 256 },
            DataType::S32,
            QuantizationInfo(1.7214160834555514e-06),
            DataLayout::NHWC });
        INode *node_pre_residual_Conv2D_bias = _graph.node(id_pre_residual_Conv2D_bias);
        node_pre_residual_Conv2D_bias->set_common_node_parameters(NodeParams{ "pre_residual_Conv2D_bias", target });
        node_pre_residual_Conv2D_bias->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/pre_residual_Conv2D_bias.npy", DataLayout::NHWC));

        NodeID id_pre_residual_FakeQuantWithMinMaxVars = _graph.add_node<ConstNode>(
                                                             TensorDescriptor
        {
            TensorShape{ 3, 3, 3, 256 },
            DataType::QASYMM8,
            QuantizationInfo(0.0004389610840007663, 127),
            DataLayout::NHWC });
        INode *node_pre_residual_FakeQuantWithMinMaxVars = _graph.node(id_pre_residual_FakeQuantWithMinMaxVars);
        node_pre_residual_FakeQuantWithMinMaxVars->set_common_node_parameters(NodeParams{ "pre_residual_FakeQuantWithMinMaxVars", target });
        node_pre_residual_FakeQuantWithMinMaxVars->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/edsr_model/pre_residual_FakeQuantWithMinMaxVars.npy",
                                                                                                DataLayout::NHWC));

        TensorShape input_shape{};
        input_shape.set(0, 3, false).set(1, 360, false).set(2, 640, false).set(3, 1, false);

        NodeID id_input = _graph.add_node<InputNode>(
                              TensorDescriptor
        {
            input_shape,
            DataType::QASYMM8,
            QuantizationInfo(0.003921568859368563),
            DataLayout::NHWC });
        INode *node_input = _graph.node(id_input);
        node_input->set_common_node_parameters(NodeParams{ "input", target });
        node_input->output(0)->set_accessor(get_input_accessor(common_params));

        NodeID id_pre_residual_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                             PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.0033370566088706255, 96));
        INode *node_pre_residual_BiasAdd = _graph.node(id_pre_residual_BiasAdd);
        node_pre_residual_BiasAdd->set_common_node_parameters(NodeParams{ "pre_residual_BiasAdd", target });
        _graph.add_connection(id_input, 0, id_pre_residual_BiasAdd, 0);
        _graph.add_connection(id_pre_residual_FakeQuantWithMinMaxVars, 0, id_pre_residual_BiasAdd, 1);
        _graph.add_connection(id_pre_residual_Conv2D_bias, 0, id_pre_residual_BiasAdd, 2);

        NodeID id_block_0_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.007344874087721109, 185));
        INode *node_block_0_1_BiasAdd = _graph.node(id_block_0_1_BiasAdd);
        node_block_0_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_0_1_BiasAdd", target });
        _graph.add_connection(id_pre_residual_BiasAdd, 0, id_block_0_1_BiasAdd, 0);
        _graph.add_connection(id_block_0_1_FakeQuantWithMinMaxVars, 0, id_block_0_1_BiasAdd, 1);
        _graph.add_connection(id_block_0_1_Conv2D_bias, 0, id_block_0_1_BiasAdd, 2);

        NodeID id_mul = _graph.add_node<EltwiseLayerNode>(
                            descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0006341293919831514, 174 } });
        INode *node_mul = _graph.node(id_mul);
        node_mul->set_common_node_parameters(NodeParams{ "mul", target });
        _graph.add_connection(id_block_0_1_BiasAdd, 0, id_mul, 0);
        _graph.add_connection(id_mul_y, 0, id_mul, 1);

        NodeID id_add = _graph.add_node<EltwiseLayerNode>(
                            descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0031092411372810602, 95 } });
        INode *node_add = _graph.node(id_add);
        node_add->set_common_node_parameters(NodeParams{ "add", target });
        _graph.add_connection(id_pre_residual_BiasAdd, 0, id_add, 0);
        _graph.add_connection(id_mul, 0, id_add, 1);

        NodeID id_block_1_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.005333727691322565, 117));
        INode *node_block_1_1_BiasAdd = _graph.node(id_block_1_1_BiasAdd);
        node_block_1_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_1_1_BiasAdd", target });
        _graph.add_connection(id_add, 0, id_block_1_1_BiasAdd, 0);
        _graph.add_connection(id_block_1_1_FakeQuantWithMinMaxVars, 0, id_block_1_1_BiasAdd, 1);
        _graph.add_connection(id_block_1_1_Conv2D_bias, 0, id_block_1_1_BiasAdd, 2);

        NodeID id_mul_1 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004965941770933568, 122 } });
        INode *node_mul_1 = _graph.node(id_mul_1);
        node_mul_1->set_common_node_parameters(NodeParams{ "mul_1", target });
        _graph.add_connection(id_block_1_1_BiasAdd, 0, id_mul_1, 0);
        _graph.add_connection(id_mul_1_y, 0, id_mul_1, 1);

        NodeID id_add_1 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0030700892675668, 96 } });
        INode *node_add_1 = _graph.node(id_add_1);
        node_add_1->set_common_node_parameters(NodeParams{ "add_1", target });
        _graph.add_connection(id_add, 0, id_add_1, 0);
        _graph.add_connection(id_mul_1, 0, id_add_1, 1);

        NodeID id_block_2_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004199742339551449, 132));
        INode *node_block_2_1_BiasAdd = _graph.node(id_block_2_1_BiasAdd);
        node_block_2_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_2_1_BiasAdd", target });
        _graph.add_connection(id_add_1, 0, id_block_2_1_BiasAdd, 0);
        _graph.add_connection(id_block_2_1_FakeQuantWithMinMaxVars, 0, id_block_2_1_BiasAdd, 1);
        _graph.add_connection(id_block_2_1_Conv2D_bias, 0, id_block_2_1_BiasAdd, 2);

        NodeID id_mul_2 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004133903712499887, 130 } });
        INode *node_mul_2 = _graph.node(id_mul_2);
        node_mul_2->set_common_node_parameters(NodeParams{ "mul_2", target });
        _graph.add_connection(id_block_2_1_BiasAdd, 0, id_mul_2, 0);
        _graph.add_connection(id_mul_2_y, 0, id_mul_2, 1);

        NodeID id_add_2 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.003026385325938463, 94 } });
        INode *node_add_2 = _graph.node(id_add_2);
        node_add_2->set_common_node_parameters(NodeParams{ "add_2", target });
        _graph.add_connection(id_add_1, 0, id_add_2, 0);
        _graph.add_connection(id_mul_2, 0, id_add_2, 1);

        NodeID id_block_3_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.003977528307586908, 142));
        INode *node_block_3_1_BiasAdd = _graph.node(id_block_3_1_BiasAdd);
        node_block_3_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_3_1_BiasAdd", target });
        _graph.add_connection(id_add_2, 0, id_block_3_1_BiasAdd, 0);
        _graph.add_connection(id_block_3_1_FakeQuantWithMinMaxVars, 0, id_block_3_1_BiasAdd, 1);
        _graph.add_connection(id_block_3_1_Conv2D_bias, 0, id_block_3_1_BiasAdd, 2);

        NodeID id_mul_3 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0003943995980080217, 141 } });
        INode *node_mul_3 = _graph.node(id_mul_3);
        node_mul_3->set_common_node_parameters(NodeParams{ "mul_3", target });
        _graph.add_connection(id_block_3_1_BiasAdd, 0, id_mul_3, 0);
        _graph.add_connection(id_mul_3_y, 0, id_mul_3, 1);

        NodeID id_add_3 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.003101327223703265, 98 } });
        INode *node_add_3 = _graph.node(id_add_3);
        node_add_3->set_common_node_parameters(NodeParams{ "add_3", target });
        _graph.add_connection(id_add_2, 0, id_add_3, 0);
        _graph.add_connection(id_mul_3, 0, id_add_3, 1);

        NodeID id_block_4_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.0045388080179691315, 146));
        INode *node_block_4_1_BiasAdd = _graph.node(id_block_4_1_BiasAdd);
        node_block_4_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_4_1_BiasAdd", target });
        _graph.add_connection(id_add_3, 0, id_block_4_1_BiasAdd, 0);
        _graph.add_connection(id_block_4_1_FakeQuantWithMinMaxVars, 0, id_block_4_1_BiasAdd, 1);
        _graph.add_connection(id_block_4_1_Conv2D_bias, 0, id_block_4_1_BiasAdd, 2);

        NodeID id_mul_4 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00044342130422592163, 143 } });
        INode *node_mul_4 = _graph.node(id_mul_4);
        node_mul_4->set_common_node_parameters(NodeParams{ "mul_4", target });
        _graph.add_connection(id_block_4_1_BiasAdd, 0, id_mul_4, 0);
        _graph.add_connection(id_mul_4_y, 0, id_mul_4, 1);

        NodeID id_add_4 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.003150839824229479, 98 } });
        INode *node_add_4 = _graph.node(id_add_4);
        node_add_4->set_common_node_parameters(NodeParams{ "add_4", target });
        _graph.add_connection(id_add_3, 0, id_add_4, 0);
        _graph.add_connection(id_mul_4, 0, id_add_4, 1);

        NodeID id_block_5_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.00402890844270587, 132));
        INode *node_block_5_1_BiasAdd = _graph.node(id_block_5_1_BiasAdd);
        node_block_5_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_5_1_BiasAdd", target });
        _graph.add_connection(id_add_4, 0, id_block_5_1_BiasAdd, 0);
        _graph.add_connection(id_block_5_1_FakeQuantWithMinMaxVars, 0, id_block_5_1_BiasAdd, 1);
        _graph.add_connection(id_block_5_1_Conv2D_bias, 0, id_block_5_1_BiasAdd, 2);

        NodeID id_mul_5 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004023382789455354, 132 } });
        INode *node_mul_5 = _graph.node(id_mul_5);
        node_mul_5->set_common_node_parameters(NodeParams{ "mul_5", target });
        _graph.add_connection(id_block_5_1_BiasAdd, 0, id_mul_5, 0);
        _graph.add_connection(id_mul_5_y, 0, id_mul_5, 1);

        NodeID id_add_5 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0030975888948887587, 94 } });
        INode *node_add_5 = _graph.node(id_add_5);
        node_add_5->set_common_node_parameters(NodeParams{ "add_5", target });
        _graph.add_connection(id_add_4, 0, id_add_5, 0);
        _graph.add_connection(id_mul_5, 0, id_add_5, 1);

        NodeID id_block_6_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.00421866774559021, 125));
        INode *node_block_6_1_BiasAdd = _graph.node(id_block_6_1_BiasAdd);
        node_block_6_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_6_1_BiasAdd", target });
        _graph.add_connection(id_add_5, 0, id_block_6_1_BiasAdd, 0);
        _graph.add_connection(id_block_6_1_FakeQuantWithMinMaxVars, 0, id_block_6_1_BiasAdd, 1);
        _graph.add_connection(id_block_6_1_Conv2D_bias, 0, id_block_6_1_BiasAdd, 2);

        NodeID id_mul_6 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00041950203012675047, 125 } });
        INode *node_mul_6 = _graph.node(id_mul_6);
        node_mul_6->set_common_node_parameters(NodeParams{ "mul_6", target });
        _graph.add_connection(id_block_6_1_BiasAdd, 0, id_mul_6, 0);
        _graph.add_connection(id_mul_6_y, 0, id_mul_6, 1);

        NodeID id_add_6 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.003155382815748453, 92 } });
        INode *node_add_6 = _graph.node(id_add_6);
        node_add_6->set_common_node_parameters(NodeParams{ "add_6", target });
        _graph.add_connection(id_add_5, 0, id_add_6, 0);
        _graph.add_connection(id_mul_6, 0, id_add_6, 1);

        NodeID id_block_7_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004250136204063892, 143));
        INode *node_block_7_1_BiasAdd = _graph.node(id_block_7_1_BiasAdd);
        node_block_7_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_7_1_BiasAdd", target });
        _graph.add_connection(id_add_6, 0, id_block_7_1_BiasAdd, 0);
        _graph.add_connection(id_block_7_1_FakeQuantWithMinMaxVars, 0, id_block_7_1_BiasAdd, 1);
        _graph.add_connection(id_block_7_1_Conv2D_bias, 0, id_block_7_1_BiasAdd, 2);

        NodeID id_mul_7 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00042401350219734013, 142 } });
        INode *node_mul_7 = _graph.node(id_mul_7);
        node_mul_7->set_common_node_parameters(NodeParams{ "mul_7", target });
        _graph.add_connection(id_block_7_1_BiasAdd, 0, id_mul_7, 0);
        _graph.add_connection(id_mul_7_y, 0, id_mul_7, 1);

        NodeID id_add_7 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0031760605052113533, 86 } });
        INode *node_add_7 = _graph.node(id_add_7);
        node_add_7->set_common_node_parameters(NodeParams{ "add_7", target });
        _graph.add_connection(id_add_6, 0, id_add_7, 0);
        _graph.add_connection(id_mul_7, 0, id_add_7, 1);

        NodeID id_block_8_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004277155734598637, 123));
        INode *node_block_8_1_BiasAdd = _graph.node(id_block_8_1_BiasAdd);
        node_block_8_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_8_1_BiasAdd", target });
        _graph.add_connection(id_add_7, 0, id_block_8_1_BiasAdd, 0);
        _graph.add_connection(id_block_8_1_FakeQuantWithMinMaxVars, 0, id_block_8_1_BiasAdd, 1);
        _graph.add_connection(id_block_8_1_Conv2D_bias, 0, id_block_8_1_BiasAdd, 2);

        NodeID id_mul_8 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00042673019925132394, 123 } });
        INode *node_mul_8 = _graph.node(id_mul_8);
        node_mul_8->set_common_node_parameters(NodeParams{ "mul_8", target });
        _graph.add_connection(id_block_8_1_BiasAdd, 0, id_mul_8, 0);
        _graph.add_connection(id_mul_8_y, 0, id_mul_8, 1);

        NodeID id_add_8 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0032156009692698717, 86 } });
        INode *node_add_8 = _graph.node(id_add_8);
        node_add_8->set_common_node_parameters(NodeParams{ "add_8", target });
        _graph.add_connection(id_add_7, 0, id_add_8, 0);
        _graph.add_connection(id_mul_8, 0, id_add_8, 1);

        NodeID id_block_9_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                          PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.00445037754252553, 129));
        INode *node_block_9_1_BiasAdd = _graph.node(id_block_9_1_BiasAdd);
        node_block_9_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_9_1_BiasAdd", target });
        _graph.add_connection(id_add_8, 0, id_block_9_1_BiasAdd, 0);
        _graph.add_connection(id_block_9_1_FakeQuantWithMinMaxVars, 0, id_block_9_1_BiasAdd, 1);
        _graph.add_connection(id_block_9_1_Conv2D_bias, 0, id_block_9_1_BiasAdd, 2);

        NodeID id_mul_9 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004448975087143481, 129 } });
        INode *node_mul_9 = _graph.node(id_mul_9);
        node_mul_9->set_common_node_parameters(NodeParams{ "mul_9", target });
        _graph.add_connection(id_block_9_1_BiasAdd, 0, id_mul_9, 0);
        _graph.add_connection(id_mul_9_y, 0, id_mul_9, 1);

        NodeID id_add_9 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0032742770854383707, 80 } });
        INode *node_add_9 = _graph.node(id_add_9);
        node_add_9->set_common_node_parameters(NodeParams{ "add_9", target });
        _graph.add_connection(id_add_8, 0, id_add_9, 0);
        _graph.add_connection(id_mul_9, 0, id_add_9, 1);

        NodeID id_block_10_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.003614710411056876, 131));
        INode *node_block_10_1_BiasAdd = _graph.node(id_block_10_1_BiasAdd);
        node_block_10_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_10_1_BiasAdd", target });
        _graph.add_connection(id_add_9, 0, id_block_10_1_BiasAdd, 0);
        _graph.add_connection(id_block_10_1_FakeQuantWithMinMaxVars, 0, id_block_10_1_BiasAdd, 1);
        _graph.add_connection(id_block_10_1_Conv2D_bias, 0, id_block_10_1_BiasAdd, 2);

        NodeID id_mul_10 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00036083892337046564, 130 } });
        INode *node_mul_10 = _graph.node(id_mul_10);
        node_mul_10->set_common_node_parameters(NodeParams{ "mul_10", target });
        _graph.add_connection(id_block_10_1_BiasAdd, 0, id_mul_10, 0);
        _graph.add_connection(id_mul_10_y, 0, id_mul_10, 1);

        NodeID id_add_10 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0031881770119071007, 81 } });
        INode *node_add_10 = _graph.node(id_add_10);
        node_add_10->set_common_node_parameters(NodeParams{ "add_10", target });
        _graph.add_connection(id_add_9, 0, id_add_10, 0);
        _graph.add_connection(id_mul_10, 0, id_add_10, 1);

        NodeID id_block_11_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.003969002980738878, 133));
        INode *node_block_11_1_BiasAdd = _graph.node(id_block_11_1_BiasAdd);
        node_block_11_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_11_1_BiasAdd", target });
        _graph.add_connection(id_add_10, 0, id_block_11_1_BiasAdd, 0);
        _graph.add_connection(id_block_11_1_FakeQuantWithMinMaxVars, 0, id_block_11_1_BiasAdd, 1);
        _graph.add_connection(id_block_11_1_Conv2D_bias, 0, id_block_11_1_BiasAdd, 2);

        NodeID id_mul_11 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0003968806122429669, 133 } });
        INode *node_mul_11 = _graph.node(id_mul_11);
        node_mul_11->set_common_node_parameters(NodeParams{ "mul_11", target });
        _graph.add_connection(id_block_11_1_BiasAdd, 0, id_mul_11, 0);
        _graph.add_connection(id_mul_11_y, 0, id_mul_11, 1);

        NodeID id_add_11 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0032707711216062307, 80 } });
        INode *node_add_11 = _graph.node(id_add_11);
        node_add_11->set_common_node_parameters(NodeParams{ "add_11", target });
        _graph.add_connection(id_add_10, 0, id_add_11, 0);
        _graph.add_connection(id_mul_11, 0, id_add_11, 1);

        NodeID id_block_12_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004366801120340824, 110));
        INode *node_block_12_1_BiasAdd = _graph.node(id_block_12_1_BiasAdd);
        node_block_12_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_12_1_BiasAdd", target });
        _graph.add_connection(id_add_11, 0, id_block_12_1_BiasAdd, 0);
        _graph.add_connection(id_block_12_1_FakeQuantWithMinMaxVars, 0, id_block_12_1_BiasAdd, 1);
        _graph.add_connection(id_block_12_1_Conv2D_bias, 0, id_block_12_1_BiasAdd, 2);

        NodeID id_mul_12 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004365936329122633, 110 } });
        INode *node_mul_12 = _graph.node(id_mul_12);
        node_mul_12->set_common_node_parameters(NodeParams{ "mul_12", target });
        _graph.add_connection(id_block_12_1_BiasAdd, 0, id_mul_12, 0);
        _graph.add_connection(id_mul_12_y, 0, id_mul_12, 1);

        NodeID id_add_12 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.003275055903941393, 79 } });
        INode *node_add_12 = _graph.node(id_add_12);
        node_add_12->set_common_node_parameters(NodeParams{ "add_12", target });
        _graph.add_connection(id_add_11, 0, id_add_12, 0);
        _graph.add_connection(id_mul_12, 0, id_add_12, 1);

        NodeID id_block_13_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004386766813695431, 139));
        INode *node_block_13_1_BiasAdd = _graph.node(id_block_13_1_BiasAdd);
        node_block_13_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_13_1_BiasAdd", target });
        _graph.add_connection(id_add_12, 0, id_block_13_1_BiasAdd, 0);
        _graph.add_connection(id_block_13_1_FakeQuantWithMinMaxVars, 0, id_block_13_1_BiasAdd, 1);
        _graph.add_connection(id_block_13_1_Conv2D_bias, 0, id_block_13_1_BiasAdd, 2);

        NodeID id_mul_13 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004385628562886268, 139 } });
        INode *node_mul_13 = _graph.node(id_mul_13);
        node_mul_13->set_common_node_parameters(NodeParams{ "mul_13", target });
        _graph.add_connection(id_block_13_1_BiasAdd, 0, id_mul_13, 0);
        _graph.add_connection(id_mul_13_y, 0, id_mul_13, 1);

        NodeID id_add_13 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0033287261612713337, 78 } });
        INode *node_add_13 = _graph.node(id_add_13);
        node_add_13->set_common_node_parameters(NodeParams{ "add_13", target });
        _graph.add_connection(id_add_12, 0, id_add_13, 0);
        _graph.add_connection(id_mul_13, 0, id_add_13, 1);

        NodeID id_block_14_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.0038069337606430054, 130));
        INode *node_block_14_1_BiasAdd = _graph.node(id_block_14_1_BiasAdd);
        node_block_14_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_14_1_BiasAdd", target });
        _graph.add_connection(id_add_13, 0, id_block_14_1_BiasAdd, 0);
        _graph.add_connection(id_block_14_1_FakeQuantWithMinMaxVars, 0, id_block_14_1_BiasAdd, 1);
        _graph.add_connection(id_block_14_1_Conv2D_bias, 0, id_block_14_1_BiasAdd, 2);

        NodeID id_mul_14 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.00037829321809113026, 130 } });
        INode *node_mul_14 = _graph.node(id_mul_14);
        node_mul_14->set_common_node_parameters(NodeParams{ "mul_14", target });
        _graph.add_connection(id_block_14_1_BiasAdd, 0, id_mul_14, 0);
        _graph.add_connection(id_mul_14_y, 0, id_mul_14, 1);

        NodeID id_add_14 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0033590947277843952, 77 } });
        INode *node_add_14 = _graph.node(id_add_14);
        node_add_14->set_common_node_parameters(NodeParams{ "add_14", target });
        _graph.add_connection(id_add_13, 0, id_add_14, 0);
        _graph.add_connection(id_mul_14, 0, id_add_14, 1);

        NodeID id_block_15_1_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                           PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.004009159281849861, 130));
        INode *node_block_15_1_BiasAdd = _graph.node(id_block_15_1_BiasAdd);
        node_block_15_1_BiasAdd->set_common_node_parameters(NodeParams{ "block_15_1_BiasAdd", target });
        _graph.add_connection(id_add_14, 0, id_block_15_1_BiasAdd, 0);
        _graph.add_connection(id_block_15_1_FakeQuantWithMinMaxVars, 0, id_block_15_1_BiasAdd, 1);
        _graph.add_connection(id_block_15_1_Conv2D_bias, 0, id_block_15_1_BiasAdd, 2);

        NodeID id_mul_15 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Mul, QuantizationInfo{ 0.0004008286341559142, 130 } });
        INode *node_mul_15 = _graph.node(id_mul_15);
        node_mul_15->set_common_node_parameters(NodeParams{ "mul_15", target });
        _graph.add_connection(id_block_15_1_BiasAdd, 0, id_mul_15, 0);
        _graph.add_connection(id_mul_15_y, 0, id_mul_15, 1);

        NodeID id_add_15 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0035031239967793226, 78 } });
        INode *node_add_15 = _graph.node(id_add_15);
        node_add_15->set_common_node_parameters(NodeParams{ "add_15", target });
        _graph.add_connection(id_add_14, 0, id_add_15, 0);
        _graph.add_connection(id_mul_15, 0, id_add_15, 1);

        NodeID id_post_residual_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                              PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.005167999770492315, 112));
        INode *node_post_residual_BiasAdd = _graph.node(id_post_residual_BiasAdd);
        node_post_residual_BiasAdd->set_common_node_parameters(NodeParams{ "post_residual_BiasAdd", target });
        _graph.add_connection(id_add_15, 0, id_post_residual_BiasAdd, 0);
        _graph.add_connection(id_post_residual_FakeQuantWithMinMaxVars, 0, id_post_residual_BiasAdd, 1);
        _graph.add_connection(id_post_residual_Conv2D_bias, 0, id_post_residual_BiasAdd, 2);

        NodeID id_add_16 = _graph.add_node<EltwiseLayerNode>(
                               descriptors::EltwiseLayerDescriptor{ EltwiseOperation::Add, QuantizationInfo{ 0.0065071373246610165, 89 } });
        INode *node_add_16 = _graph.node(id_add_16);
        node_add_16->set_common_node_parameters(NodeParams{ "add_16", target });
        _graph.add_connection(id_post_residual_BiasAdd, 0, id_add_16, 0);
        _graph.add_connection(id_pre_residual_BiasAdd, 0, id_add_16, 1);

        NodeID id_pre_upscale_BiasAdd = _graph.add_node<ConvolutionLayerNode>(
                                            PadStrideInfo
        {
            1, 1,
            1, 1,
            1, 1,
            DimensionRoundingType::FLOOR },
        1,
        arm_compute::graph::ConvolutionMethod::Default,
        FastMathHint::Disabled,
        QuantizationInfo(0.005013593938201666, 26));
        INode *node_pre_upscale_BiasAdd = _graph.node(id_pre_upscale_BiasAdd);
        node_pre_upscale_BiasAdd->set_common_node_parameters(NodeParams{ "pre_upscale_BiasAdd", target });
        _graph.add_connection(id_add_16, 0, id_pre_upscale_BiasAdd, 0);
        _graph.add_connection(id_pre_upscale_FakeQuantWithMinMaxVars, 0, id_pre_upscale_BiasAdd, 1);
        _graph.add_connection(id_pre_upscale_Conv2D_bias, 0, id_pre_upscale_BiasAdd, 2);

        NodeID id_upscale_net_FakeQuantWithMinMaxVars_1 = _graph.add_node<DeconvolutionLayerNode>(
                                                              descriptors::DeconvolutionLayerDescriptor
        {
            PadStrideInfo{
                2, 2,
                0, 0,
                0, 0,
                DimensionRoundingType::FLOOR },
            QuantizationInfo{ 0.004990961868315935, 26 } });
        INode *node_upscale_net_FakeQuantWithMinMaxVars_1 = _graph.node(id_upscale_net_FakeQuantWithMinMaxVars_1);
        node_upscale_net_FakeQuantWithMinMaxVars_1->set_common_node_parameters(NodeParams{ "upscale_net_FakeQuantWithMinMaxVars_1", target });
        _graph.add_connection(id_pre_upscale_BiasAdd, 0, id_upscale_net_FakeQuantWithMinMaxVars_1, 0);
        _graph.add_connection(id_upscale_net_FakeQuantWithMinMaxVars_transposed, 0, id_upscale_net_FakeQuantWithMinMaxVars_1, 1);
        TensorShape output_shape;
        output_shape.set(0, 3, false).set(1, 720, false).set(2, 1280, false).set(3, 1, false);

        NodeID id_output_140211982446376   = _graph.add_node<OutputNode>();
        INode *node_output_140211982446376 = _graph.node(id_output_140211982446376);
        node_output_140211982446376->set_common_node_parameters(NodeParams{ "output_140211982446376", target });
        _graph.add_connection(id_upscale_net_FakeQuantWithMinMaxVars_1, 0, id_output_140211982446376, 0);
        node_output_140211982446376->input(0)->set_accessor(get_npy_output_accessor(expected_output_filename.value(), output_shape, common_params.data_type,
                                                                                    common_params.data_layout));

        return true;
    }

    arm_compute::graph::Graph &graph()
    {
        return _graph;
    }

private:
    arm_compute::graph::Graph _graph;
};

#endif /* ARM_COMPUTE_GRAPH_EDSR_H */
