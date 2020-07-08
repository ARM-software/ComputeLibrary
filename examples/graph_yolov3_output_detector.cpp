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

#include "arm_compute/graph.h"
#include "arm_compute/graph/Utils.h"

#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::graph;
using namespace arm_compute::utils;

class GraphYoloV3OutputDetector
{
public:
    GraphYoloV3OutputDetector()
        : _graph(0, "GraphYoloV3OutputDetector")
    {
    }

    bool setup(const CommonGraphParams &common_params, const SimpleOption<std::string> &expected_output_filename)
    {
        using namespace arm_compute;
        using namespace graph_utils;

        const DataLayout  data_layout = common_params.data_layout;
        const std::string data_path   = common_params.data_path;
        const Target      target      = common_params.target;

        const DataLayoutDimension x_dim = (data_layout == DataLayout::NHWC) ? DataLayoutDimension::CHANNEL : DataLayoutDimension::WIDTH;
        const DataLayoutDimension y_dim = (data_layout == DataLayout::NHWC) ? DataLayoutDimension::WIDTH : DataLayoutDimension::HEIGHT;

        NodeID id_ConstantFolding_truediv_1_recip = _graph.add_node<ConstNode>(
                                                        TensorDescriptor
        {
            TensorShape{ 1, 1, 1 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_ConstantFolding_truediv_1_recip = _graph.node(id_ConstantFolding_truediv_1_recip);
        node_ConstantFolding_truediv_1_recip->set_common_node_parameters(NodeParams{ "ConstantFolding_truediv_1_recip", target });
        node_ConstantFolding_truediv_1_recip->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/ConstantFolding_truediv_1_recip.npy", data_layout));

        NodeID id_ConstantFolding_truediv_recip = _graph.add_node<ConstNode>(
                                                      TensorDescriptor
        {
            TensorShape{ 1, 1, 1 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_ConstantFolding_truediv_recip = _graph.node(id_ConstantFolding_truediv_recip);
        node_ConstantFolding_truediv_recip->set_common_node_parameters(NodeParams{ "ConstantFolding_truediv_recip", target });
        node_ConstantFolding_truediv_recip->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/ConstantFolding_truediv_recip.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_6_y = _graph.add_node<ConstNode>(
                                                 TensorDescriptor
        {
            TensorShape{ 2 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_6_y = _graph.node(id_detector_yolo_v3_mul_6_y);
        node_detector_yolo_v3_mul_6_y->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_6_y", target });
        node_detector_yolo_v3_mul_6_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_6_y.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_3_y = _graph.add_node<ConstNode>(
                                                 TensorDescriptor
        {
            TensorShape{ 2 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_3_y = _graph.node(id_detector_yolo_v3_mul_3_y);
        node_detector_yolo_v3_mul_3_y->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_3_y", target });
        node_detector_yolo_v3_mul_3_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_3_y.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_y = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 2 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_y = _graph.node(id_detector_yolo_v3_mul_y);
        node_detector_yolo_v3_mul_y->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_y", target });
        node_detector_yolo_v3_mul_y->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_y.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_7 = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 2, 8112 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_7 = _graph.node(id_detector_yolo_v3_mul_7);
        node_detector_yolo_v3_mul_7->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_7", target });
        node_detector_yolo_v3_mul_7->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_7.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_11 = _graph.add_node<ConstNode>(
                                                    TensorDescriptor
        {
            TensorShape{ 2, 8112 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_Reshape_11 = _graph.node(id_detector_yolo_v3_Reshape_11);
        node_detector_yolo_v3_Reshape_11->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_11", target });
        node_detector_yolo_v3_Reshape_11->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_Reshape_11.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_4 = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 2, 2028 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_4 = _graph.node(id_detector_yolo_v3_mul_4);
        node_detector_yolo_v3_mul_4->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_4", target });
        node_detector_yolo_v3_mul_4->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_4.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_7 = _graph.add_node<ConstNode>(
                                                   TensorDescriptor
        {
            TensorShape{ 2, 2028 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_Reshape_7 = _graph.node(id_detector_yolo_v3_Reshape_7);
        node_detector_yolo_v3_Reshape_7->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_7", target });
        node_detector_yolo_v3_Reshape_7->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_Reshape_7.npy", data_layout));

        NodeID id_detector_yolo_v3_mul_1 = _graph.add_node<ConstNode>(
                                               TensorDescriptor
        {
            TensorShape{ 2, 507 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_mul_1 = _graph.node(id_detector_yolo_v3_mul_1);
        node_detector_yolo_v3_mul_1->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_1", target });
        node_detector_yolo_v3_mul_1->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_mul_1.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_3 = _graph.add_node<ConstNode>(
                                                   TensorDescriptor
        {
            TensorShape{ 2, 507 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_detector_yolo_v3_Reshape_3 = _graph.node(id_detector_yolo_v3_Reshape_3);
        node_detector_yolo_v3_Reshape_3->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_3", target });
        node_detector_yolo_v3_Reshape_3->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/detector_yolo-v3_Reshape_3.npy", data_layout));

        NodeID id_input_to_detector_3 = _graph.add_node<InputNode>(
                                            TensorDescriptor
        {
            TensorShape{ 255, 52, 52, 1 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_input_to_detector_3 = _graph.node(id_input_to_detector_3);
        node_input_to_detector_3->set_common_node_parameters(NodeParams{ "input_to_detector_3", target });
        node_input_to_detector_3->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/input_to_detector_3.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_10 = _graph.add_node<ReshapeLayerNode>(
                                                    TensorShape{ 85, 8112 });
        INode *node_detector_yolo_v3_Reshape_10 = _graph.node(id_detector_yolo_v3_Reshape_10);
        node_detector_yolo_v3_Reshape_10->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_10", target });
        _graph.add_connection(id_input_to_detector_3, 0, id_detector_yolo_v3_Reshape_10, 0);

        NodeID id_detector_yolo_v3_split_2 = _graph.add_node<SplitLayerNode>(
                                                 4,
                                                 0,
                                                 std::vector<int> { 2, 2, 1, 80 });
        INode *node_detector_yolo_v3_split_2 = _graph.node(id_detector_yolo_v3_split_2);
        node_detector_yolo_v3_split_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_split_2", target });
        _graph.add_connection(id_detector_yolo_v3_Reshape_10, 0, id_detector_yolo_v3_split_2, 0);

        NodeID id_detector_yolo_v3_Sigmoid_6 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_6 = _graph.node(id_detector_yolo_v3_Sigmoid_6);
        node_detector_yolo_v3_Sigmoid_6->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_6", target });
        _graph.add_connection(id_detector_yolo_v3_split_2, 0, id_detector_yolo_v3_Sigmoid_6, 0);

        NodeID id_detector_yolo_v3_add_2 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Add,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_add_2 = _graph.node(id_detector_yolo_v3_add_2);
        node_detector_yolo_v3_add_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_add_2", target });
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_6, 0, id_detector_yolo_v3_add_2, 0);
        _graph.add_connection(id_detector_yolo_v3_Reshape_11, 0, id_detector_yolo_v3_add_2, 1);

        NodeID id_detector_yolo_v3_mul_6 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul_6 = _graph.node(id_detector_yolo_v3_mul_6);
        node_detector_yolo_v3_mul_6->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_6", target });
        _graph.add_connection(id_detector_yolo_v3_add_2, 0, id_detector_yolo_v3_mul_6, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_6_y, 0, id_detector_yolo_v3_mul_6, 1);

        NodeID id_detector_yolo_v3_Sigmoid_7 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_7 = _graph.node(id_detector_yolo_v3_Sigmoid_7);
        node_detector_yolo_v3_Sigmoid_7->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_7", target });
        _graph.add_connection(id_detector_yolo_v3_split_2, 2, id_detector_yolo_v3_Sigmoid_7, 0);

        NodeID id_detector_yolo_v3_Exp_2 = _graph.add_node<UnaryEltwiseLayerNode>(
                                               descriptors::UnaryEltwiseLayerDescriptor
        {
            UnaryEltwiseOperation::Exp,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_Exp_2 = _graph.node(id_detector_yolo_v3_Exp_2);
        node_detector_yolo_v3_Exp_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Exp_2", target });
        _graph.add_connection(id_detector_yolo_v3_split_2, 1, id_detector_yolo_v3_Exp_2, 0);

        NodeID id_detector_yolo_v3_mul_8 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul_8 = _graph.node(id_detector_yolo_v3_mul_8);
        node_detector_yolo_v3_mul_8->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_8", target });
        _graph.add_connection(id_detector_yolo_v3_Exp_2, 0, id_detector_yolo_v3_mul_8, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_7, 0, id_detector_yolo_v3_mul_8, 1);

        NodeID id_detector_yolo_v3_Sigmoid_8 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_8 = _graph.node(id_detector_yolo_v3_Sigmoid_8);
        node_detector_yolo_v3_Sigmoid_8->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_8", target });
        _graph.add_connection(id_detector_yolo_v3_split_2, 3, id_detector_yolo_v3_Sigmoid_8, 0);

        NodeID id_detector_yolo_v3_concat_8 = _graph.add_node<ConcatenateLayerNode>(
                                                  4,
                                                  descriptors::ConcatLayerDescriptor{ x_dim });
        INode *node_detector_yolo_v3_concat_8 = _graph.node(id_detector_yolo_v3_concat_8);
        node_detector_yolo_v3_concat_8->set_common_node_parameters(NodeParams{ "detector_yolo_v3_concat_8", target });
        _graph.add_connection(id_detector_yolo_v3_mul_6, 0, id_detector_yolo_v3_concat_8, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_8, 0, id_detector_yolo_v3_concat_8, 1);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_7, 0, id_detector_yolo_v3_concat_8, 2);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_8, 0, id_detector_yolo_v3_concat_8, 3);

        NodeID id_input_to_detector_2 = _graph.add_node<InputNode>(
                                            TensorDescriptor
        {
            TensorShape{ 255, 26, 26, 1 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_input_to_detector_2 = _graph.node(id_input_to_detector_2);
        node_input_to_detector_2->set_common_node_parameters(NodeParams{ "input_to_detector_2", target });
        node_input_to_detector_2->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/input_to_detector_2.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_6 = _graph.add_node<ReshapeLayerNode>(
                                                   TensorShape{ 85, 2028 });
        INode *node_detector_yolo_v3_Reshape_6 = _graph.node(id_detector_yolo_v3_Reshape_6);
        node_detector_yolo_v3_Reshape_6->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_6", target });
        _graph.add_connection(id_input_to_detector_2, 0, id_detector_yolo_v3_Reshape_6, 0);

        NodeID id_detector_yolo_v3_split_1 = _graph.add_node<SplitLayerNode>(
                                                 4,
                                                 0,
                                                 std::vector<int> { 2, 2, 1, 80 });
        INode *node_detector_yolo_v3_split_1 = _graph.node(id_detector_yolo_v3_split_1);
        node_detector_yolo_v3_split_1->set_common_node_parameters(NodeParams{ "detector_yolo_v3_split_1", target });
        _graph.add_connection(id_detector_yolo_v3_Reshape_6, 0, id_detector_yolo_v3_split_1, 0);

        NodeID id_detector_yolo_v3_Sigmoid_3 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_3 = _graph.node(id_detector_yolo_v3_Sigmoid_3);
        node_detector_yolo_v3_Sigmoid_3->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_3", target });
        _graph.add_connection(id_detector_yolo_v3_split_1, 0, id_detector_yolo_v3_Sigmoid_3, 0);

        NodeID id_detector_yolo_v3_add_1 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Add,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_add_1 = _graph.node(id_detector_yolo_v3_add_1);
        node_detector_yolo_v3_add_1->set_common_node_parameters(NodeParams{ "detector_yolo_v3_add_1", target });
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_3, 0, id_detector_yolo_v3_add_1, 0);
        _graph.add_connection(id_detector_yolo_v3_Reshape_7, 0, id_detector_yolo_v3_add_1, 1);

        NodeID id_detector_yolo_v3_mul_3 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul_3 = _graph.node(id_detector_yolo_v3_mul_3);
        node_detector_yolo_v3_mul_3->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_3", target });
        _graph.add_connection(id_detector_yolo_v3_add_1, 0, id_detector_yolo_v3_mul_3, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_3_y, 0, id_detector_yolo_v3_mul_3, 1);

        NodeID id_detector_yolo_v3_Sigmoid_4 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_4 = _graph.node(id_detector_yolo_v3_Sigmoid_4);
        node_detector_yolo_v3_Sigmoid_4->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_4", target });
        _graph.add_connection(id_detector_yolo_v3_split_1, 2, id_detector_yolo_v3_Sigmoid_4, 0);

        NodeID id_detector_yolo_v3_Exp_1 = _graph.add_node<UnaryEltwiseLayerNode>(
                                               descriptors::UnaryEltwiseLayerDescriptor
        {
            UnaryEltwiseOperation::Exp,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_Exp_1 = _graph.node(id_detector_yolo_v3_Exp_1);
        node_detector_yolo_v3_Exp_1->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Exp_1", target });
        _graph.add_connection(id_detector_yolo_v3_split_1, 1, id_detector_yolo_v3_Exp_1, 0);

        NodeID id_detector_yolo_v3_mul_5 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul_5 = _graph.node(id_detector_yolo_v3_mul_5);
        node_detector_yolo_v3_mul_5->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_5", target });
        _graph.add_connection(id_detector_yolo_v3_Exp_1, 0, id_detector_yolo_v3_mul_5, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_4, 0, id_detector_yolo_v3_mul_5, 1);

        NodeID id_detector_yolo_v3_Sigmoid_5 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_5 = _graph.node(id_detector_yolo_v3_Sigmoid_5);
        node_detector_yolo_v3_Sigmoid_5->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_5", target });
        _graph.add_connection(id_detector_yolo_v3_split_1, 3, id_detector_yolo_v3_Sigmoid_5, 0);

        NodeID id_detector_yolo_v3_concat_5 = _graph.add_node<ConcatenateLayerNode>(
                                                  4,
                                                  descriptors::ConcatLayerDescriptor{ x_dim });
        INode *node_detector_yolo_v3_concat_5 = _graph.node(id_detector_yolo_v3_concat_5);
        node_detector_yolo_v3_concat_5->set_common_node_parameters(NodeParams{ "detector_yolo_v3_concat_5", target });
        _graph.add_connection(id_detector_yolo_v3_mul_3, 0, id_detector_yolo_v3_concat_5, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_5, 0, id_detector_yolo_v3_concat_5, 1);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_4, 0, id_detector_yolo_v3_concat_5, 2);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_5, 0, id_detector_yolo_v3_concat_5, 3);

        NodeID id_input_to_detector_1 = _graph.add_node<InputNode>(
                                            TensorDescriptor
        {
            TensorShape{ 255, 13, 13, 1 },
            DataType::F32,
            QuantizationInfo(),
            data_layout });
        INode *node_input_to_detector_1 = _graph.node(id_input_to_detector_1);
        node_input_to_detector_1->set_common_node_parameters(NodeParams{ "input_to_detector_1", target });
        node_input_to_detector_1->output(0)->set_accessor(get_weights_accessor(data_path, "/cnn_data/yolov3_output_detector/input_to_detector_1.npy", data_layout));

        NodeID id_detector_yolo_v3_Reshape_2 = _graph.add_node<ReshapeLayerNode>(
                                                   TensorShape{ 85, 507 });
        INode *node_detector_yolo_v3_Reshape_2 = _graph.node(id_detector_yolo_v3_Reshape_2);
        node_detector_yolo_v3_Reshape_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Reshape_2", target });
        _graph.add_connection(id_input_to_detector_1, 0, id_detector_yolo_v3_Reshape_2, 0);

        NodeID id_detector_yolo_v3_split = _graph.add_node<SplitLayerNode>(
                                               4,
                                               0,
                                               std::vector<int> { 2, 2, 1, 80 });
        INode *node_detector_yolo_v3_split = _graph.node(id_detector_yolo_v3_split);
        node_detector_yolo_v3_split->set_common_node_parameters(NodeParams{ "detector_yolo_v3_split", target });
        _graph.add_connection(id_detector_yolo_v3_Reshape_2, 0, id_detector_yolo_v3_split, 0);

        NodeID id_detector_yolo_v3_Sigmoid = _graph.add_node<ActivationLayerNode>(
                                                 ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid = _graph.node(id_detector_yolo_v3_Sigmoid);
        node_detector_yolo_v3_Sigmoid->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid", target });
        _graph.add_connection(id_detector_yolo_v3_split, 0, id_detector_yolo_v3_Sigmoid, 0);

        NodeID id_detector_yolo_v3_add = _graph.add_node<EltwiseLayerNode>(
                                             descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Add,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_add = _graph.node(id_detector_yolo_v3_add);
        node_detector_yolo_v3_add->set_common_node_parameters(NodeParams{ "detector_yolo_v3_add", target });
        _graph.add_connection(id_detector_yolo_v3_Sigmoid, 0, id_detector_yolo_v3_add, 0);
        _graph.add_connection(id_detector_yolo_v3_Reshape_3, 0, id_detector_yolo_v3_add, 1);

        NodeID id_detector_yolo_v3_mul = _graph.add_node<EltwiseLayerNode>(
                                             descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul = _graph.node(id_detector_yolo_v3_mul);
        node_detector_yolo_v3_mul->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul", target });
        _graph.add_connection(id_detector_yolo_v3_add, 0, id_detector_yolo_v3_mul, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_y, 0, id_detector_yolo_v3_mul, 1);

        NodeID id_detector_yolo_v3_Sigmoid_1 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_1 = _graph.node(id_detector_yolo_v3_Sigmoid_1);
        node_detector_yolo_v3_Sigmoid_1->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_1", target });
        _graph.add_connection(id_detector_yolo_v3_split, 2, id_detector_yolo_v3_Sigmoid_1, 0);

        NodeID id_detector_yolo_v3_Exp = _graph.add_node<UnaryEltwiseLayerNode>(
                                             descriptors::UnaryEltwiseLayerDescriptor
        {
            UnaryEltwiseOperation::Exp,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_Exp = _graph.node(id_detector_yolo_v3_Exp);
        node_detector_yolo_v3_Exp->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Exp", target });
        _graph.add_connection(id_detector_yolo_v3_split, 1, id_detector_yolo_v3_Exp, 0);

        NodeID id_detector_yolo_v3_mul_2 = _graph.add_node<EltwiseLayerNode>(
                                               descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_detector_yolo_v3_mul_2 = _graph.node(id_detector_yolo_v3_mul_2);
        node_detector_yolo_v3_mul_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_mul_2", target });
        _graph.add_connection(id_detector_yolo_v3_Exp, 0, id_detector_yolo_v3_mul_2, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_1, 0, id_detector_yolo_v3_mul_2, 1);

        NodeID id_detector_yolo_v3_Sigmoid_2 = _graph.add_node<ActivationLayerNode>(
                                                   ActivationLayerInfo{ ActivationLayerInfo::ActivationFunction::LOGISTIC, 0, 0 });
        INode *node_detector_yolo_v3_Sigmoid_2 = _graph.node(id_detector_yolo_v3_Sigmoid_2);
        node_detector_yolo_v3_Sigmoid_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_Sigmoid_2", target });
        _graph.add_connection(id_detector_yolo_v3_split, 3, id_detector_yolo_v3_Sigmoid_2, 0);

        NodeID id_detector_yolo_v3_concat_2 = _graph.add_node<ConcatenateLayerNode>(
                                                  4,
                                                  descriptors::ConcatLayerDescriptor{ x_dim });
        INode *node_detector_yolo_v3_concat_2 = _graph.node(id_detector_yolo_v3_concat_2);
        node_detector_yolo_v3_concat_2->set_common_node_parameters(NodeParams{ "detector_yolo_v3_concat_2", target });
        _graph.add_connection(id_detector_yolo_v3_mul, 0, id_detector_yolo_v3_concat_2, 0);
        _graph.add_connection(id_detector_yolo_v3_mul_2, 0, id_detector_yolo_v3_concat_2, 1);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_1, 0, id_detector_yolo_v3_concat_2, 2);
        _graph.add_connection(id_detector_yolo_v3_Sigmoid_2, 0, id_detector_yolo_v3_concat_2, 3);

        NodeID id_detector_yolo_v3_concat_9 = _graph.add_node<ConcatenateLayerNode>(
                                                  3,
                                                  descriptors::ConcatLayerDescriptor{ y_dim });
        INode *node_detector_yolo_v3_concat_9 = _graph.node(id_detector_yolo_v3_concat_9);
        node_detector_yolo_v3_concat_9->set_common_node_parameters(NodeParams{ "detector_yolo_v3_concat_9", target });
        _graph.add_connection(id_detector_yolo_v3_concat_2, 0, id_detector_yolo_v3_concat_9, 0);
        _graph.add_connection(id_detector_yolo_v3_concat_5, 0, id_detector_yolo_v3_concat_9, 1);
        _graph.add_connection(id_detector_yolo_v3_concat_8, 0, id_detector_yolo_v3_concat_9, 2);

        NodeID id_split = _graph.add_node<SplitLayerNode>(
                              5,
                              0,
                              std::vector<int> { 1, 1, 1, 1, -1 });
        INode *node_split = _graph.node(id_split);
        node_split->set_common_node_parameters(NodeParams{ "split", target });
        _graph.add_connection(id_detector_yolo_v3_concat_9, 0, id_split, 0);

        NodeID id_truediv = _graph.add_node<EltwiseLayerNode>(
                                descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_truediv = _graph.node(id_truediv);
        node_truediv->set_common_node_parameters(NodeParams{ "truediv", target });
        _graph.add_connection(id_split, 2, id_truediv, 0);
        _graph.add_connection(id_ConstantFolding_truediv_recip, 0, id_truediv, 1);

        NodeID id_sub = _graph.add_node<EltwiseLayerNode>(
                            descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Sub,
            QuantizationInfo() });
        INode *node_sub = _graph.node(id_sub);
        node_sub->set_common_node_parameters(NodeParams{ "sub", target });
        _graph.add_connection(id_split, 0, id_sub, 0);
        _graph.add_connection(id_truediv, 0, id_sub, 1);

        NodeID id_add = _graph.add_node<EltwiseLayerNode>(
                            descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Add,
            QuantizationInfo() });
        INode *node_add = _graph.node(id_add);
        node_add->set_common_node_parameters(NodeParams{ "add", target });
        _graph.add_connection(id_split, 0, id_add, 0);
        _graph.add_connection(id_truediv, 0, id_add, 1);

        NodeID id_truediv_1 = _graph.add_node<EltwiseLayerNode>(
                                  descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Mul,
            QuantizationInfo() });
        INode *node_truediv_1 = _graph.node(id_truediv_1);
        node_truediv_1->set_common_node_parameters(NodeParams{ "truediv_1", target });
        _graph.add_connection(id_split, 3, id_truediv_1, 0);
        _graph.add_connection(id_ConstantFolding_truediv_1_recip, 0, id_truediv_1, 1);

        NodeID id_sub_1 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Sub,
            QuantizationInfo() });
        INode *node_sub_1 = _graph.node(id_sub_1);
        node_sub_1->set_common_node_parameters(NodeParams{ "sub_1", target });
        _graph.add_connection(id_split, 1, id_sub_1, 0);
        _graph.add_connection(id_truediv_1, 0, id_sub_1, 1);

        NodeID id_add_1 = _graph.add_node<EltwiseLayerNode>(
                              descriptors::EltwiseLayerDescriptor
        {
            EltwiseOperation::Add,
            QuantizationInfo() });
        INode *node_add_1 = _graph.node(id_add_1);
        node_add_1->set_common_node_parameters(NodeParams{ "add_1", target });
        _graph.add_connection(id_split, 1, id_add_1, 0);
        _graph.add_connection(id_truediv_1, 0, id_add_1, 1);

        NodeID id_output_boxes = _graph.add_node<ConcatenateLayerNode>(
                                     5,
                                     descriptors::ConcatLayerDescriptor{ x_dim });
        INode *node_output_boxes = _graph.node(id_output_boxes);
        node_output_boxes->set_common_node_parameters(NodeParams{ "output_boxes", target });
        _graph.add_connection(id_sub, 0, id_output_boxes, 0);
        _graph.add_connection(id_sub_1, 0, id_output_boxes, 1);
        _graph.add_connection(id_add, 0, id_output_boxes, 2);
        _graph.add_connection(id_add_1, 0, id_output_boxes, 3);
        _graph.add_connection(id_split, 4, id_output_boxes, 4);

        NodeID id_output_140640247016360   = _graph.add_node<OutputNode>();
        INode *node_output_140640247016360 = _graph.node(id_output_140640247016360);
        node_output_140640247016360->set_common_node_parameters(NodeParams{ "output_140640247016360", target });
        _graph.add_connection(id_output_boxes, 0, id_output_140640247016360, 0);
        node_output_140640247016360->input(0)->set_accessor(get_npy_output_accessor(expected_output_filename.value(), TensorShape(85U, 10647U), DataType::F32, data_layout));

        return true;
    }

    Graph &graph()
    {
        return _graph;
    }

private:
    Graph _graph;
};
class GraphYoloV3OutputDetectorExample : public Example
{
public:
    GraphYoloV3OutputDetectorExample()
        : cmd_parser(), common_opts(cmd_parser), common_params()
    {
        expected_output_filename = cmd_parser.add_option<SimpleOption<std::string>>("expected-output-filename", "");
        expected_output_filename->set_help("Name of npy file containing the expected output to validate the graph output.");
    }
    GraphYoloV3OutputDetectorExample(const GraphYoloV3OutputDetectorExample &) = delete;
    GraphYoloV3OutputDetectorExample &operator=(const GraphYoloV3OutputDetectorExample &) = delete;

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        model.setup(common_params, *expected_output_filename);

        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;

        context.set_config(config);

        auto pass_manager = create_default_pass_manager(common_params.target, config);
        manager.finalize_graph(model.graph(), context, pass_manager, common_params.target);

        return true;
    }

    void do_run() override
    {
        manager.execute_graph(model.graph());
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;

    GraphContext context{};
    GraphManager manager{};

    GraphYoloV3OutputDetector model{};

    SimpleOption<std::string> *expected_output_filename{ nullptr };
};

int main(int argc, char **argv)
{
    return run_example<GraphYoloV3OutputDetectorExample>(argc, argv);
}
