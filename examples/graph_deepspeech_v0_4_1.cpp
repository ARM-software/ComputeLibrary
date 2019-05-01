/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement DeepSpeech v0.4.1's network using the Compute Library's graph API */
class GraphDeepSpeechExample : public Example
{
public:
    GraphDeepSpeechExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "DeepSpeech v0.4.1")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string       data_path  = common_params.data_path;
        const std::string model_path = "/cnn_data/deepspeech_model/";

        if(!data_path.empty())
        {
            data_path += model_path;
        }

        // How many timesteps to process at once, higher values mean more latency
        // Notice that this corresponds to the number of LSTM cells that will be instantiated
        const unsigned int n_steps = 16;

        // ReLU clipping value for non-recurrent layers
        const float cell_clip = 20.f;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(26U, 19U, n_steps, 1U), DataLayout::NHWC, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NHWC;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor,
                            get_weights_accessor(data_path, "input_values_x" + std::to_string(n_steps) + ".npy", weights_layout))
              .set_name("input_node");

        if(common_params.data_layout == DataLayout::NCHW)
        {
            graph << PermuteLayer(PermutationVector(2U, 0U, 1U), common_params.data_layout).set_name("permute_to_nhwc");
        }

        graph << ReshapeLayer(TensorShape(494U, n_steps)).set_name("Reshape_input")
              // Layer 1
              << FullyConnectedLayer(
                  2048U,
                  get_weights_accessor(data_path, "h1_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "MatMul_bias.npy"))
              .set_name("fc0")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, cell_clip))
              .set_name("Relu")
              // Layer 2
              << FullyConnectedLayer(
                  2048U,
                  get_weights_accessor(data_path, "h2_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "MatMul_1_bias.npy"))
              .set_name("fc1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, cell_clip))
              .set_name("Relu_1")
              // Layer 3
              << FullyConnectedLayer(
                  2048U,
                  get_weights_accessor(data_path, "h3_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "MatMul_2_bias.npy"))
              .set_name("fc2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, cell_clip))
              .set_name("Relu_2")
              // Layer 4
              << ReshapeLayer(TensorShape(2048U, 1U, n_steps)).set_name("Reshape_1");

        // Unstack Layer (using SplitLayerNode)
        NodeParams unstack_params = { "unstack", graph.hints().target_hint };
        NodeID     unstack_nid    = GraphBuilder::add_split_node(graph.graph(), unstack_params, { graph.tail_node(), 0 }, n_steps, 2);

        // Create input state descriptor
        TensorDescriptor state_descriptor = TensorDescriptor(TensorShape(2048U), common_params.data_type).set_layout(common_params.data_layout);
        SubStream        previous_state(graph);
        SubStream        add_y(graph);

        // Initial state for LSTM is all zeroes for both state_h and state_c, therefore only one input is created
        previous_state << InputLayer(state_descriptor,
                                     get_weights_accessor(data_path, "zeros.npy"))
                       .set_name("previous_state_c_h");
        add_y << InputLayer(state_descriptor,
                            get_weights_accessor(data_path, "ones.npy"))
              .set_name("add_y");

        // TODO(COMPMID-2103): Use sub stream for FC weights and bias in LSTM cells
        // Create LSTM Fully Connected weights and bias descriptors
        //TensorDescriptor lstm_weights_descriptor = TensorDescriptor(TensorShape(4096U, 8192U), common_params.data_type).set_layout(common_params.data_layout);
        //TensorDescriptor lstm_bias_descriptor    = TensorDescriptor(TensorShape(8192U), common_params.data_type).set_layout(common_params.data_layout);
        //SubStream        lstm_fc_weights(graph);
        //SubStream        lstm_fc_bias(graph);

        //lstm_fc_weights << InputLayer(lstm_weights_descriptor,
        //                              get_weights_accessor(data_path, "rnn_lstm_cell_kernel_transpose.npy", weights_layout))
        //                .set_name("h5/transpose");
        //lstm_fc_bias << InputLayer(lstm_bias_descriptor,
        //                           get_weights_accessor(data_path, "rnn_lstm_cell_MatMul_bias.npy"))
        //             .set_name("MatMul_3_bias");

        // LSTM Block
        std::pair<SubStream, SubStream> new_state_1  = add_lstm_cell(data_path, unstack_nid, 0, previous_state, previous_state, add_y);
        std::pair<SubStream, SubStream> new_state_2  = add_lstm_cell(data_path, unstack_nid, 1, new_state_1.first, new_state_1.second, add_y);
        std::pair<SubStream, SubStream> new_state_3  = add_lstm_cell(data_path, unstack_nid, 2, new_state_2.first, new_state_2.second, add_y);
        std::pair<SubStream, SubStream> new_state_4  = add_lstm_cell(data_path, unstack_nid, 3, new_state_3.first, new_state_3.second, add_y);
        std::pair<SubStream, SubStream> new_state_5  = add_lstm_cell(data_path, unstack_nid, 4, new_state_4.first, new_state_4.second, add_y);
        std::pair<SubStream, SubStream> new_state_6  = add_lstm_cell(data_path, unstack_nid, 5, new_state_5.first, new_state_5.second, add_y);
        std::pair<SubStream, SubStream> new_state_7  = add_lstm_cell(data_path, unstack_nid, 6, new_state_6.first, new_state_6.second, add_y);
        std::pair<SubStream, SubStream> new_state_8  = add_lstm_cell(data_path, unstack_nid, 7, new_state_7.first, new_state_7.second, add_y);
        std::pair<SubStream, SubStream> new_state_9  = add_lstm_cell(data_path, unstack_nid, 8, new_state_8.first, new_state_8.second, add_y);
        std::pair<SubStream, SubStream> new_state_10 = add_lstm_cell(data_path, unstack_nid, 9, new_state_9.first, new_state_9.second, add_y);
        std::pair<SubStream, SubStream> new_state_11 = add_lstm_cell(data_path, unstack_nid, 10, new_state_10.first, new_state_10.second, add_y);
        std::pair<SubStream, SubStream> new_state_12 = add_lstm_cell(data_path, unstack_nid, 11, new_state_11.first, new_state_11.second, add_y);
        std::pair<SubStream, SubStream> new_state_13 = add_lstm_cell(data_path, unstack_nid, 12, new_state_12.first, new_state_12.second, add_y);
        std::pair<SubStream, SubStream> new_state_14 = add_lstm_cell(data_path, unstack_nid, 13, new_state_13.first, new_state_13.second, add_y);
        std::pair<SubStream, SubStream> new_state_15 = add_lstm_cell(data_path, unstack_nid, 14, new_state_14.first, new_state_14.second, add_y);
        std::pair<SubStream, SubStream> new_state_16 = add_lstm_cell(data_path, unstack_nid, 15, new_state_15.first, new_state_15.second, add_y);

        if(n_steps > 1)
        {
            // Concatenate new states on height
            const int axis = 1;
            graph << StackLayer(axis,
                                std::move(new_state_1.second),
                                std::move(new_state_2.second),
                                std::move(new_state_3.second),
                                std::move(new_state_4.second),
                                std::move(new_state_5.second),
                                std::move(new_state_6.second),
                                std::move(new_state_7.second),
                                std::move(new_state_8.second),
                                std::move(new_state_9.second),
                                std::move(new_state_10.second),
                                std::move(new_state_11.second),
                                std::move(new_state_12.second),
                                std::move(new_state_13.second),
                                std::move(new_state_14.second),
                                std::move(new_state_15.second),
                                std::move(new_state_16.second))
                  .set_name("concat");
        }

        graph << FullyConnectedLayer(
                  2048U,
                  get_weights_accessor(data_path, "h5_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "MatMul_3_bias.npy"))
              .set_name("fc3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, cell_clip))
              .set_name("Relu3")
              << FullyConnectedLayer(
                  29U,
                  get_weights_accessor(data_path, "h6_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "MatMul_4_bias.npy"))
              .set_name("fc3")
              << SoftmaxLayer().set_name("logits");

        graph << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_file  = common_params.tuner_file;

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    Status set_node_params(Graph &g, NodeID nid, NodeParams &params)
    {
        INode *node = g.node(nid);
        ARM_COMPUTE_RETURN_ERROR_ON(!node);

        node->set_common_node_parameters(params);

        return Status{};
    }

    std::pair<SubStream, SubStream> add_lstm_cell(const std::string &data_path,
                                                  NodeID       unstack_nid,
                                                  unsigned int unstack_idx,
                                                  SubStream    previous_state_c,
                                                  SubStream    previous_state_h,
                                                  SubStream    add_y)
    // TODO(COMPMID-2103): Use sub streams for FC weights and bias
    //SubStream lstm_fc_weights,
    //SubStream lstm_fc_bias)
    {
        const std::string         cell_name("rnn/lstm_cell_" + std::to_string(unstack_idx));
        const DataLayoutDimension concat_dim = (common_params.data_layout == DataLayout::NHWC) ? DataLayoutDimension::CHANNEL : DataLayoutDimension::WIDTH;

        // Concatenate result of Unstack with previous_state_h
        NodeParams concat_params = { cell_name + "/concat", graph.hints().target_hint };
        NodeID     concat_nid    = graph.graph().add_node<ConcatenateLayerNode>(2, concat_dim);
        graph.graph().add_connection(unstack_nid, unstack_idx, concat_nid, 0);
        graph.graph().add_connection(previous_state_h.tail_node(), 0, concat_nid, 1);
        set_node_params(graph.graph(), concat_nid, concat_params);
        graph.forward_tail(concat_nid);

        graph << FullyConnectedLayer(
                  8192U,
                  get_weights_accessor(data_path, "rnn_lstm_cell_kernel_transpose.npy", DataLayout::NHWC),
                  get_weights_accessor(data_path, "rnn_lstm_cell_MatMul_bias.npy"))
              .set_name(cell_name + "/BiasAdd");

        // Split Layer
        const unsigned int num_splits = 4;
        const unsigned int split_axis = 0;

        NodeParams split_params = { cell_name + "/split", graph.hints().target_hint };
        NodeID     split_nid    = GraphBuilder::add_split_node(graph.graph(), split_params, { graph.tail_node(), 0 }, num_splits, split_axis);

        NodeParams sigmoid_1_params = { cell_name + "/Sigmoid_1", graph.hints().target_hint };
        NodeParams add_params       = { cell_name + "/add", graph.hints().target_hint };
        NodeParams sigmoid_2_params = { cell_name + "/Sigmoid_2", graph.hints().target_hint };
        NodeParams tanh_params      = { cell_name + "/Tanh", graph.hints().target_hint };

        // Sigmoid 1 (first split)
        NodeID sigmoid_1_nid = graph.graph().add_node<ActivationLayerNode>(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        graph.graph().add_connection(split_nid, 0, sigmoid_1_nid, 0);
        set_node_params(graph.graph(), sigmoid_1_nid, sigmoid_1_params);

        // Tanh (second split)
        NodeID tanh_nid = graph.graph().add_node<ActivationLayerNode>(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f));
        graph.graph().add_connection(split_nid, 1, tanh_nid, 0);
        set_node_params(graph.graph(), tanh_nid, tanh_params);

        SubStream tanh_ss(graph);
        tanh_ss.forward_tail(tanh_nid);

        // Add (third split)
        NodeID add_nid = graph.graph().add_node<EltwiseLayerNode>(EltwiseOperation::Add);
        graph.graph().add_connection(split_nid, 2, add_nid, 0);
        graph.graph().add_connection(add_y.tail_node(), 0, add_nid, 1);
        set_node_params(graph.graph(), add_nid, add_params);

        // Sigmoid 2 (fourth split)
        NodeID sigmoid_2_nid = graph.graph().add_node<ActivationLayerNode>(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        graph.graph().add_connection(split_nid, 3, sigmoid_2_nid, 0);
        set_node_params(graph.graph(), sigmoid_2_nid, sigmoid_2_params);

        SubStream sigmoid_1_ss(graph);
        sigmoid_1_ss.forward_tail(sigmoid_1_nid);
        SubStream mul_1_ss(sigmoid_1_ss);
        mul_1_ss << EltwiseLayer(std::move(sigmoid_1_ss), std::move(tanh_ss), EltwiseOperation::Mul)
                 .set_name(cell_name + "/mul_1");

        SubStream tanh_1_ss_tmp(graph);
        tanh_1_ss_tmp.forward_tail(add_nid);

        tanh_1_ss_tmp << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC))
                      .set_name(cell_name + "/Sigmoid");
        SubStream tanh_1_ss_tmp2(tanh_1_ss_tmp);
        tanh_1_ss_tmp2 << EltwiseLayer(std::move(tanh_1_ss_tmp), std::move(previous_state_c), EltwiseOperation::Mul)
                       .set_name(cell_name + "/mul");
        SubStream tanh_1_ss(tanh_1_ss_tmp2);
        tanh_1_ss << EltwiseLayer(std::move(tanh_1_ss_tmp2), std::move(mul_1_ss), EltwiseOperation::Add)
                  .set_name(cell_name + "/new_state_c");
        SubStream new_state_c(tanh_1_ss);

        tanh_1_ss << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f))
                  .set_name(cell_name + "/Tanh_1");

        SubStream sigmoid_2_ss(graph);
        sigmoid_2_ss.forward_tail(sigmoid_2_nid);
        graph << EltwiseLayer(std::move(sigmoid_2_ss), std::move(tanh_1_ss), EltwiseOperation::Mul)
              .set_name(cell_name + "/new_state_h");

        SubStream new_state_h(graph);
        return std::pair<SubStream, SubStream>(new_state_c, new_state_h);
    }
};

/** Main program for DeepSpeech v0.4.1
 *
 * Model is based on:
 *      https://arxiv.org/abs/1412.5567
 *      "Deep Speech: Scaling up end-to-end speech recognition"
 *      Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, Andrew Y. Ng
 *
 * Provenance: https://github.com/mozilla/DeepSpeech
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphDeepSpeechExample>(argc, argv);
}
