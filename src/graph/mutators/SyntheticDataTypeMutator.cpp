/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/graph/mutators/SyntheticDataTypeMutator.h"

#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "support/Cast.h"

#include <set>

namespace arm_compute
{
namespace graph
{
namespace
{
/** Empty accessor class */
class EmptyAccessor final : public graph::ITensorAccessor
{
public:
    /** Default Constructor */
    EmptyAccessor() = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override
    {
        ARM_COMPUTE_UNUSED(tensor);
        return true;
    }
};

/** Check if the mutation pass can be applied
 *
 * @param[in] g Graph the mutation pass need to be applied on
 *
 * @return True if the pass can be applied else false
 */
bool is_mutation_supported(Graph &g)
{
    const std::set<NodeType> unsupported_node_types = { NodeType::DetectionOutputLayer,
                                                        NodeType::NormalizationLayer,
                                                        NodeType::PriorBoxLayer
                                                      };

    for(const auto &utype : unsupported_node_types)
    {
        if(!g.nodes(utype).empty())
        {
            return false;
        }
    }
    return true;
}

/** Remove nodes that get optimized out during conversion
 *
 * @param[in, out] g Graph to remove the nodes from.
 */
void remove_optimized_nodes(Graph &g)
{
    const std::set<NodeType> optimized_node_types = { NodeType::BatchNormalizationLayer };

    for(const auto &opt_type : optimized_node_types)
    {
        const std::vector<NodeID> opt_nodes_ids = g.nodes(opt_type);
        for(const auto &node_id : opt_nodes_ids)
        {
            INode *node = g.node(node_id);

            // Get input edge
            Edge *input_edge = node->input_edge(0);
            ARM_COMPUTE_ERROR_ON(input_edge == nullptr);

            // Get producer node
            INode       *producer         = input_edge->producer();
            const EdgeID producer_edge_id = input_edge->producer_idx();
            ARM_COMPUTE_ERROR_ON(producer == nullptr);

            // Get driving nodes
            std::vector<NodeIdxPair> driving_nodes = get_driving_nodes(*node);

            // Remove node
            g.remove_node(node->id());

            // Update connections
            for(auto &driving_node : driving_nodes)
            {
                g.add_connection(producer->id(), producer_edge_id, driving_node.node_id, driving_node.index);
            }
        }
    }
}

/** Convert tensor meta-data
 *
 * @param[in,out] g Graph to convert tensors of.
 */
void convert_tensors(Graph &g, DataType data_type)
{
    auto &tensors = g.tensors();
    for(auto &tensor : tensors)
    {
        if(tensor != nullptr)
        {
            switch(data_type)
            {
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                {
                    tensor->desc().quant_info = QuantizationInfo(0.125f, -10);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Unsupported mutation type");
                    break;
                }
            }
            tensor->desc().data_type = data_type;
        }
    }
}

/** Convert special node
 *
 * @param[in,out] g                  Graph to convert tensors of.
 * @param[in]     fnc                Conversion function.
 * @param[in]     optional_arguments Conversion function arguments.
 */
template <typename NT>
void convert_special_node(Graph &g, std::function<bool(INode *, Tensor *)> const &f)
{
    const std::vector<NodeID> nodes_ids = g.nodes(NT::node_type);
    for(const auto &nodes_id : nodes_ids)
    {
        INode *node = arm_compute::utils::cast::polymorphic_downcast<NT *>(g.node(nodes_id));
        ARM_COMPUTE_ERROR_ON(node == nullptr);

        Tensor *output_tensor = node->output(0);
        ARM_COMPUTE_ERROR_ON(output_tensor == nullptr);

        f(node, output_tensor);
    }
}

/** Converts special tensors
 *
 * @param[in,out] g Graph to convert tensors of.
 */
void convert_special_tensors(Graph &g)
{
    auto softmax_func = [](INode * node, Tensor * tensor)
    {
        ARM_COMPUTE_UNUSED(node);
        if(tensor->desc().data_type == DataType::QASYMM8)
        {
            tensor->desc().quant_info = QuantizationInfo(1.f / 256.f, 0);
        }
        else if(tensor->desc().data_type == DataType::QASYMM8_SIGNED)
        {
            tensor->desc().quant_info = QuantizationInfo(1.f / 256.f, -128);
        }
        return true;
    };

    auto act_func = [](INode * node, Tensor * tensor)
    {
        auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(node);
        if(tensor->desc().data_type == DataType::QASYMM8)
        {
            if(act_node->activation_info().activation() == ActivationLayerInfo::ActivationFunction::TANH)
            {
                tensor->desc().quant_info = QuantizationInfo(1.f / 128.f, 128);
            }
            else if(act_node->activation_info().activation() == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                tensor->desc().quant_info = QuantizationInfo(1.f / 256.f, 0);
            }
        }
        else if(tensor->desc().data_type == DataType::QASYMM8_SIGNED)
        {
            if(act_node->activation_info().activation() == ActivationLayerInfo::ActivationFunction::TANH)
            {
                tensor->desc().quant_info = QuantizationInfo(1.f / 128.f, 0);
            }
            else if(act_node->activation_info().activation() == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                tensor->desc().quant_info = QuantizationInfo(1.f / 256.f, -128);
            }
        }
        return true;
    };

    convert_special_node<ActivationLayerNode>(g, act_func);
    convert_special_node<SoftmaxLayerNode>(g, softmax_func);
}

/** Handle nodes with bias
 *
 * @note Special tensors are for now biases that the data type differ
 *
 * @param[in,out] g Graph to convert tensors of.
 */
void handle_nodes_with_bias(Graph &g)
{
    const std::set<NodeType> special_node_types = { NodeType::ConvolutionLayer,
                                                    NodeType::DeconvolutionLayer,
                                                    NodeType::DepthwiseConvolutionLayer,
                                                    NodeType::FullyConnectedLayer
                                                  };

    for(const auto &spc_type : special_node_types)
    {
        const std::vector<NodeID> scp_nodes_ids = g.nodes(spc_type);
        for(const auto &node_id : scp_nodes_ids)
        {
            INode *node = g.node(node_id);
            if(node != nullptr)
            {
                Tensor *tensor = node->input(2);
                if(tensor != nullptr)
                {
                    tensor->desc().data_type = DataType::S32;
                }
                else
                {
                    auto params = node->common_node_params();
                    params.name = params.name.empty() ? "" : params.name + "Bias";

                    TensorDescriptor b_desc = node->input(1)->desc();
                    auto             depth  = b_desc.shape[get_dimension_idx(b_desc.layout, DataLayoutDimension::BATCHES)];
                    b_desc.shape            = TensorShape(depth);

                    auto accessor = std::make_unique<EmptyAccessor>();
                    auto b_nid    = GraphBuilder::add_const_node(g, params, b_desc, std::move(accessor));
                    g.add_connection(b_nid, 0, node_id, 2);
                }
            }
        }
    }
}
} // namespace

SyntheticDataTypeMutator::SyntheticDataTypeMutator(DataType mutate_type)
    : _mutate_type{ mutate_type }
{
}

const char *SyntheticDataTypeMutator::name()
{
    return "SyntheticDataTypeMutator";
}

IGraphMutator::MutationType SyntheticDataTypeMutator::type() const
{
    return IGraphMutator::MutationType::IR;
}

void SyntheticDataTypeMutator::mutate(Graph &g)
{
    if(is_mutation_supported(g))
    {
        // Remove nodes that get optimized out (e.g. BatchNorm)
        remove_optimized_nodes(g);

        // Convert tensor
        convert_tensors(g, _mutate_type);
        convert_special_tensors(g);

        // Handle special nodes
        handle_nodes_with_bias(g);
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Synthetic data type mutator couldn't be applied" << std::endl);
    }
}
} // namespace graph
} // namespace arm_compute
