/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/core/experimental/OperatorGraph.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/OperatorGraphImpl.h"
#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
void check_dependency_graph_op_success(OperatorGraph &graph, const Status &status)
{
    if(!bool(status))
    {
        graph.impl()->status = Status{ status.error_code(), "Cycles or loops are not allowed" };
    }
}

// Check if there are more than one roots in the graph
void check_multiple_roots(OperatorGraph &graph)
{
    if(graph.impl()->graph.get_root_ops().size() > 1)
    {
        graph.impl()->status = Status{ ErrorCode::RUNTIME_ERROR, "Multiple roots are not allowed" };
    }
}

void check_execution_shape(OperatorGraph &graph, const ITensorInfo &dst_info)
{
    const auto roots = graph.impl()->graph.get_root_ops();
    for(auto root : roots)
    {
        // We assume exactly 1 dst tensor for all operators
        const auto root_info = graph.impl()->tensors[graph.impl()->graph.dst_tensors(root)[0]]->get_tensor_info();
        for(unsigned int dim = 0; dim < root_info->num_dimensions(); ++dim)
        {
            if(root_info->dimension(dim) != dst_info.dimension(dim))
            {
                graph.impl()->status = Status{ ErrorCode::RUNTIME_ERROR, "Cannot change execution space" };
                return;
            }
        }
    }
}
} // namespace

OpTensor::OpTensor(Id id)
    : _id{ id }
{
}

OpTensor::Id OpTensor::id() const
{
    return _id;
}

bool operator<(const OpTensor &t0, const OpTensor &t1)
{
    return t0.id() < t1.id();
}

Operator::Operator(Id id)
    : _id{ id }
{
}

Operator::Id Operator::id() const
{
    return _id;
}

bool operator<(const Operator &op0, const Operator &op1)
{
    return op0.id() < op1.id();
}

OperatorGraph::OperatorGraph()
    : _impl{ std::make_unique<Implementation>() }
{
}

OperatorGraph::~OperatorGraph() = default;

OperatorGraph::Implementation *OperatorGraph::impl()
{
    return _impl.get();
}

const OperatorGraph::Implementation *OperatorGraph::impl() const
{
    return _impl.get();
}

Status validate(const OperatorGraph &graph)
{
    return graph.impl()->status;
}

OpTensor add_tensor(OperatorGraph &graph, ITensorInfo &info)
{
    auto     id = graph.impl()->graph.add_tensor();
    OpTensor op_tensor(id);
    graph.impl()->add_tensor(id, &info);
    return op_tensor;
}

Operator add_op_conv2d(OperatorGraph &graph, const Conv2dDescriptor &desc, OpTensor input, OpTensor weights, OpTensor bias, OpTensor dst)
{
    // Check if map is empty as a complex operator can only be root
    if(!graph.impl()->graph.get_root_ops().empty())
    {
        graph.impl()->status = Status{ ErrorCode::RUNTIME_ERROR, "Cannot add multiple complex operators" };
        return Operator{};
    }

    std::pair<Status, DependencyGraph::Id> status_id;

    if(bias.id() == -1)
    {
        status_id = graph.impl()->graph.add_operator({ input.id(), weights.id() }, { dst.id() });
    }
    else
    {
        status_id = graph.impl()->graph.add_operator({ input.id(), weights.id(), bias.id() }, { dst.id() });
    }

    check_dependency_graph_op_success(graph, status_id.first);

    Operator op_node(status_id.second);

    // Infer TensorInfo
    OpTensorContent *dst_tensor = graph.impl()->tensors[dst.id()].get();
    if(dst_tensor->get_tensor_info()->total_size() == 0)
    {
        auto src   = graph.impl()->tensors[input.id()]->get_tensor_info();
        auto wts   = graph.impl()->tensors[weights.id()]->get_tensor_info();
        auto shape = misc::shape_calculator::compute_deep_convolution_shape(src->tensor_shape(), src->data_layout(), wts->tensor_shape(), PadStrideInfo(desc.stride.x(), desc.stride.y(), desc.pad.left,
                                                                            desc.pad.right,
                                                                            desc.pad.top, desc.pad.bottom, DimensionRoundingType::FLOOR)); // use the default DimensionRoundingType

        auto_init_if_empty(*(dst_tensor->get_tensor_info()), src->clone()->set_tensor_shape(shape));
    }

    // Check execution space
    auto dst_info = dst_tensor->get_tensor_info();
    check_execution_shape(graph, *dst_info);

    ITensorDescPack<OpTensorContent> tensors;
    tensors.add_const_tensor(ACL_SRC_0, graph.impl()->tensors[input.id()].get());
    tensors.add_const_tensor(ACL_SRC_1, graph.impl()->tensors[weights.id()].get());
    if(bias.id() != -1)
    {
        tensors.add_const_tensor(ACL_SRC_2, graph.impl()->tensors[bias.id()].get());
    }
    tensors.add_const_tensor(ACL_DST_0, graph.impl()->tensors[dst.id()].get());

    graph.impl()->add_node<Conv2dContent>(status_id.second, desc, tensors);
    check_multiple_roots(graph);

    return op_node;
}

Operator add_op_conv2d(OperatorGraph &graph, const Conv2dDescriptor &desc, OpTensor input, OpTensor weights, OpTensor dst)
{
    return add_op_conv2d(graph, desc, input, weights, OpTensor(-1), dst);
}

void force_conv2d_method(OperatorGraph &graph, Operator conv2d, ConvolutionMethod method)
{
    auto node = utils::cast::polymorphic_downcast<Conv2dContent *>(graph.impl()->operators[conv2d.id()].get());
    node->set_method(method);
}

Operator add_op_elementwise_add(OperatorGraph &graph, const AddDescriptor &desc, OpTensor lhs, OpTensor rhs, OpTensor dst)
{
    auto id = graph.impl()->graph.add_operator({ rhs.id(), lhs.id() }, { dst.id() });
    check_dependency_graph_op_success(graph, id.first);

    Operator op_node(id.second);

    // Infer TensorInfo
    auto             node_lhs = graph.impl()->tensors[lhs.id()]->get_tensor_info();
    auto             node_rhs = graph.impl()->tensors[rhs.id()]->get_tensor_info();
    OpTensorContent *node_dst = graph.impl()->tensors[dst.id()].get();

    if(node_dst->get_tensor_info()->total_size() == 0)
    {
        const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*node_rhs, *node_lhs);
        auto_init_if_empty(*(node_dst->get_tensor_info()), node_lhs->clone()->set_tensor_shape(broadcast_pair.first));
    }

    // Check execution space
    auto dst_info = node_dst->get_tensor_info();
    check_execution_shape(graph, *dst_info);

    ITensorDescPack<OpTensorContent> tensors;
    tensors.add_const_tensor(ACL_SRC_0, graph.impl()->tensors[lhs.id()].get());
    tensors.add_const_tensor(ACL_SRC_1, graph.impl()->tensors[rhs.id()].get());
    tensors.add_const_tensor(ACL_DST_0, graph.impl()->tensors[dst.id()].get());
    graph.impl()->add_node<AddContent>(id.second, desc, tensors);
    check_multiple_roots(graph);

    return op_node;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */