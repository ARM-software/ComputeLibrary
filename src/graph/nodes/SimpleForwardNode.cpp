#include "arm_compute/graph/nodes/SimpleForwardLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
SimpleForwardLayerNode::SimpleForwardLayerNode(unsigned int total_nodes) : _total_nodes(total_nodes)
{
    _input_edges.resize(_total_nodes, EmptyEdgeID);
    _outputs.resize(_total_nodes, NullTensorID);
}

bool SimpleForwardLayerNode::forward_descriptors()
{
    if (output_id(0) != NullTensorID)
    {
        Tensor *t = output(0);
        ARM_COMPUTE_ERROR_ON(t == nullptr);
        t->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor SimpleForwardLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    return input(idx)->desc();
}

NodeType SimpleForwardLayerNode::type() const
{
    return NodeType::SimpleForwardLayer;
}

void SimpleForwardLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
