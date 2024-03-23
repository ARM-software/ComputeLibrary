#include "arm_compute/graph/nodes/SimpleForwardLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
SimpleForwardLayerNode::SimpleForwardLayerNode(int total_tensors) : _total_tensors(total_tensors)
{
    _input_edges.resize(_total_tensors, EmptyEdgeID);
    _outputs.resize(_total_tensors, NullTensorID);
}

int SimpleForwardLayerNode::total_tensors()
{
    return _total_tensors;
}

bool SimpleForwardLayerNode::forward_descriptors()
{
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
