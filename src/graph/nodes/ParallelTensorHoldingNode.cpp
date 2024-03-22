#include "arm_compute/graph/nodes/ParallelTensorHoldingNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
ParallelTensorHoldingNode::ParallelTensorHoldingNode(unsigned int total_nodes) : _total_nodes(total_nodes)
{
    _input_edges.resize(_total_nodes, EmptyEdgeID);
    _outputs.resize(_total_nodes, NullTensorID);
}

/* Map input to output */
bool ParallelTensorHoldingNode::forward_descriptors()
{
    for(unsigned int idx = 0; idx < _total_nodes; idx++)
    {
        _outputs[idx] = input_id(idx);
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(idx);
        if ((input_id(idx) == NullTensorID))
        {
            return false;
        }
    }
    return true;
}

TensorDescriptor ParallelTensorHoldingNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    return input(idx)->desc();
}

NodeType ParallelTensorHoldingNode::type() const
{
    return NodeType::ParallelTensorHoldingLayer;
}

void ParallelTensorHoldingNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
