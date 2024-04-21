#include "arm_compute/graph/nodes/SimpleForwardLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
SimpleForwardLayerNode::SimpleForwardLayerNode(int total_tensors) : _total_tensors(total_tensors)
{
    _input_edges.resize(total_tensors, EmptyEdgeID);
    _outputs.resize(total_tensors, NullTensorID);
}

int SimpleForwardLayerNode::total_tensors()
{
    return _total_tensors;
}

bool SimpleForwardLayerNode::forward_descriptors()
{
    for(size_t idx=0; idx <num_inputs(); idx++)
    {   
        if ((input_id(idx) == NullTensorID) || (output_id(idx) == NullTensorID))
        {
            return false;
        }
        Tensor *dst = output(idx);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(idx);
    }
    return true;

    /*
    if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
    */
}

TensorDescriptor SimpleForwardLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(idx);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return src->desc();
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
