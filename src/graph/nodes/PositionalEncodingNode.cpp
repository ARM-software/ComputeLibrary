
#include "arm_compute/graph/nodes/PositionalEncodingNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
PositionalEncodingNode::PositionalEncodingNode(PositionalEncodingLayerInfo info) : _info(std::move(info))
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

PositionalEncodingLayerInfo PositionalEncodingNode::positional_encoding_info() const
{
    return _info;
}


TensorDescriptor PositionalEncodingNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                   PositionalEncodingLayerInfo info)
{
    TensorDescriptor output_descriptor = input_descriptor;
    const unsigned int pool_size_x  = info.seq_len();
    return output_descriptor;
}


bool PositionalEncodingNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor PositionalEncodingNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return compute_output_descriptor(src->desc(), _info);
}


NodeType PositionalEncodingNode::type() const
{
    return NodeType::PositionalEncodingLayer;
}

void PositionalEncodingNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
