
#include "arm_compute/graph/nodes/EmbeddingSumLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
EmbeddingSumLayerNode::EmbeddingSumLayerNode()
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}


TensorDescriptor EmbeddingSumLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const TensorDescriptor &vector_descriptor)
{
    TensorDescriptor output_descriptor = input_descriptor;

    return output_descriptor;
}


bool EmbeddingSumLayerNode::forward_descriptors()
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

TensorDescriptor EmbeddingSumLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    const Tensor *dst = input(1);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return compute_output_descriptor(src->desc(), dst->desc());
}


NodeType EmbeddingSumLayerNode::type() const
{
    return NodeType::EmbeddingSumLayer;
}

void EmbeddingSumLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
