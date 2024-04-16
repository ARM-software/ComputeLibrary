#include "arm_compute/graph/nodes/LayerNormNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
LayerNormNode::LayerNormNode(LayerNormLayerInfo info): _info(std::move(info))
{
    _input_edges.resize(1, EmptyEdgeID); // Input, weight, bias
    _outputs.resize(1, NullTensorID);
}

const LayerNormLayerInfo& LayerNormNode::layer_norm_info() const
{
    return _info;
}

bool LayerNormNode::forward_descriptors()
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


TensorDescriptor LayerNormNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return src->desc();
}


NodeType LayerNormNode::type() const
{
    return NodeType::LinearLayer;
}

void LayerNormNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
