
#include "arm_compute/graph/nodes/TokenEmbeddingLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
TokenEmbeddingLayerNode::TokenEmbeddingLayerNode(TokenEmbeddingLayerInfo info) : _info(std::move(info))
{
    std::cout << "Token embedding created " << std::endl;
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

TokenEmbeddingLayerInfo TokenEmbeddingLayerNode::token_embedding_info() const
{
    return _info;
}

TensorDescriptor TokenEmbeddingLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                   TokenEmbeddingLayerInfo info)
{
    TensorDescriptor output_descriptor = input_descriptor;
    const unsigned int seq_len  = info.d_model();
    output_descriptor.shape.set(1,seq_len);
    return output_descriptor;
}

bool TokenEmbeddingLayerNode::forward_descriptors()
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

TensorDescriptor TokenEmbeddingLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return compute_output_descriptor(src->desc(), _info);
}


NodeType TokenEmbeddingLayerNode::type() const
{
    return NodeType::TokenEmbeddingLayer;
}

void TokenEmbeddingLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
