#include "arm_compute/graph/nodes/PositionEmbeddingLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{

PositionEmbeddingLayerNode::PositionEmbeddingLayerNode()
{
    _input_edges.resize(2, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

bool PositionEmbeddingLayerNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && input_id(1) != NullTensorID && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor PositionEmbeddingLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0/*token id input*/);
    const Tensor *vec = input(1/*vector const input*/);
    ARM_COMPUTE_ERROR_ON(src == nullptr);
    ARM_COMPUTE_ERROR_ON(vec == nullptr);

    return compute_output_descriptor(src->desc(),vec->desc());
}

TensorDescriptor PositionEmbeddingLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                    const TensorDescriptor &vector_descriptor)
{
    TensorDescriptor output_descriptor = vector_descriptor;
    output_descriptor.shape.set(1, input_descriptor.shape.x());
    std::cout << "src/graph/nodes/PositionEmbeddingLayerNode.cpp compute_output_descriptor" << std::endl;
    std::cout << "output_descriptor shape: " ;
        for(auto v: output_descriptor.shape)std::cout << " "<<v;
    std::cout << std::endl;

    switch (output_descriptor.layout)
    {
        case DataLayout::NCHW :
            std::cout<< "DataLayout: NCHW" << std::endl;
            break;
        case DataLayout::NCDHW:
            std::cout<< "DataLayout: NCDHW" << std::endl;
            break;
        case DataLayout::NDHWC :
            std::cout<< "DataLayout: NDHWC" << std::endl;
            break;
        case DataLayout::NHWC :
            std::cout<< "DataLayout: NHWC" << std::endl;
            break;
        
        default:
            std::cout<< "DataLayout: Unknown" << std::endl;
            break;
    }

    switch (output_descriptor.data_type)
    {
        case DataType::F32 :
            std::cout<< "DataType::F32" << std::endl;
            break;
        case DataType::U8 :
            std::cout<< "DataType::U8" << std::endl;
            break;
        
        default:
            std::cout<< "DataLayout: Unknown" << std::endl;
            break;
    }
    
    return output_descriptor;
}

NodeType PositionEmbeddingLayerNode::type() const
{
    return NodeType::PositionEmbeddingLayer;
}

void PositionEmbeddingLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
