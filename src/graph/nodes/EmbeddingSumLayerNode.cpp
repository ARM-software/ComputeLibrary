
#include "arm_compute/graph/nodes/EmbeddingSumLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
EmbeddingSumLayerNode::EmbeddingSumLayerNode(EmbeddingLayerInfo info): _info(info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

EmbeddingLayerInfo EmbeddingSumLayerNode::embedding_sum_info() const
{
    return _info;
}

bool EmbeddingSumLayerNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (output_id(0) != NullTensorID))
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

    const Tensor *src1 = input(0);
    ARM_COMPUTE_ERROR_ON(src1 == nullptr);

    const Tensor *src2 = input(1);
    ARM_COMPUTE_ERROR_ON(src2 == nullptr);

    auto output_info = src1->desc();

    TensorShape out_shape = TensorShape::broadcast_shape(src1->desc().shape, src2->desc().shape);
    ARM_COMPUTE_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    output_info.set_shape(out_shape);

    return output_info;
}

TensorDescriptor EmbeddingSumLayerNode::compute_output_descriptor(const TensorDescriptor &token_descriptor,
                                                                  const TensorDescriptor &segment_descriptor,
                                                                  const TensorDescriptor &position_descriptor)
{
    TensorDescriptor output_descriptor = token_descriptor;

    std::cout << "src/graph/nodes/EmbeddingSumLayerNode.cpp compute_output_descriptor" << std::endl;
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

    ARM_COMPUTE_UNUSED(segment_descriptor);
    ARM_COMPUTE_UNUSED(position_descriptor);
    
    return output_descriptor;
}

ConvertPolicy EmbeddingSumLayerNode::convert_policy() const
{
    return _info.c_policy();
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
