#ifndef ARM_COMPUTE_GRAPH_EMBEDDINGSUMLAYERNODE_H
#define ARM_COMPUTE_GRAPH_EMBEDDINGSUMLAYERNODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Position embedding Node */
class EmbeddingSumLayerNode final : public INode
{
public:
    /** Constructor 
     * 
     * @param[in] info (optional)Embedding layer information
     */
    EmbeddingSumLayerNode(EmbeddingLayerInfo info = EmbeddingLayerInfo());
    /** Embedding Layer Info Accessor
     * 
     * @return Embedding Layer Info
     */
    EmbeddingLayerInfo embedding_sum_info() const;
    /** Computes embedding sum output descriptor
     *
     * @param[in] token_descriptor      Token embedding input tensor descriptor
     * @param[in] segment_descriptor    Segment embedding input tensor descriptor
     * @param[in] position_descriptor   Position embedding input tensor descriptor
     * 
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &token_descriptor,
                                                      const TensorDescriptor &segment_descriptor,
                                                      const TensorDescriptor &position_descriptor);
                                                      
    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    EmbeddingLayerInfo _info;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_EMBEDDINGSUMLAYERNODE_H */