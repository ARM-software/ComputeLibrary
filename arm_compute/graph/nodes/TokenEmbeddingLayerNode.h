#ifndef ARM_COMPUTE_GRAPH_TOKENEMBEDDINGLAYERNODE_H
#define ARM_COMPUTE_GRAPH_TOKENEMBEDDINGLAYERNODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Token embedding Node */
class TokenEmbeddingLayerNode final : public INode
{
public:
    /** Constructor 
     * 
     * @param[in] info Token embedding layer information
     */
    TokenEmbeddingLayerNode(EmbeddingLayerInfo info);
    /** Token embedding Info Accessor
     * 
     * @return Token embedding Info
     */
    EmbeddingLayerInfo token_embedding_info() const;
    /** Computes token embedding output descriptor
     *
     * @param[in] input_descriptor      Text id input tensor descriptor
     * @param[in] vector_descriptor     Vector input tensor descriptor
     * @param[in] EmbeddingLayerInfo    Embedding layer information
     * 
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const TensorDescriptor &vector_descriptor,
                                                      EmbeddingLayerInfo info);

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
#endif /* ARM_COMPUTE_GRAPH_TOKENEMBEDDINGLAYERNODE_H */