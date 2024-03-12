
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
     * @param[in] TokenEmbeddingLayerInfo Token embedding layer information
     */
    TokenEmbeddingLayerNode(TokenEmbeddingLayerInfo info);
    /** Token embedding Info Accessor
     * 
     * @return Token embedding Info
     */
    TokenEmbeddingLayerInfo token_embedding_info() const;
    /** Computes token embedding output descriptor
     *
     * @param[in] TokenEmbeddingLayerInfo Token embedding layer information
     * 
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor, TokenEmbeddingLayerInfo info);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    TokenEmbeddingLayerInfo _info;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_TOKENEMBEDDINGLAYERNODE_H */