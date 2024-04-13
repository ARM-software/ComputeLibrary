#ifndef ARM_COMPUTE_GRAPH_POSITIONEMBEDDINGLAYERNODE_H
#define ARM_COMPUTE_GRAPH_POSITIONEMBEDDINGLAYERNODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Position embedding Node */
class PositionEmbeddingLayerNode final : public INode
{
public:
    /** Constructor  */
    PositionEmbeddingLayerNode();

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_POSITIONEMBEDDINGLAYERNODE_H */