
#ifndef ARM_COMPUTE_GRAPH_POSITIONALENCODINGNODE_H
#define ARM_COMPUTE_GRAPH_POSITIONALENCODINGNODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Positional Encoding Node */
class PositionalEncodingNode final : public INode
{
public:
    /** Constructor 
     * 
     * @param[in] PositionalEncodingLayerInfo Positional encoding layer information
     */
    PositionalEncodingNode(PositionalEncodingLayerInfo info);
    /** Positional Encoding Info Accessor
     * 
     * @return Positional Encoding Info
     */
    PositionalEncodingLayerInfo positional_encoding_info() const;
    /** Computes positional encoding output descriptor
     *
     * @param[in] PositionalEncodingLayerInfo Positional encoding layer information
     * 
     * @return Output descriptor
     */
    //static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor, PositionalEncodingLayerInfo info);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    PositionalEncodingLayerInfo _info;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_POSITIONALENCODINGNODE_H */