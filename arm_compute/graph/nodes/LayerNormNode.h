#ifndef ARM_COMPUTE_GRAPH_LAYER_NORM_NODE_H
#define ARM_COMPUTE_GRAPH_LAYER_NORM_NODE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Layer Normalization node */
class LayerNormNode final : public INode
{
    public:
    /** Constructor
     *
     * @param[in] linear_info Contains information described in @ref LayerNormLayerInfo.
     */
    LayerNormNode(LayerNormLayerInfo info);
    /** Prevent instances of this class from being copy constructed */
    LayerNormNode(const LayerNormNode &) = delete;
    /** Prevent instances of this class from being copied */
    LayerNormNode &operator=(const LayerNormNode &) = delete;

    /** LayerNormInfo accessor
     *
     * @return LayerNormInfo
     */
    const LayerNormLayerInfo &layer_norm_info() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

    private:
    LayerNormLayerInfo _info;
};
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_LAYER_NORM_NODE_H */
