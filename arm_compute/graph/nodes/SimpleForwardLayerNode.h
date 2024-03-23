#ifndef ARM_COMPUTE_GRAPH_SIMPLE_FORWARD_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_SIMPLE_FORWARD_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** A helper node to hold multiple input, output in one node
 *  by simply forward input to ouput
*/
class SimpleForwardLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] desc Tensor descriptor
     */
    SimpleForwardLayerNode(int total_tensors);

    // Return _total_tensors
    int total_tensors();

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    int     _total_tensors;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SIMPLE_FORWARD_LAYER_NODE_H */
