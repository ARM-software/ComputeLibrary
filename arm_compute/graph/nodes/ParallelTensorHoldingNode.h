#ifndef ARM_COMPUTE_GRAPH_PARALLEL_TENSOR_HOLDING_NODE_H
#define ARM_COMPUTE_GRAPH_PARALLEL_TENSOR_HOLDING_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** A helper node to hold multiple input, output in one node*/
class ParallelTensorHoldingNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] desc Tensor descriptor
     */
    ParallelTensorHoldingNode(unsigned int total_nodes);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    unsigned int     _total_nodes;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_PARALLEL_TENSOR_HOLDING_NODE_H */
