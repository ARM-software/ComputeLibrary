/*
 * Copyright (c) 2018-2019,2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_GRAPH_INODE_H
#define ARM_COMPUTE_GRAPH_INODE_H

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/LayerDescriptors.h"
#include "arm_compute/graph/TensorDescriptor.h"
#include "arm_compute/graph/Types.h"

#include <list>
#include <set>

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;
class Edge;
class INodeVisitor;
class Tensor;

/** Node interface */
class INode
{
public:
    /** Constructor */
    INode();
    /** Destructor **/
    virtual ~INode() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INode(const INode &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
    INode &operator=(const INode &) = delete;
    /** Allow instances of this class to be moved */
    INode(INode &&) = default;
    /** Allow instances of this class to be move assigned */
    INode &operator=(INode &&) = default;
    /** Validate node
     *
     * @return Status containing any errors
     */
    virtual Status validate() const;
    /** Returns node's type
     *
     * @return Node's type
     */
    virtual NodeType type() const = 0;
    /** Accepts a node visitor
     *
     * @param[in] v Visitor to accept
     */
    virtual void accept(INodeVisitor &v) = 0;
    /** Forwards descriptor information to outputs if possible
     *
     * @return True if descriptor information could be forwarded otherwise false
     */
    virtual bool forward_descriptors() = 0;
    /** Calculates output configuration
     *
     * @param[in] idx Output index to configure
     *
     * @return Output descriptor configuration
     */
    virtual TensorDescriptor configure_output(size_t idx) const = 0;
    /** Returns node's name
     *
     * @return Node name
     */
    std::string name() const;
    /** Returns node's ID
     *
     * @return Node's ID
     */
    NodeID id() const;
    /** Returns node's Graph
     *
     * @return Node's graph
     */
    const Graph *graph() const;
    /** Returns node's Graph
     *
     * @return Node's graph
     */
    Graph *graph();
    /** Sets the graph that this node is registered to
     *
     * @param[in] g Back reference to graph
     */
    void set_graph(Graph *g);
    /** Sets the node id
     *
     * @param[in] id Node id
     */
    void set_id(NodeID id);
    /** Sets common node parameters
     *
     * @param[in] common_params Common node parameters to set
     */
    void set_common_node_parameters(NodeParams common_params);
    /** Sets target preference
     *
     * @note This is not the target that the graph executor might choose, its just an indication
     *
     * @param[in] target Target preference
     */
    void set_requested_target(Target target);
    /** Sets the final execution target
     *
     * @note GraphManager might change this target
     *
     * @param[in] target Final execution target
     */
    void set_assigned_target(Target target);
    /** Sets the output tensor of at a given index
     *
     * @note All edges will get updated
     *
     * @param[in] tid Tensor ID
     * @param[in] idx Output index
     */
    void set_output_tensor(TensorID tid, size_t idx);
    /** Returns inputs of the node
     *
     * @return Inputs of the node
     */
    const std::vector<TensorID> &inputs() const;
    /** Returns outputs of the node
     *
     * @return Outputs of the node
     */
    const std::vector<TensorID> &outputs() const;
    /** Returns input edge set
     *
     * @return Set of input edges
     */
    const std::vector<EdgeID> &input_edges() const;
    /** Returns output edge set
     *
     * @return Set of output edges
     */
    const std::set<EdgeID> &output_edges() const;
    /** Returns the tensor ID of a given input of the node
     *
     * @note Precondition : idx should be a valid input index
     *
     * @param[in] idx Index of the node input
     *
     * @return TensorID of the requested input
     */
    TensorID input_id(size_t idx) const;
    /** Returns the tensor ID of a given output of the node
     *
     * @note Precondition : idx should be a valid output index
     *
     * @param[in] idx Index of the node output
     *
     * @return TensorID of the requested output
     */
    TensorID output_id(size_t idx) const;
    /** Returns the tensor of a given input of the node
     *
     * @note Precondition : idx should be a valid input index
     *
     * @param[in] idx Index of the node input
     *
     * @return Tensor of the requested input
     */
    Tensor *input(size_t idx) const;
    /** Returns the tensor of a given output of the node
     *
     * @note Precondition : idx should be a valid output index
     *
     * @param[in] idx Index of the node output
     *
     * @return Tensor of the requested output
     */
    Tensor *output(size_t idx) const;
    /** Returns the edge ID of a given input of the node
     *
     * @note Precondition : idx should be a valid input index
     *
     * @param[in] idx Index of the node input
     *
     * @return EdgeID of the requested input
     */
    EdgeID input_edge_id(size_t idx) const;
    /** Returns the edge of a given input of the node
     *
     * @note Precondition : idx should be a valid input index
     *
     * @param[in] idx Index of the node input
     *
     * @return Edge of the requested input
     */
    Edge *input_edge(size_t idx) const;
    /** Returns number of inputs of the node
     *
     * @return Number of inputs
     */
    size_t num_inputs() const;
    /** Returns number of outputs of the node
     *
     * @return Number of outputs
     */
    size_t num_outputs() const;
    /** Returns common node parameters
     *
     * @return Common node parameters
     */
    NodeParams common_node_params() const;
    /** Returns requested target for this node
     *
     * @return Requested execution target
     */
    Target requested_target() const;
    /** Returns assigned target for this node
     *
     * @return Assigned target of this node
     */
    Target assigned_target() const;
    /** Post operator info list
     *
     * @return Post operator info list
     */
    const std::list<std::unique_ptr<ConvPostOpInfo>> &post_op_info_list() const;
    /** Post operator info list
     *
     * @return Post operator info list
     */
    std::list<std::unique_ptr<ConvPostOpInfo>> &post_op_info_list();

protected:
    friend class Graph;

protected:
    Graph                                     *_graph;             /**< Backward reference to graph owning the node */
    NodeID                                     _id;                /**< Node ID */
    NodeParams                                 _common_params;     /**< Node common params */
    std::vector<TensorID>                      _outputs;           /**< Output of the node */
    std::vector<EdgeID>                        _input_edges;       /**< Inputs edge set */
    std::set<EdgeID>                           _output_edges;      /**< Output edge set */
    Target                                     _assigned_target;   /**< Assigned target by the Graph executor */
    std::list<std::unique_ptr<ConvPostOpInfo>> _post_op_info_list; /**< Post operator info list */
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_INODE_H */
