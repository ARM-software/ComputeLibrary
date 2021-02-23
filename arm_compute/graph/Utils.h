/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_UTILS_H
#define ARM_COMPUTE_GRAPH_UTILS_H

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/PassManager.h"

namespace arm_compute
{
namespace graph
{
// Forward Declaration
class GraphContext;

inline bool is_utility_node(INode *node)
{
    std::set<NodeType> utility_node_types = { NodeType::PrintLayer };
    return utility_node_types.find(node->type()) != utility_node_types.end();
}

/** Returns the tensor descriptor of a given tensor
 *
 * @param[in] g   Graph that the tensor belongs to
 * @param[in] tid Tensor ID
 *
 * @return Tensor descriptor if tensor was found else empty descriptor
 */
inline TensorDescriptor get_tensor_descriptor(const Graph &g, TensorID tid)
{
    const Tensor *tensor = g.tensor(tid);
    return (tensor != nullptr) ? tensor->desc() : TensorDescriptor();
}
/** Sets an accessor on a given tensor
 *
 * @param[in] tensor   Tensor to set the accessor to
 * @param[in] accessor Accessor to set
 *
 * @return True if accessor was set else false
 */
inline Status set_tensor_accessor(Tensor *tensor, std::unique_ptr<ITensorAccessor> accessor)
{
    ARM_COMPUTE_RETURN_ERROR_ON(tensor == nullptr);
    tensor->set_accessor(std::move(accessor));

    return Status{};
}
/** Checks if a specific target is supported
 *
 * @param[in] target Target to check
 *
 * @return True if target is support else false
 */
bool is_target_supported(Target target);
/** Returns default target for execution
 *
 * @note If an OpenCL backend exists then OpenCL is returned,
 *       else if the Neon backend exists returns Neon as target.
 *       If no backends are registered an error is raised.
 *
 * @return Default target
 */
Target get_default_target();
/** Forces a single target to all graph constructs
 *
 * @param[in] g      Graph to force target on
 * @param[in] target Target to force
 */
void force_target_to_graph(Graph &g, Target target);
/** Creates a default @ref PassManager
 *
 * @param[in] target Target to create the pass manager for
 * @param[in] cfg    Graph configuration meta-data
 *
 * @return A PassManager with default mutating passes
 */
PassManager create_default_pass_manager(Target target, const GraphConfig &cfg);
/** Setups requested backend context if it exists, is supported and hasn't been initialized already.
 *
 * @param[in,out] ctx    Graph Context.
 * @param[in]     target Target to setup the backend for.
 */
void setup_requested_backend_context(GraphContext &ctx, Target target);
/** Default releases the graph context if not done manually
 *
 * @param[in,out] ctx Graph Context
 */
void release_default_graph_context(GraphContext &ctx);
/** Get size of a tensor's given dimension depending on its layout
 *
 * @param[in] descriptor            Descriptor
 * @param[in] data_layout_dimension Tensor data layout dimension
 *
 * @return Size of requested dimension
 */
size_t get_dimension_size(const TensorDescriptor &descriptor, const DataLayoutDimension data_layout_dimension);
/** Get index of a tensor's given dimension depending on its layout
 *
 * @param[in] data_layout           Data layout of the tensor
 * @param[in] data_layout_dimension Tensor data layout dimension
 *
 * @return Idx of given dimension
 */
size_t get_dimension_idx(DataLayout data_layout, const DataLayoutDimension data_layout_dimension);
/** Get the list of driving nodes of a given node
 *
 * @param[in] node Node to find the driving node of
 *
 * @return A list with the driving node of a given node
 */
std::vector<NodeIdxPair> get_driving_nodes(const INode &node);
/** Configures tensor
 *
 * @param[in, out] tensor Tensor to configure
 */
void configure_tensor(Tensor *tensor);
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_UTILS_H */
