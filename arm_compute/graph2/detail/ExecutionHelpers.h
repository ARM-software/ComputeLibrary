/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH2_DETAIL_EXECUTION_HELPERS_H__
#define __ARM_COMPUTE_GRAPH2_DETAIL_EXECUTION_HELPERS_H__

#include "arm_compute/graph2/Types.h"

namespace arm_compute
{
namespace graph2
{
// Forward declarations
class Graph;
class GraphContext;
class ExecutionWorkload;
class Tensor;

namespace detail
{
/** Initializes the available backends **/
void default_initialize_backends();
/** Configures all nodes of a graph
 *
 * @param[in] g Graph to configure
 */
void configure_all_tensors(Graph &g);
/** Allocates all tensors of a graph
 *
 * @param[in] g Graph to allocate the tensors
 */
void allocate_all_tensors(Graph &g);
/** Validates all nodes
 *
 * @param[in] g Graph to validate
 */
void validate_all_nodes(Graph &g);
/** Configures all nodes of graph
 *
 * @param[in] g   Graph to configure the nodes
 * @param[in] ctx Graph context to use
 *
 * @return The execution workload
 */
ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx);
/** Calls accessor of a given tensor
 *
 * @param[in] tensor The tensor of which the accessor should be called
 */
void call_tensor_accessor(Tensor *tensor);
/** Call all const node accessors
 *
 * @param[in] g Graph containing the const nodes
 */
void call_all_const_node_accessors(Graph &g);
/** Call all input node accessors
 *
 * @param[in] workload Workload to execute
 */
void call_all_input_node_accessors(ExecutionWorkload &workload);
/** Call all output node accessors
 *
 * @param[in] workload Workload to execute
 */
void call_all_output_node_accessors(ExecutionWorkload &workload);
/** Executes all tasks of a workload
 *
 * @param[in] workload Workload to execute
 */
void call_all_tasks(ExecutionWorkload &workload);
} // namespace detail
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_DETAIL_EXECUTION_HELPERS_H__ */
