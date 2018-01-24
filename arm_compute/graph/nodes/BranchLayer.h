/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_BRANCH_LAYER_H__
#define __ARM_COMPUTE_GRAPH_BRANCH_LAYER_H__

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/SubGraph.h"
#include "arm_compute/graph/SubTensor.h"
#include "arm_compute/graph/Types.h"

#include "arm_compute/core/utils/misc/utility.h"

#include <vector>

namespace arm_compute
{
namespace graph
{
/** Branch Layer node */
class BranchLayer final : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] merge_method    Branch merging method
     * @param[in] sub_graph1      First graph branch
     * @param[in] sub_graph2      Second graph branch
     * @param[in] rest_sub_graphs Rest sub-graph branches
     */
    template <typename... Ts>
    BranchLayer(BranchMergeMethod merge_method, SubGraph &&sub_graph1, SubGraph &&sub_graph2, Ts &&... rest_sub_graphs)
        : _branch_merge_method(merge_method), _sub_graphs()
    {
        _sub_graphs.push_back(arm_compute::support::cpp14::make_unique<SubGraph>(std::move(sub_graph1)));
        _sub_graphs.push_back(arm_compute::support::cpp14::make_unique<SubGraph>(std::move(sub_graph2)));

        utility::for_each([&](SubGraph && sub_graph)
        {
            _sub_graphs.push_back(arm_compute::support::cpp14::make_unique<SubGraph>(std::move(sub_graph)));
        },
        std::move(rest_sub_graphs)...);
    }
    /** Default Constructor
     *
     * @param[in] sub_graph Sub graph
     */
    template <typename... Ts>
    BranchLayer(SubGraph &&sub_graph)
        : _branch_merge_method(BranchMergeMethod::DEPTH_CONCATENATE), _sub_graphs()
    {
        _sub_graphs.push_back(arm_compute::support::cpp14::make_unique<SubGraph>(std::move(sub_graph)));
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output) override;

private:
    BranchMergeMethod                      _branch_merge_method;
    std::vector<std::unique_ptr<SubGraph>> _sub_graphs;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_BRANCH_LAYER_H__ */
