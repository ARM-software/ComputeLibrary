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
#include "arm_compute/graph/nodes/BranchLayer.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/SubGraph.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

#include <memory>
#include <tuple>
#include <vector>

using namespace arm_compute::graph;

namespace
{
void depth_concatenate_output_info(ITensorInfo *info, ITensorInfo *sub_tensor_info)
{
    ARM_COMPUTE_ERROR_ON(info == nullptr);
    ARM_COMPUTE_ERROR_ON(sub_tensor_info == nullptr);

    TensorShape        info_shape            = info->tensor_shape();
    const TensorShape &sub_tensor_info_shape = sub_tensor_info->tensor_shape();

    // Update parent info and valid region
    if(info_shape.total_size() == 0)
    {
        arm_compute::auto_init_if_empty(*info,
                                        sub_tensor_info->tensor_shape(),
                                        sub_tensor_info->num_channels(),
                                        sub_tensor_info->data_type(), sub_tensor_info->fixed_point_position());
        info->set_valid_region(sub_tensor_info->valid_region());
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(info->num_channels() != sub_tensor_info->num_channels());
        ARM_COMPUTE_ERROR_ON(info->data_type() != sub_tensor_info->data_type());
        ARM_COMPUTE_ERROR_ON(info->fixed_point_position() != sub_tensor_info->fixed_point_position());

        // Concatenate depth
        ARM_COMPUTE_ERROR_ON(info_shape.x() != sub_tensor_info_shape.x());
        ARM_COMPUTE_ERROR_ON(info_shape.y() != sub_tensor_info_shape.y());
        info_shape.set(2, info_shape.z() + sub_tensor_info_shape.z());
        info->set_tensor_shape(info_shape);

        // Update valid region
        arm_compute::ValidRegion info_valid_region = info->valid_region();
        info_valid_region.shape.set(2, info_shape.z());
        arm_compute::ValidRegion updated_region = arm_compute::intersect_valid_regions(info_valid_region, sub_tensor_info->valid_region());
        info->set_valid_region(updated_region);
    }
}
} // namespace

/** Branch function */
class BranchFunction final : public arm_compute::IFunction
{
public:
    /** Default Constructor */
    BranchFunction()
        : _graphs()
    {
    }
    /** Registers graph to be executed by the branch function
     *
     * @param[in] graph Graph to register
     */
    void register_graph(std::unique_ptr<Graph> graph)
    {
        _graphs.push_back(std::move(graph));
    }
    // Inherited methods overriden:
    void run() override
    {
        for(auto &g : _graphs)
        {
            ARM_COMPUTE_ERROR_ON(g.get() == nullptr);
            g->run();
        }
    }

private:
    std::vector<std::unique_ptr<Graph>> _graphs;
};

std::unique_ptr<arm_compute::IFunction> BranchLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON(_branch_merge_method != BranchMergeMethod::DEPTH_CONCATENATE);
    ARM_COMPUTE_UNUSED(_branch_merge_method);
    ARM_COMPUTE_ERROR_ON(input == nullptr || input->tensor() == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr || output->tensor() == nullptr);

    // Create branch function
    auto func = arm_compute::support::cpp14::make_unique<BranchFunction>();

    // Track output SubTensorInfo and depth
    TensorInfo out_info;
    int        depth = 0;

    // Constuct all sub-graphs given the input/output
    for(auto &sg : _sub_graphs)
    {
        ARM_COMPUTE_ERROR_ON(sg.get() == nullptr);

        // IO buffers
        std::unique_ptr<ITensorObject> in;
        std::unique_ptr<ITensorObject> out;
        SubTensor                     *out_sub_tensor = nullptr;

        // Create input sub-tensor
        if(!sg->has_input())
        {
            ARM_COMPUTE_ERROR_ON(dynamic_cast<Tensor *>(input) == nullptr);
            in = arm_compute::support::cpp14::make_unique<SubTensor>(*dynamic_cast<Tensor *>(input),
                                                                     input->tensor()->info()->tensor_shape(),
                                                                     Coordinates());
        }

        // Create output sub-tensor
        if(!sg->has_output())
        {
            ARM_COMPUTE_ERROR_ON(dynamic_cast<Tensor *>(output) == nullptr);
            out = arm_compute::support::cpp14::make_unique<SubTensor>(*dynamic_cast<Tensor *>(output),
                                                                      output->tensor()->info()->tensor_shape(),
                                                                      Coordinates(0, 0, depth));
            out_sub_tensor = dynamic_cast<SubTensor *>(out.get());
        }

        // Construct sub_graph
        auto g = sg->construct(ctx.hints().target_hint(), std::move(in), std::move(out));

        // Register graph to function
        func->register_graph(std::move(g));

        // Update and track depth
        if(out_sub_tensor != nullptr)
        {
            ARM_COMPUTE_ERROR_ON(out_sub_tensor->tensor() == nullptr);
            depth += out_sub_tensor->tensor()->info()->tensor_shape()[2];
            depth_concatenate_output_info(&out_info, out_sub_tensor->tensor()->info());
        }
    }

    // Auto-init output
    arm_compute::auto_init_if_empty(*output->tensor()->info(),
                                    out_info.tensor_shape(),
                                    out_info.num_channels(),
                                    out_info.data_type(),
                                    out_info.fixed_point_position());

    return std::move(func);
}