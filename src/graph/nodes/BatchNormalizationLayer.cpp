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
#include "arm_compute/graph/nodes/BatchNormalizationLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> BatchNormalizationLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();
    _target_hint              = ctx.hints().target_hint();

    unsigned int batch_norm_size = in->info()->dimension(2);
    if(_mean.tensor() == nullptr)
    {
        _mean.set_info(TensorInfo(TensorShape(batch_norm_size), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }
    if(_var.tensor() == nullptr)
    {
        _var.set_info(TensorInfo(TensorShape(batch_norm_size), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }
    if(_beta.tensor() == nullptr)
    {
        _beta.set_info(TensorInfo(TensorShape(batch_norm_size), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }
    if(_gamma.tensor() == nullptr)
    {
        _gamma.set_info(TensorInfo(TensorShape(batch_norm_size), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }

    bool mean_is_loaded  = _mean.tensor() != nullptr;
    bool var_is_loaded   = _var.tensor() != nullptr;
    bool gamma_is_loaded = _gamma.tensor() != nullptr;
    bool beta_is_loaded  = _beta.tensor() != nullptr;

    // Set mean, var, gamma and beta target
    _mean.set_target(_target_hint);
    _var.set_target(_target_hint);
    _gamma.set_target(_target_hint);
    _beta.set_target(_target_hint);

    // Create node context
    NodeContext node_ctx(OperationType::BatchNormalizationLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_input(_mean.tensor());
    node_ctx.add_input(_var.tensor());
    node_ctx.add_input(_beta.tensor());
    node_ctx.add_input(_gamma.tensor());
    node_ctx.add_output(out);
    node_ctx.add_parameter<float>("epsilon", _epsilon);

    // Configure operation
    auto func = OperationRegistry::get().find_operation(OperationType::BatchNormalizationLayer, _target_hint)->configure(node_ctx);

    // Fill tensors
    if(!mean_is_loaded)
    {
        _mean.allocate_and_fill_if_needed();
    }
    if(!var_is_loaded)
    {
        _var.allocate_and_fill_if_needed();
    }
    if(!gamma_is_loaded)
    {
        _gamma.allocate_and_fill_if_needed();
    }
    if(!beta_is_loaded)
    {
        _beta.allocate_and_fill_if_needed();
    }

    // Get function
    return func;
}