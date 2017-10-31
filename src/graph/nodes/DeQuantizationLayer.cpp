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
#include "arm_compute/graph/nodes/DequantizationLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> DequantizationLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    _target_hint              = ctx.hints().target_hint();
    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();

    if(_min_max.tensor() == nullptr)
    {
        TensorShape shape = in->info()->tensor_shape();
        shape.set(Window::DimX, 2);
        shape.remove_dimension(1);
        shape.remove_dimension(1);

        _min_max.set_info(TensorInfo(shape, in->info()->num_channels(), DataType::F32));
        _min_max.set_target(_target_hint);
    }

    bool minmax_is_loaded = _min_max.tensor() != nullptr;

    // Create node context
    NodeContext node_ctx(OperationType::DequantizationLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_output(_min_max.tensor());
    node_ctx.add_output(out);

    // Fill min max
    if(!minmax_is_loaded)
    {
        _min_max.allocate_and_fill_if_needed();
    }

    // Get function
    return OperationRegistry::get().find_operation(OperationType::DequantizationLayer, _target_hint)->configure(node_ctx);
}
