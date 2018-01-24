/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/graph/nodes/DepthwiseConvolutionLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> DepthwiseConvolutionLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();
    _target_hint              = ctx.hints().target_hint();

    if(_weights.tensor() == nullptr)
    {
        TensorShape shape = in->info()->tensor_shape();
        shape.set(Window::DimX, _conv_width);
        shape.set(Window::DimY, _conv_height);
        TensorInfo info = TensorInfo(TensorShape(shape), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position());
        info.set_quantization_info(_quant_info);
        _weights.set_info(std::move(info));
    }
    if(_biases.has_accessor() && _biases.tensor() == nullptr)
    {
        DataType dt = in->info()->data_type();
        _biases.set_info(TensorInfo(TensorShape(in->info()->dimension(2)), in->info()->num_channels(), is_data_type_quantized_asymmetric(dt) ? DataType::S32 : dt, in->info()->fixed_point_position()));
    }

    bool weights_is_loaded = _weights.tensor() != nullptr;
    bool biases_is_loaded  = _biases.has_accessor() ? _biases.tensor() != nullptr : true;

    _weights.set_target(_target_hint);
    if(_biases.has_accessor())
    {
        _biases.set_target(_target_hint);
    }

    // Create node context
    NodeContext node_ctx(OperationType::DepthwiseConvolutionLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_input(_weights.tensor());
    if(_biases.has_accessor())
    {
        node_ctx.add_input(_biases.tensor());
    }
    node_ctx.add_output(out);
    node_ctx.add_parameter<PadStrideInfo>("ConvolutionInfo", _conv_info);
    node_ctx.add_parameter<bool>("Optimized3x3", _opt3x3);

    // Configure operation
    auto func = OperationRegistry::get().find_operation(OperationType::DepthwiseConvolutionLayer, _target_hint)->configure(node_ctx);

    // Fill tensors
    if(!weights_is_loaded)
    {
        _weights.allocate_and_fill_if_needed();
    }
    if(!biases_is_loaded)
    {
        _biases.allocate_and_fill_if_needed();
    }

    // Get function
    return func;
}
