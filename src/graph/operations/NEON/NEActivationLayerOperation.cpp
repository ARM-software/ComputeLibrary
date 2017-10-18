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
#include "arm_compute/graph/operations/NEON/NEActivationLayerOperation.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/graph/OperationRegistrar.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphTypePrinter.h"
#include "utils/TypePrinter.h"

#include <memory>

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> NEActivationLayerOperation::configure(NodeContext &ctx)
{
    ARM_COMPUTE_ERROR_ON(ctx.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(ctx.num_outputs() != 1);
    ARM_COMPUTE_ERROR_ON(dynamic_cast<arm_compute::ITensor *>(ctx.input(0)) == nullptr);
    ARM_COMPUTE_ERROR_ON(dynamic_cast<arm_compute::ITensor *>(ctx.output(0)) == nullptr);

    // Extract IO and info
    auto      *in       = dynamic_cast<arm_compute::ITensor *>(ctx.input(0));
    auto      *out      = dynamic_cast<arm_compute::ITensor *>(ctx.output(0));
    const auto act_info = ctx.parameter<ActivationLayerInfo>("ActivationLayerInfo");

    // Create and configure function
    auto activation = arm_compute::support::cpp14::make_unique<NEActivationLayer>();
    activation->configure(in, out, act_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiating NEActivationLayer"
                               << " Data Type: " << in->info()->data_type()
                               << " Input shape: " << in->info()->tensor_shape()
                               << " Output shape: " << out->info()->tensor_shape()
                               << " Activation function: " << act_info.activation()
                               << " a: " << act_info.a()
                               << " b: " << act_info.b()
                               << std::endl);

    return std::move(activation);
}

TargetHint NEActivationLayerOperation::target() const
{
    return TargetHint::NEON;
}

static detail::OperationRegistrar<NEActivationLayerOperation> registrar("ActivationLayer");