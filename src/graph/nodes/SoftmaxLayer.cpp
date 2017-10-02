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
#include "arm_compute/graph/nodes/SoftmaxLayer.h"

#include "arm_compute/core/Logger.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename SoftmaxType, typename TensorType, TargetHint hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, ITensor *output)
{
    auto softmax = arm_compute::support::cpp14::make_unique<SoftmaxType>();
    softmax->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output));

    return std::move(softmax);
}

template <TargetHint                    target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, ITensor *output);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(ITensor *input, ITensor *output)
{
    return instantiate_function<arm_compute::CLSoftmaxLayer, arm_compute::CLTensor, TargetHint::OPENCL>(input, output);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(ITensor *input, ITensor *output)
{
    return instantiate_function<arm_compute::NESoftmaxLayer, arm_compute::Tensor, TargetHint::NEON>(input, output);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> SoftmaxLayer::instantiate_node(GraphContext &ctx, ITensor *input, ITensor *output)
{
    std::unique_ptr<arm_compute::IFunction> func;
    _target_hint = ctx.hints().target_hint();

    if(_target_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(input, output);
        ARM_COMPUTE_LOG("Instantiating CLSoftmaxLayer");
    }
    else
    {
        func = instantiate<TargetHint::NEON>(input, output);
        ARM_COMPUTE_LOG("Instantiating NESoftmaxLayer");
    }

    ARM_COMPUTE_LOG(" Data Type: " << input->info()->data_type()
                    << " Input shape: " << input->info()->tensor_shape()
                    << " Output shape: " << output->info()->tensor_shape()
                    << std::endl);

    return func;
}
