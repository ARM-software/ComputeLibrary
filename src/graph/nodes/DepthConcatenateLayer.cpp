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
#include <algorithm>
#include <vector>

#include "arm_compute/graph/nodes/DepthConcatenateLayer.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLDepthConcatenate.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenate.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename DepthConcatenationType, typename TensorType, TargetHint hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(std::vector<arm_compute::ITensor *> inputs, arm_compute::ITensor *output)
{
    auto                      depth_concat = arm_compute::support::cpp14::make_unique<DepthConcatenationType>();
    std::vector<TensorType *> casted_inputs;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(casted_inputs), [](arm_compute::ITensor * input)
    {
        return dynamic_cast<TensorType *>(input);
    });
    depth_concat->configure(
        casted_inputs,
        dynamic_cast<TensorType *>(output));

    return std::move(depth_concat);
}

template <TargetHint                    hint>
std::unique_ptr<arm_compute::IFunction> instantiate(std::vector<arm_compute::ITensor *> inputs, arm_compute::ITensor *output);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(std::vector<arm_compute::ITensor *> inputs, arm_compute::ITensor *output)
{
    return instantiate_function<arm_compute::CLDepthConcatenate, arm_compute::ICLTensor, TargetHint::OPENCL>(std::move(inputs), output);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(std::vector<arm_compute::ITensor *> inputs, arm_compute::ITensor *output)
{
    return instantiate_function<arm_compute::NEDepthConcatenate, arm_compute::ITensor, TargetHint::NEON>(std::move(inputs), output);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> DepthConcatenateLayer::instantiate_node(GraphContext &ctx, std::vector<arm_compute::ITensor *> inputs, arm_compute::ITensor *output)
{
    std::unique_ptr<arm_compute::IFunction> func;
    _hint   = ctx.hints().target_hint();
    _inputs = std::move(inputs);
    _output = output;

    if(_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(_inputs, _output);
    }
    else
    {
        func = instantiate<TargetHint::NEON>(_inputs, _output);
    }
    return func;
}

void DepthConcatenateLayer::print_info()
{
    if(_hint == TargetHint::OPENCL)
    {
        std::cout << "Instantiating NEDepthConcatenate";
    }
    else
    {
        std::cout << "Instantiating CLDepthConcatenate";
    }

    for(const auto &i : _inputs)
    {
        std::cout << " Input: " << i->info()->tensor_shape();
    }
    std::cout << " Output: " << _output->info()->tensor_shape();
}
