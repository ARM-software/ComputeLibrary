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
#include "arm_compute/graph/nodes/NormalizationLayer.h"

#include "arm_compute/core/Logger.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename NormalizationType, typename TensorType, TargetHint target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input, arm_compute::ITensor *output, const NormalizationLayerInfo &norm_info)
{
    auto norm = arm_compute::support::cpp14::make_unique<NormalizationType>();
    norm->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output),
        norm_info);

    return std::move(norm);
}

template <TargetHint                    target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor *input, arm_compute::ITensor *output, const NormalizationLayerInfo &norm_info);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(arm_compute::ITensor *input, arm_compute::ITensor *output, const NormalizationLayerInfo &norm_info)
{
    return instantiate_function<arm_compute::CLNormalizationLayer, arm_compute::ICLTensor, TargetHint::OPENCL>(input, output, norm_info);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(arm_compute::ITensor *input, arm_compute::ITensor *output, const NormalizationLayerInfo &norm_info)
{
    return instantiate_function<arm_compute::NENormalizationLayer, arm_compute::ITensor, TargetHint::NEON>(input, output, norm_info);
}
} // namespace

NormalizationLayer::NormalizationLayer(const NormalizationLayerInfo norm_info)
    : _norm_info(norm_info)
{
}

std::unique_ptr<arm_compute::IFunction> NormalizationLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr || input->tensor() == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr || output->tensor() == nullptr);

    std::unique_ptr<arm_compute::IFunction> func;
    _target_hint = ctx.hints().target_hint();

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();

    if(_target_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(in, out, _norm_info);
    }
    else
    {
        func = instantiate<TargetHint::NEON>(in, out, _norm_info);
    }

    ARM_COMPUTE_LOG(" Data Type: " << in->info()->data_type()
                    << " Input shape: " << in->info()->tensor_shape()
                    << " Output shape: " << out->info()->tensor_shape()
                    << " Normalization info: " << _norm_info
                    << std::endl);

    return func;
}
