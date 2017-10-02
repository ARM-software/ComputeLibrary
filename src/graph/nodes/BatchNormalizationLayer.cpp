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

#include "arm_compute/core/Logger.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename BatchBatchNormalizationLayer, typename TensorType, TargetHint target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input, arm_compute::ITensor *output, Tensor &mean, Tensor &var, Tensor &beta, Tensor &gamma, float epsilon)
{
    auto norm = arm_compute::support::cpp14::make_unique<BatchBatchNormalizationLayer>();
    norm->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output),
        dynamic_cast<TensorType *>(mean.set_target(target_hint)),
        dynamic_cast<TensorType *>(var.set_target(target_hint)),
        dynamic_cast<TensorType *>(beta.set_target(target_hint)),
        dynamic_cast<TensorType *>(gamma.set_target(target_hint)),
        epsilon);

    return std::move(norm);
}

template <TargetHint                    target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor *input, arm_compute::ITensor *output, Tensor &mean, Tensor &var, Tensor &beta, Tensor &gamma, float epsilon);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(arm_compute::ITensor *input, arm_compute::ITensor *output, Tensor &mean, Tensor &var, Tensor &beta, Tensor &gamma,
                                                                        float epsilon)
{
    return instantiate_function<arm_compute::CLBatchNormalizationLayer, arm_compute::ICLTensor, TargetHint::OPENCL>(input, output, mean, var, beta, gamma, epsilon);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(arm_compute::ITensor *input, arm_compute::ITensor *output, Tensor &mean, Tensor &var, Tensor &beta, Tensor &gamma, float epsilon)
{
    return instantiate_function<arm_compute::NEBatchNormalizationLayer, arm_compute::ITensor, TargetHint::NEON>(input, output, mean, var, beta, gamma, epsilon);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> BatchNormalizationLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr || input->tensor() == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr || output->tensor() == nullptr);

    std::unique_ptr<arm_compute::IFunction> func;
    _target_hint = ctx.hints().target_hint();

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();

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

    if(_target_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(in, out, _mean, _var, _beta, _gamma, _epsilon);
        ARM_COMPUTE_LOG("Instantiating CLBatchNormalizationLayer");
    }
    else
    {
        func = instantiate<TargetHint::NEON>(in, out, _mean, _var, _beta, _gamma, _epsilon);
        ARM_COMPUTE_LOG("Instantiating NEBatchNormalizationLayer");
    }

    ARM_COMPUTE_LOG(" Data Type: " << in->info()->data_type()
                    << " Input shape: " << in->info()->tensor_shape()
                    << " Output shape: " << out->info()->tensor_shape()
                    << std::endl);

    return func;
}