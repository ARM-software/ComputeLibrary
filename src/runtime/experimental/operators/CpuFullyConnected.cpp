/*
 * Copyright (c) 2025 Arm Limited.
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
#include "src/cpu/operators/CpuFullyConnected.h"

#include "arm_compute/runtime/experimental/operators/CpuFullyConnected.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
using namespace arm_compute::experimental;

struct CpuFullyConnected::Impl
{
    std::unique_ptr<cpu::CpuFullyConnected> op{nullptr};
};

CpuFullyConnected::CpuFullyConnected() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuFullyConnected>();
}

CpuFullyConnected::~CpuFullyConnected() = default;

void CpuFullyConnected::configure(const ITensorInfo      *src,
                                  const ITensorInfo      *weights,
                                  const ITensorInfo      *biases,
                                  ITensorInfo            *dst,
                                  FullyConnectedLayerInfo fc_info,
                                  const WeightsInfo      &weights_info)
{
    _impl->op->configure(src, weights, biases, dst, fc_info, weights_info);
}
Status CpuFullyConnected::validate(const ITensorInfo      *src,
                                   const ITensorInfo      *weights,
                                   const ITensorInfo      *biases,
                                   const ITensorInfo      *dst,
                                   FullyConnectedLayerInfo fc_info,
                                   const WeightsInfo      &weights_info)
{
    bool fp32_ok = src->data_type() == DataType::F32 && weights->data_type() == DataType::F32 &&
                   (biases->data_type() == DataType::F32 || biases == nullptr) && dst->data_type() == DataType::F32;
    bool fp16_ok = src->data_type() == DataType::F16 && weights->data_type() == DataType::F16 &&
                   (biases->data_type() == DataType::F16 || biases == nullptr) && dst->data_type() == DataType::F16;
    if (!(fp32_ok || fp16_ok))
    {
        return Status(ErrorCode::RUNTIME_ERROR, "datatype is not supported");
    }
    if (!arm_compute::is_fixed_format(weights_info.weight_format()))
    {
        return Status(ErrorCode::RUNTIME_ERROR, "only support fixed format weight");
    }
    if (fc_info.transpose_weights)
    {
        return Status(ErrorCode::RUNTIME_ERROR, "transpose weight is not supported");
    }
    const bool is_batched_fc_layer = dst->dimension(1) > 1;
    bool       is_fc_after_conv    = true;
    if (is_batched_fc_layer)
    {
        is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) &&
                           (std::equal(src->tensor_shape().cbegin() + 3, src->tensor_shape().cend(),
                                       dst->tensor_shape().cbegin() + 1));
    }
    else
    {
        is_fc_after_conv = src->num_dimensions() > 1;
    }
    if (is_fc_after_conv)
    {
        return Status(ErrorCode::RUNTIME_ERROR, "only support fully connected layer after fully connected layer");
    }
    return cpu::CpuFullyConnected::validate(src, weights, biases, dst, fc_info, weights_info);
}
Status CpuFullyConnected::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                                       const ITensorInfo         *src,
                                       const ITensorInfo         *weights,
                                       const ITensorInfo         *biases,
                                       const ITensorInfo         *dst,
                                       FullyConnectedLayerInfo    fc_info,
                                       WeightsInfo                weights_info)
{
    return cpu::CpuFullyConnected::has_opt_impl(expected_weight_format, src, weights, biases, dst, fc_info,
                                                weights_info);
}

void CpuFullyConnected::run(ITensorPack &tensors)
{
    _impl->op->run(tensors);
}

void CpuFullyConnected::prepare(ITensorPack &tensors)
{
    _impl->op->prepare(tensors);
}

experimental::MemoryRequirements CpuFullyConnected::workspace() const
{
    return _impl->op->workspace();
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
