/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/cpu/CpuTensor.h"

#include "src/common/utils/LegacySupport.h"

namespace arm_compute
{
namespace cpu
{
CpuTensor::CpuTensor(IContext *ctx, const AclTensorDescriptor &desc)
    : ITensorV2(ctx), _legacy_tensor()
{
    ARM_COMPUTE_ASSERT((ctx != nullptr) && (ctx->type() == Target::Cpu));
    _legacy_tensor = std::make_unique<Tensor>();
    _legacy_tensor->allocator()->init(arm_compute::detail::convert_to_legacy_tensor_info(desc));
}

void *CpuTensor::map()
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

    if(_legacy_tensor == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[CpuTensor:map]: Backing tensor does not exist!");
        return nullptr;
    }
    return _legacy_tensor->buffer();
}

StatusCode CpuTensor::allocate()
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

    _legacy_tensor->allocator()->allocate();
    return StatusCode::Success;
}

StatusCode CpuTensor::unmap()
{
    // No-op
    return StatusCode::Success;
}

StatusCode CpuTensor::import(void *handle, ImportMemoryType type)
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);
    ARM_COMPUTE_UNUSED(type);

    const auto st = _legacy_tensor->allocator()->import_memory(handle);
    return bool(st) ? StatusCode::Success : StatusCode::RuntimeError;
}

arm_compute::ITensor *CpuTensor::tensor() const
{
    return _legacy_tensor.get();
}
} // namespace cpu
} // namespace arm_compute
