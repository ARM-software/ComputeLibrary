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
#include "src/gpu/cl/ClTensor.h"

#include "src/common/utils/LegacySupport.h"

namespace arm_compute
{
namespace gpu
{
namespace opencl
{
ClTensor::ClTensor(IContext *ctx, const AclTensorDescriptor &desc) : ITensorV2(ctx), _legacy_tensor()
{
    ARM_COMPUTE_ASSERT((ctx != nullptr) && (ctx->type() == Target::GpuOcl));
    _legacy_tensor = std::make_unique<CLTensor>();
    _legacy_tensor->allocator()->init(arm_compute::detail::convert_to_legacy_tensor_info(desc));
}

void *ClTensor::map()
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

    if (_legacy_tensor == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[ClTensor:map]: Backing tensor does not exist!");
        return nullptr;
    }

    _legacy_tensor->map();
    return _legacy_tensor->buffer();
}

StatusCode ClTensor::unmap()
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

    if (_legacy_tensor == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[ClTensor:unmap]: Backing tensor does not exist!");
        return StatusCode::RuntimeError;
    }
    _legacy_tensor->unmap();

    return StatusCode::Success;
}

StatusCode ClTensor::allocate()
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);

    _legacy_tensor->allocator()->allocate();
    return StatusCode::Success;
}

StatusCode ClTensor::import(void *handle, ImportMemoryType type)
{
    ARM_COMPUTE_ASSERT(_legacy_tensor.get() != nullptr);
    ARM_COMPUTE_UNUSED(type, handle);

    return StatusCode::Success;
}

arm_compute::ITensor *ClTensor::tensor() const
{
    return _legacy_tensor.get();
}
} // namespace opencl
} // namespace gpu
} // namespace arm_compute
