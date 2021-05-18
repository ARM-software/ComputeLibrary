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
#include "src/common/TensorPack.h"
#include "src/common/ITensorV2.h"
#include "src/common/utils/Validate.h"

namespace arm_compute
{
TensorPack::TensorPack(IContext *ctx)
    : AclTensorPack_(), _pack()
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(ctx);
    this->header.ctx = ctx;
    this->header.ctx->inc_ref();
}

TensorPack::~TensorPack()
{
    this->header.ctx->dec_ref();
    this->header.type = detail::ObjectType::Invalid;
}

AclStatus TensorPack::add_tensor(ITensorV2 *tensor, int32_t slot_id)
{
    _pack.add_tensor(slot_id, tensor->tensor());
    return AclStatus::AclSuccess;
}

size_t TensorPack::size() const
{
    return _pack.size();
}

bool TensorPack::empty() const
{
    return _pack.empty();
}

bool TensorPack::is_valid() const
{
    return this->header.type == detail::ObjectType::TensorPack;
}

arm_compute::ITensor *TensorPack::get_tensor(int32_t slot_id)
{
    return _pack.get_tensor(slot_id);
}

arm_compute::ITensorPack &TensorPack::get_tensor_pack()
{
    return _pack;
}
} // namespace arm_compute
