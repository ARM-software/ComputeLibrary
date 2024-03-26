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
#include "arm_compute/AclEntrypoints.h"

#include "src/common/ITensorV2.h"
#include "src/common/TensorPack.h"
#include "src/common/utils/Macros.h"

namespace
{
using namespace arm_compute;
StatusCode PackTensorInternal(TensorPack &pack, AclTensor external_tensor, int32_t slot_id)
{
    auto status = StatusCode::Success;
    auto tensor = get_internal(external_tensor);

    status = detail::validate_internal_tensor(tensor);

    if (status != StatusCode::Success)
    {
        return status;
    }

    pack.add_tensor(tensor, slot_id);

    return status;
}
} // namespace

extern "C" AclStatus AclCreateTensorPack(AclTensorPack *external_pack, AclContext external_ctx)
{
    using namespace arm_compute;

    IContext *ctx = get_internal(external_ctx);

    const StatusCode status = detail::validate_internal_context(ctx);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    auto pack = new TensorPack(ctx);
    if (pack == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Couldn't allocate internal resources!");
        return AclOutOfMemory;
    }
    *external_pack = pack;

    return AclSuccess;
}

extern "C" AclStatus AclPackTensor(AclTensorPack external_pack, AclTensor external_tensor, int32_t slot_id)
{
    using namespace arm_compute;

    auto pack = get_internal(external_pack);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(detail::validate_internal_pack(pack));
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(PackTensorInternal(*pack, external_tensor, slot_id));
    return AclStatus::AclSuccess;
}

extern "C" AclStatus
AclPackTensors(AclTensorPack external_pack, AclTensor *external_tensors, int32_t *slot_ids, size_t num_tensors)
{
    using namespace arm_compute;

    auto pack = get_internal(external_pack);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(detail::validate_internal_pack(pack));

    for (unsigned i = 0; i < num_tensors; ++i)
    {
        ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(PackTensorInternal(*pack, external_tensors[i], slot_ids[i]));
    }
    return AclStatus::AclSuccess;
}

extern "C" AclStatus AclDestroyTensorPack(AclTensorPack external_pack)
{
    using namespace arm_compute;

    auto       pack   = get_internal(external_pack);
    StatusCode status = detail::validate_internal_pack(pack);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete pack;

    return AclSuccess;
}
