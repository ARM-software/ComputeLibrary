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
#include "arm_compute/AclUtils.h"
#include "src/common/ITensorV2.h"
#include "src/common/utils/Macros.h"

namespace
{
using namespace arm_compute;
/**< Maximum allowed dimensions by Compute Library */
constexpr int32_t max_allowed_dims = 6;

/** Check if a descriptor is valid
 *
 * @param desc  Descriptor to validate
 *
 * @return true in case of success else false
 */
bool is_desc_valid(const AclTensorDescriptor &desc)
{
    if(desc.data_type > AclFloat32 || desc.data_type <= AclDataTypeUnknown)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclCreateTensor]: Unknown data type!");
        return false;
    }
    if(desc.ndims > max_allowed_dims)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclCreateTensor]: Dimensions surpass the maximum allowed value!");
        return false;
    }
    if(desc.ndims > 0 && desc.shape == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclCreateTensor]: Dimensions values are empty while dimensionality is > 0!");
        return false;
    }
    return true;
}

StatusCode convert_and_validate_tensor(AclTensor tensor, ITensorV2 **internal_tensor)
{
    *internal_tensor = get_internal(tensor);
    return detail::validate_internal_tensor(*internal_tensor);
}
} // namespace

extern "C" AclStatus AclCreateTensor(AclTensor                 *external_tensor,
                                     AclContext                 external_ctx,
                                     const AclTensorDescriptor *desc,
                                     bool                       allocate)
{
    using namespace arm_compute;

    IContext *ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if(desc == nullptr || !is_desc_valid(*desc))
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclCreateTensor]: Descriptor is invalid!");
        return AclInvalidArgument;
    }

    auto tensor = ctx->create_tensor(*desc, allocate);
    if(tensor == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclCreateTensor]: Couldn't allocate internal resources for tensor creation!");
        return AclOutOfMemory;
    }
    *external_tensor = tensor;

    return AclSuccess;
}

extern "C" AclStatus AclMapTensor(AclTensor external_tensor, void **handle)
{
    using namespace arm_compute;

    auto       tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if(handle == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[AclMapTensor]: Handle object is nullptr!");
        return AclInvalidArgument;
    }

    *handle = tensor->map();

    return AclSuccess;
}

extern "C" AclStatus AclUnmapTensor(AclTensor external_tensor, void *handle)
{
    ARM_COMPUTE_UNUSED(handle);

    using namespace arm_compute;

    auto       tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = tensor->unmap();
    return AclSuccess;
}

extern "C" AclStatus AclTensorImport(AclTensor external_tensor, void *handle, AclImportMemoryType type)
{
    using namespace arm_compute;

    auto       tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = tensor->import(handle, utils::as_enum<ImportMemoryType>(type));
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return AclSuccess;
}

extern "C" AclStatus AclDestroyTensor(AclTensor external_tensor)
{
    using namespace arm_compute;

    auto tensor = get_internal(external_tensor);

    StatusCode status = detail::validate_internal_tensor(tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete tensor;

    return AclSuccess;
}

extern "C" AclStatus AclGetTensorSize(AclTensor tensor, uint64_t *size)
{
    using namespace arm_compute;

    if(size == nullptr)
    {
        return AclStatus::AclInvalidArgument;
    }

    ITensorV2 *internal_tensor{ nullptr };
    auto       status = convert_and_validate_tensor(tensor, &internal_tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    *size = internal_tensor->get_size();
    return utils::as_cenum<AclStatus>(status);
}

extern "C" AclStatus AclGetTensorDescriptor(AclTensor tensor, AclTensorDescriptor *desc)
{
    using namespace arm_compute;

    if(desc == nullptr)
    {
        return AclStatus::AclInvalidArgument;
    }

    ITensorV2 *internal_tensor{ nullptr };
    const auto status = convert_and_validate_tensor(tensor, &internal_tensor);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    *desc = internal_tensor->get_descriptor();
    return utils::as_cenum<AclStatus>(status);
}