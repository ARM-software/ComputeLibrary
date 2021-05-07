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

#include "src/common/IContext.h"
#include "src/common/utils/Macros.h"
#include "src/common/utils/Validate.h"

#ifdef ARM_COMPUTE_CPU_ENABLED
#include "src/cpu/CpuContext.h"
#endif /* ARM_COMPUTE_CPU_ENABLED */

#ifdef ARM_COMPUTE_OPENCL_ENABLED
#include "src/gpu/cl/ClContext.h"
#endif /* ARM_COMPUTE_OPENCL_ENABLED */

namespace
{
template <typename ContextType>
arm_compute::IContext *create_backend_ctx(const AclContextOptions *options)
{
    return new(std::nothrow) ContextType(options);
}

bool is_target_valid(AclTarget target)
{
    return arm_compute::utils::is_in(target, { AclCpu, AclGpuOcl });
}

bool are_context_options_valid(const AclContextOptions *options)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(options);
    return arm_compute::utils::is_in(options->mode, { AclPreferFastRerun, AclPreferFastStart });
}

arm_compute::IContext *create_context(AclTarget target, const AclContextOptions *options)
{
    ARM_COMPUTE_UNUSED(options);

    switch(target)
    {
#ifdef ARM_COMPUTE_CPU_ENABLED
        case AclCpu:
            return create_backend_ctx<arm_compute::cpu::CpuContext>(options);
#endif /* ARM_COMPUTE_CPU_ENABLED */
#ifdef ARM_COMPUTE_OPENCL_ENABLED
        case AclGpuOcl:
            return create_backend_ctx<arm_compute::gpu::opencl::ClContext>(options);
#endif /* ARM_COMPUTE_OPENCL_ENABLED */
        default:
            return nullptr;
    }
    return nullptr;
}
} // namespace

extern "C" AclStatus AclCreateContext(AclContext              *ctx,
                                      AclTarget                target,
                                      const AclContextOptions *options)
{
    if(!is_target_valid(target))
    {
        ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Target is invalid!");
        return AclUnsupportedTarget;
    }

    if(options != nullptr && !are_context_options_valid(options))
    {
        ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Context options are invalid!");
        return AclInvalidArgument;
    }

    auto acl_ctx = create_context(target, options);
    if(ctx == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Couldn't allocate internal resources for context creation!");
        return AclOutOfMemory;
    }
    *ctx = acl_ctx;

    return AclSuccess;
}

extern "C" AclStatus AclDestroyContext(AclContext external_ctx)
{
    using namespace arm_compute;

    IContext *ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if(ctx->refcount() != 0)
    {
        ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Context has references on it that haven't been released!");
        // TODO: Fix the refcount with callback when reaches 0
    }

    delete ctx;

    return utils::as_cenum<AclStatus>(status);
}
