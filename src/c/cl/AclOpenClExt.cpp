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
#include "arm_compute/AclOpenClExt.h"

#include "src/common/ITensorV2.h"
#include "src/common/Types.h"
#include "src/gpu/cl/ClContext.h"

#include "arm_compute/core/CL/ICLTensor.h"

#include "support/Cast.h"

extern "C" AclStatus AclGetClContext(AclContext external_ctx, cl_context *opencl_context)
{
    using namespace arm_compute;
    IContext *ctx = get_internal(external_ctx);

    if(detail::validate_internal_context(ctx) != StatusCode::Success)
    {
        return AclStatus::AclInvalidArgument;
    }

    if(ctx->type() != Target::GpuOcl)
    {
        return AclStatus::AclInvalidTarget;
    }

    if(opencl_context == nullptr)
    {
        return AclStatus::AclInvalidArgument;
    }

    *opencl_context = utils::cast::polymorphic_downcast<arm_compute::gpu::opencl::ClContext *>(ctx)->cl_ctx().get();

    return AclStatus::AclSuccess;
}

extern "C" AclStatus AclSetClContext(AclContext external_ctx, cl_context opencl_context)
{
    using namespace arm_compute;
    IContext *ctx = get_internal(external_ctx);

    if(detail::validate_internal_context(ctx) != StatusCode::Success)
    {
        return AclStatus::AclInvalidArgument;
    }

    if(ctx->type() != Target::GpuOcl)
    {
        return AclStatus::AclInvalidTarget;
    }

    if(ctx->refcount() != 0)
    {
        return AclStatus::AclUnsupportedConfig;
    }

    auto cl_ctx = utils::cast::polymorphic_downcast<arm_compute::gpu::opencl::ClContext *>(ctx);
    if(!cl_ctx->set_cl_ctx(::cl::Context(opencl_context)))
    {
        return AclStatus::AclRuntimeError;
    }

    return AclStatus::AclSuccess;
}

extern "C" AclStatus AclGetClMem(AclTensor external_tensor, cl_mem *opencl_mem)
{
    using namespace arm_compute;
    ITensorV2 *tensor = get_internal(external_tensor);

    if(detail::validate_internal_tensor(tensor) != StatusCode::Success)
    {
        return AclStatus::AclInvalidArgument;
    }

    if(tensor->header.ctx->type() != Target::GpuOcl)
    {
        return AclStatus::AclInvalidTarget;
    }

    if(opencl_mem == nullptr)
    {
        return AclStatus::AclInvalidArgument;
    }

    auto cl_tensor = utils::cast::polymorphic_downcast<arm_compute::ICLTensor *>(tensor->tensor());
    *opencl_mem    = cl_tensor->cl_buffer().get();

    return AclStatus::AclSuccess;
}