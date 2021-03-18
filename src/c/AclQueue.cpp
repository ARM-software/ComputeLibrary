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

#include "src/common/IQueue.h"
#include "src/common/utils/Macros.h"
#include "src/common/utils/Validate.h"

namespace
{
/** Check if queue options are valid
 *
 * @param[in] options Queue options
 *
 * @return true in case of success else false
 */
bool is_mode_valid(const AclQueueOptions *options)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(options);
    return arm_compute::utils::is_in(options->mode, { AclTuningModeNone, AclRapid, AclNormal, AclExhaustive });
}
} // namespace

extern "C" AclStatus AclCreateQueue(AclQueue *external_queue, AclContext external_ctx, const AclQueueOptions *options)
{
    using namespace arm_compute;

    auto ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if(options != nullptr && !is_mode_valid(options))
    {
        ARM_COMPUTE_LOG_ERROR_ACL("Queue options are invalid");
        return AclInvalidArgument;
    }

    auto queue = ctx->create_queue(options);
    if(queue == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("Couldn't allocate internal resources");
        return AclOutOfMemory;
    }

    *external_queue = queue;

    return AclSuccess;
}

extern "C" AclStatus AclQueueFinish(AclQueue external_queue)
{
    using namespace arm_compute;

    auto queue = get_internal(external_queue);

    StatusCode status = detail::validate_internal_queue(queue);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = queue->finish();
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return AclSuccess;
}

extern "C" AclStatus AclDestroyQueue(AclQueue external_queue)
{
    using namespace arm_compute;

    auto queue = get_internal(external_queue);

    StatusCode status = detail::validate_internal_queue(queue);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete queue;

    return AclSuccess;
}
