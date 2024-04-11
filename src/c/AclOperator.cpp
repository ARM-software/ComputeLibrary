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

#include "src/common/IOperator.h"
#include "src/common/IQueue.h"
#include "src/common/TensorPack.h"
#include "src/common/utils/Macros.h"

extern "C" AclStatus AclRunOperator(AclOperator external_op, AclQueue external_queue, AclTensorPack external_tensors)
{
    using namespace arm_compute;

    auto op    = get_internal(external_op);
    auto queue = get_internal(external_queue);
    auto pack  = get_internal(external_tensors);

    StatusCode status = StatusCode::Success;
    status            = detail::validate_internal_operator(op);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);
    status = detail::validate_internal_queue(queue);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);
    status = detail::validate_internal_pack(pack);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = op->run(*queue, pack->get_tensor_pack());
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return AclSuccess;
}

extern "C" AclStatus AclDestroyOperator(AclOperator external_op)
{
    using namespace arm_compute;

    auto op = get_internal(external_op);

    StatusCode status = detail::validate_internal_operator(op);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete op;

    return AclSuccess;
}
