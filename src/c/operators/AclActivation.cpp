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
#include "arm_compute/AclOperators.h"

#include "src/common/IOperator.h"
#include "src/common/utils/Macros.h"
#include "src/common/utils/Validate.h"

extern "C" AclStatus AclActivation(AclOperator                  *external_op,
                                   AclContext                    external_ctx,
                                   const AclTensorDescriptor    *src,
                                   const AclTensorDescriptor    *dst,
                                   const AclActivationDescriptor info)
{
    using namespace arm_compute;

    // Extract internal context
    auto       ctx    = get_internal(external_ctx);
    StatusCode status = detail::validate_internal_context(ctx);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    bool is_validate = (external_op == ARM_COMPUTE_VALIDATE_OPERATOR_SUPPORT);

    std::tie(*external_op, status) = ctx->create_activation(*src, *dst, info, is_validate);
    ARM_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return AclSuccess;
}