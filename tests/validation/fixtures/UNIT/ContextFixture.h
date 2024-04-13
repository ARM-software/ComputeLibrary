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
#ifndef ARM_COMPUTE_TEST_UNIT_CONTEXT_FIXTURE
#define ARM_COMPUTE_TEST_UNIT_CONTEXT_FIXTURE

#include "arm_compute/Acl.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Test-case for AclDestroyContext
 *
 * Validate that AclDestroyContext behaves as expected when invalid inputs as context are given
 *
 * Test Steps:
 *  - Call AclDestroyContext with null context
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyContext on empty array
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyContext on an ACL object other than AclContext
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that context is still nullptr
 */
template <AclTarget Target>
class DestroyInvalidContextFixture : public framework::Fixture
{
public:
    void setup()
    {
        AclContext ctx = nullptr;
        std::array<char, 256> empty_array{};
        AclContext valid_ctx = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateContext(&valid_ctx, Target, nullptr) == AclStatus::AclSuccess);
        ARM_COMPUTE_ASSERT(AclDestroyContext(ctx) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyContext(reinterpret_cast<AclContext>(empty_array.data())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(ctx == nullptr);
        ARM_COMPUTE_ASSERT(AclDestroyContext(valid_ctx) == AclStatus::AclSuccess);
    };
};

/** Test-case for AclCreateContext and AclDestroyContext
 *
 * Validate that AclCreateContext can create and destroy a context through the C API
 *
 * Test Steps:
 *  - Call AclCreateContext with valid target
 *  - Confirm that context is not nullptr and error code is AclSuccess
 *  - Destroy context
 *  - Confirm that AclSuccess is reported
 */
template <AclTarget Target>
class SimpleContextCApiFixture : public framework::Fixture
{
public:
    void setup()
    {
        AclContext ctx = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateContext(&ctx, Target, nullptr) == AclStatus::AclSuccess);
        ARM_COMPUTE_ASSERT(ctx != nullptr);
        ARM_COMPUTE_ASSERT(AclDestroyContext(ctx) == AclStatus::AclSuccess);
    };
};

/** Test-case for Context from the C++ interface
 *
 * Test Steps:
 *  - Create a Context obejct
 *  - Confirm that StatusCode::Success is reported
 *  - Confirm that equality operator works
 *  - Confirm that inequality operator works
 */
template <acl::Target Target>
class SimpleContextCppApiFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode status = acl::StatusCode::Success;
        acl::Context    ctx(Target, &status);
        ARM_COMPUTE_ASSERT(status == acl::StatusCode::Success);

        auto ctx_eq = ctx;
        ARM_COMPUTE_ASSERT(ctx_eq == ctx);

        acl::Context ctx_ienq(Target, &status);
        ARM_COMPUTE_ASSERT(status == acl::StatusCode::Success);
        ARM_COMPUTE_ASSERT(ctx_ienq != ctx);
    };
};

/** Test-case for multiple contexes
 *
 * Validate that AclCreateContext can create/destroy multiple contexts with different options
 *
 * Test Steps:
 *  - Call AclCreateContext with different targets
 *  - Confirm that AclSuccess is reported
 *  - Destroy all contexts
 *  - Confirm that AclSuccess is reported
 */
template <AclTarget Target>
class MultipleContextsFixture : public framework::Fixture
{
public:
    void setup()
    {
        const unsigned int num_tests = 5;
        std::array<AclContext, num_tests> ctxs{};
        for(unsigned int i = 0; i < num_tests; ++i)
        {
            ARM_COMPUTE_ASSERT(AclCreateContext(&ctxs[i], Target, nullptr) == AclStatus::AclSuccess);
            ARM_COMPUTE_ASSERT(ctxs[i] != nullptr);
            ARM_COMPUTE_ASSERT(AclDestroyContext(ctxs[i]) == AclStatus::AclSuccess);
        }
    };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_CONTEXT_FIXTURE */
