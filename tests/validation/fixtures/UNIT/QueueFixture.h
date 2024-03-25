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
#ifndef ARM_COMPUTE_TEST_UNIT_QUEUE_FIXTURE
#define ARM_COMPUTE_TEST_UNIT_QUEUE_FIXTURE

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
/** Test case for AclCreateQueue
 *
 * Validate that AclCreateQueue behaves as expected with invalid context
 *
 * Test Steps:
 *  - Call AclCreateQueue with an invalid context
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that the queue is still nullptr
 */
class CreateQueueWithInvalidContextFixture : public framework::Fixture
{
public:
    void setup()
    {
        AclQueue queue = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateQueue(&queue, nullptr, nullptr) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(queue == nullptr);
    };
};

/** Test-case for AclCreateQueue
 *
 * Validate that AclCreateQueue behaves as expected with invalid options
 *
 * Test Steps:
 *  - Call AclCreateQueue with valid context but invalid options
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that queue is still nullptr
 */
template <acl::Target Target>
class CreateQueuerWithInvalidOptionsFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context ctx(Target);

        // Check invalid tuning mode
        AclQueueOptions invalid_queue_opts;
        invalid_queue_opts.mode = static_cast<AclTuningMode>(-1);

        AclQueue queue = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateQueue(&queue, ctx.get(), &invalid_queue_opts) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(queue == nullptr);
    };
};

/** Test case for AclDestroyQueue
*
* Validate that AclDestroyQueue behaves as expected when an invalid queue is given
*
* Test Steps:
*  - Call AclDestroyQueue with null queue
*  - Confirm that AclInvalidArgument is reported
*  - Call AclDestroyQueue on empty array
*  - Confirm that AclInvalidArgument is reported
*  - Call AclDestroyQueue on an ACL object other than AclQueue
*  - Confirm that AclInvalidArgument is reported
*  - Confirm that queue is still nullptr
*/
template <acl::Target Target>
class DestroyInvalidQueueFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context ctx(Target);

        std::array<char, 256> empty_array{};
        AclQueue queue = nullptr;

        ARM_COMPUTE_ASSERT(AclDestroyQueue(queue) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyQueue(reinterpret_cast<AclQueue>(ctx.get())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyQueue(reinterpret_cast<AclQueue>(empty_array.data())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(queue == nullptr);
    };
};

/** Test case for AclCreateQueue
 *
 * Validate that a queue can be created successfully
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid queue
 *  - Confirm that AclSuccess is returned
 */
template <acl::Target Target>
class SimpleQueueFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        acl::Queue queue(ctx, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
    };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_QUEUE_FIXTURE */
