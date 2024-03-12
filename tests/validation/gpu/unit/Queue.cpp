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
#include "tests/validation/fixtures/UNIT/QueueFixture.h"

#include "arm_compute/AclOpenClExt.h"
#include "src/gpu/cl/ClQueue.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(Queue)

EMPTY_BODY_FIXTURE_TEST_CASE(CreateQueueWithInvalidContext, CreateQueueWithInvalidContextFixture, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(CreateQueuerWithInvalidOptions, CreateQueuerWithInvalidOptionsFixture<acl::Target::GpuOcl>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(DestroyInvalidQueue, DestroyInvalidQueueFixture<acl::Target::GpuOcl>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(SimpleQueue, SimpleQueueFixture<acl::Target::GpuOcl>, framework::DatasetMode::ALL)

TEST_CASE(KhrQueuePriorities, framework::DatasetMode::ALL)
{
    acl::StatusCode err = acl::StatusCode::Success;

    acl::Context ctx(acl::Target::GpuOcl, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    acl::Queue queue(ctx, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    cl_device_id cl_dev;
    auto         status = AclGetClDevice(ctx.get(), &cl_dev);
    ARM_COMPUTE_ASSERT(status == AclSuccess);

    std::string extensions = cl::Device(cl_dev).getInfo<CL_DEVICE_EXTENSIONS>();
    if(extensions.find("cl_khr_priority_hints") != std::string::npos)
    {
        cl_int error = CL_SUCCESS;

        cl_context cl_ctx;
        auto       status = AclGetClContext(ctx.get(), &cl_ctx);
        ARM_COMPUTE_ASSERT(status == AclSuccess);

        /* Check a queue with high priority */
        cl_queue_properties queue_properties[] = { CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0 };
        cl_command_queue    priority_queue     = clCreateCommandQueueWithProperties(cl_ctx, cl_dev, queue_properties, &error);
        ARM_COMPUTE_ASSERT(error == CL_SUCCESS);

        clReleaseCommandQueue(priority_queue);
    }
}

TEST_SUITE_END() // Queue
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
