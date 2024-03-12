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

#include "src/cpu/CpuQueue.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CPU)
TEST_SUITE(UNIT)
TEST_SUITE(Queue)

EMPTY_BODY_FIXTURE_TEST_CASE(CreateQueueWithInvalidContext, CreateQueueWithInvalidContextFixture, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(CreateQueuerWithInvalidOptions, CreateQueuerWithInvalidOptionsFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(DestroyInvalidQueue, DestroyInvalidQueueFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(SimpleQueue, SimpleQueueFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)

TEST_SUITE_END() // Queue
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CPU
} // namespace validation
} // namespace test
} // namespace arm_compute
