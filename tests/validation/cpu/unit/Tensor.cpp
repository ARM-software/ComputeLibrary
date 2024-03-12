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
#include "tests/validation/fixtures/UNIT/TensorFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CPU)
TEST_SUITE(UNIT)
TEST_SUITE(Tensor)

EMPTY_BODY_FIXTURE_TEST_CASE(CreateTensorWithInvalidContext, CreateTensorWithInvalidContextFixture, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(CreateTensorWithInvalidDescriptor, CreateTensorWithInvalidDescriptorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(DestroyInvalidTensor, DestroyInvalidTensorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(SimpleTensor, SimpleTensorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(TensorStress, TensorStressFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(MapInvalidTensor, MapInvalidTensorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(MapNotAllocatedTensor, MapNotAllocatedTensorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(MapAllocatedTensor, MapAllocatedTensorFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(ImportMemory, ImportMemoryFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(GetSize, TensorSizeFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(GetInvalidSize, InvalidTensorSizeFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(GetDescriptor, DescriptorConversionFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(GetInvalidDescriptor, InvalidDescriptorConversionFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)

TEST_SUITE_END() // Tensor
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CPU
} // namespace validation
} // namespace test
} // namespace arm_compute
