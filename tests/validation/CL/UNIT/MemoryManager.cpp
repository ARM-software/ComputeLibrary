/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/UNIT/MemoryManagerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.05f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(MemoryManager)

using CLBlobMemoryManagerSimpleWithinFunctionLevelFixture = BlobMemoryManagerSimpleTestCaseFixture<CLTensor,
      CLAccessor,
      CLBufferAllocator,
      CLFullyConnectedLayer>;
FIXTURE_TEST_CASE(BlobMemoryManagerSimpleWithinFunctionLevel,
                  CLBlobMemoryManagerSimpleWithinFunctionLevelFixture,
                  framework::DatasetMode::ALL)
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
