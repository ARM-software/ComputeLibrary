/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include "arm_compute/core/SubTensorInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(SubTensorInfo)

/** Validate sub-tensor creation
 *
 * Test performed:
 *
 *  - Negative testing on X indexing
 *  - Negative testing on Y indexing
 *  - Positive testing by indexing on X,Y indexing
 * */
TEST_CASE(SubTensorCreation, framework::DatasetMode::ALL)
{
    // Create tensor info
    TensorInfo info(TensorShape(23U, 17U, 3U), 1, DataType::F32);

    // Negative testing on X
    ARM_COMPUTE_EXPECT_THROW(SubTensorInfo(&info, TensorShape(13U, 17U, 3U), Coordinates(24, 0, 0)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_THROW(SubTensorInfo(&info, TensorShape(13U, 17U, 3U), Coordinates(15, 0, 0)), framework::LogLevel::ERRORS);

    // Negative testing on Y
    ARM_COMPUTE_EXPECT_THROW(SubTensorInfo(&info, TensorShape(23U, 8U, 3U), Coordinates(0, 18, 0)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_THROW(SubTensorInfo(&info, TensorShape(23U, 8U, 3U), Coordinates(0, 13, 0)), framework::LogLevel::ERRORS);

    // Positive testing on XY indexing
    ARM_COMPUTE_EXPECT_NO_THROW(SubTensorInfo(&info, TensorShape(4U, 3U, 2U), Coordinates(5, 2, 1)), framework::LogLevel::ERRORS);
}

/** Validate when extending padding on sub-tensor
 *
 * Tests performed:
 *  - A) Extend padding when SubTensor XY does not match parent tensor should fail
 *    B) Extend with zero padding when SubTensor XY does not match parent tensor should succeed
 *  - C) Extend padding when SubTensor XY matches parent tensor should succeed
 *  - D) Set lock padding to true, so that extend padding would fail
 */
TEST_CASE(SubTensorPaddingExpansion, framework::DatasetMode::ALL)
{
    // Test A
    {
        TensorInfo    tensor_info(TensorShape(23U, 17U, 3U), 1, DataType::F32);
        SubTensorInfo sub_tensor_info(&tensor_info, TensorShape(4U, 3U, 2U), Coordinates(5, 2, 1));
        ARM_COMPUTE_EXPECT_THROW(sub_tensor_info.extend_padding(PaddingSize(2, 1)), framework::LogLevel::ERRORS);
    }

    // Test B
    {
        TensorInfo    tensor_info(TensorShape(23U, 17U, 3U), 1, DataType::F32);
        SubTensorInfo sub_tensor_info(&tensor_info, TensorShape(4U, 3U, 1U), Coordinates(5, 2, 1));
        ARM_COMPUTE_EXPECT_NO_THROW(sub_tensor_info.extend_padding(PaddingSize(0, 0)), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensor_info.padding().uniform(), framework::LogLevel::ERRORS);
    }

    // Test C
    {
        TensorInfo    tensor_info(TensorShape(23U, 17U, 3U), 1, DataType::F32);
        SubTensorInfo sub_tensor_info(&tensor_info, TensorShape(23U, 17U, 1U), Coordinates(0, 0, 1));
        ARM_COMPUTE_EXPECT_NO_THROW(sub_tensor_info.extend_padding(PaddingSize(2, 1)), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensor_info.padding().top == 2, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensor_info.padding().right == 1, framework::LogLevel::ERRORS);
    }

    // Test D
    {
        TensorInfo    tensor_info(TensorShape(23U, 17U, 3U), 1, DataType::F32);
        SubTensorInfo sub_tensor_info(&tensor_info, TensorShape(4U, 3U, 1U), Coordinates(5, 2, 1));
        sub_tensor_info.set_lock_paddings(true);
        ARM_COMPUTE_EXPECT_THROW(sub_tensor_info.extend_padding(PaddingSize(2, 1)), framework::LogLevel::ERRORS);
    }
}

TEST_SUITE_END() // SubTensorInfo
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
