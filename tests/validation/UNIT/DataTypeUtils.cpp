/*
 * Copyright (c) 2025 Arm Limited.
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

#include "arm_compute/core/utils/DataTypeUtils.h"

#include "tests/datasets/DatatypeDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/CartesianProductDataset.h"

#include <string>

namespace arm_compute
{
namespace test
{
namespace validation
{

TEST_SUITE(UNIT)
TEST_SUITE(DataTypeUtils)

// Whenever a new data type is added, this test will ensure string conversion
// explicitly deals with that data type
DATA_TEST_CASE(CheckDataTypeIsPrinted, framework::DatasetMode::ALL,
    datasets::AllDataTypes("DataType"), dtype)
{
    ARM_COMPUTE_EXPECT(string_from_data_type(dtype) != "",
        framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // DataTypeUtils
TEST_SUITE_END() // UNIT

} // namespace validation
} // namespace test
} // namespace arm_compute
