/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEIm2Col.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Im2Col)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::U8),      // Unsupported data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QS8, 2),  // Mismatching fixed point
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("HasBias", { true, true, true, true, false })),
               framework::dataset::make("Expected", { false, false, false, false, true })),
               input_info, output_info, has_bias, expected)
{
    bool status = bool(NEIm2Col::validate(&input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
