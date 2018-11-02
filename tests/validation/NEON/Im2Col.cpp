/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Im2ColFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto conv_args = combine(combine(combine(framework::dataset::make("KernelDims", { Size2D(3U, 3U), Size2D(5U, 5U) }), framework::dataset::make("PadStride", { PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(2U, 2U, 0U, 2U) })),
                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Im2Col)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::U8),      // Unsupported data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QS8, 2),  // Mismatching fixed point
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Mismatching shapes
                                                       TensorInfo(TensorShape(10U, 12U, 2U, 2U), 1, DataType::QASYMM8),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(18U, 80U, 1U, 2U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("HasBias", { true, true, true, true, false, false })),
               framework::dataset::make("Expected", { false, false, false, false, false, true })),
               input_info, output_info, has_bias, expected)
{
    bool status = bool(NEIm2Col::validate(&input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias, false, false));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEIm2ColFixture = Im2ColValidationFixture<Tensor, Accessor, NEIm2Col, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                              conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F16)),
                                                                                             conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE_END()

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
