/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthConvertLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets **/
const auto DepthConvertLayerQASYMM8toF16Dataset = combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerQASYMM8toF32Dataset = combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerQASYMM8toS32Dataset = combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerU8toU16Dataset      = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U16));
const auto DepthConvertLayerU8toS16Dataset      = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
const auto DepthConvertLayerU8toS32Dataset      = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerU8toF16Dataset      = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerU8toF32Dataset      = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerU16toU8Dataset      = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerU16toU32Dataset     = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U32));
const auto DepthConvertLayerS16toU8Dataset      = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerS16toS32Dataset     = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerBF16toF32Dataset    = combine(framework::dataset::make("DataType", DataType::BFLOAT16), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerF16toU8Dataset      = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerF16toF32Dataset     = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerF16toS32Dataset     = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerF32toF16Dataset     = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerF32toS32Dataset     = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerF32toU8Dataset      = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerF32toBF16Dataset    = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::BFLOAT16));

const auto DepthConvertLayerS32toF32Dataset     = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerS32toQASYMM8Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::QASYMM8));
const auto DepthConvertLayerS32toF16Dataset     = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerS32toU8Dataset      = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::U8));

const auto DepthConvertLayerF16toQASYMM8Dataset   = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::QASYMM8));
const auto DepthConvertLayerF32toQASYMM8Dataset   = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::QASYMM8));
const auto DepthConvertLayerShiftDatasetNightly   = framework::dataset::make("Shift", 0, 7);
const auto DepthConvertLayerShiftDatasetPrecommit = framework::dataset::make("Shift", { 0, 3, 6 });
const auto DepthConvertLayerZeroShiftDataset      = framework::dataset::make("Shift", 0);

constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
constexpr AbsoluteTolerance<int32_t> tolerance_one_int32(1);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<uint8_t> tolerance_one_uint8(1);
#endif /*  __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthConvertLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U16),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),  // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),  // Invalid shift
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                     })),
               framework::dataset::make("Policy",{ ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                     })),
               framework::dataset::make("Shift",{ 1, 1, 1, 1, 1, 1, 8, 1,
                                                     })),
               framework::dataset::make("Expected", { false, false, false, false, true})),
               input_info, output_info, policy, shift, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEDepthConvertLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), policy, shift)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEDepthConvertLayerToU16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint16_t>;
template <typename T>
using NEDepthConvertLayerToS16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, int16_t>;
template <typename T>
using NEDepthConvertLayerToS32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, int32_t>;
template <typename T>
using NEDepthConvertLayerToU8Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint8_t>;
template <typename T>
using NEDepthConvertLayerToU32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint32_t>;
template <typename T>
using NEDepthConvertLayerToBF16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, bfloat16>;
template <typename T>
using NEDepthConvertLayerToF16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, half>;
template <typename T>
using NEDepthConvertLayerToF32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, float>;
template <typename T>
using NEDepthConvertLayerToQASYMM8Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint8_t>;
template <typename T>
using NEDepthConvertLayerQuantizedToF16Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, half>;
template <typename T>
using NEDepthConvertLayerQuantizedToF32Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, float>;
template <typename T>
using NEDepthConvertLayerQuantizedToS32Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, int32_t>;

TEST_SUITE(QASYMM8_to_F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerQuantizedToF32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerQASYMM8toF32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerQuantizedToF32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                       DepthConvertLayerQASYMM8toF32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_to_F32

TEST_SUITE(QASYMM8_to_S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerQuantizedToS32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerQASYMM8toS32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerQuantizedToS32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                       DepthConvertLayerQASYMM8toS32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_to_S32

TEST_SUITE(U8_to_U16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8_to_U16

TEST_SUITE(U8_to_S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8_to_S16
TEST_SUITE(U8_to_S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8_to_S32

TEST_SUITE(U8_to_F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toF32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toF32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8_to_F32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(U8_to_F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toF16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toF16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8_to_F36
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(U16_to_U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16_to_U8

TEST_SUITE(U16_to_U32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                       DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16_to_U32

TEST_SUITE(S16_to_U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16_to_U8

TEST_SUITE(S16_to_S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDatasetPrecommit))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDatasetNightly))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16_to_S32

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16)
TEST_SUITE(BFLOAT16_to_F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF32Fixture<bfloat16>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerBF16toF32Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                       DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // BFLOAT16_to_F32

TEST_SUITE(F32_to_BFLOAT16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToBF16Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF32toBF16Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32_to_BFLOAT16
#endif           /* defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16) */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16_to_QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToQASYMM8Fixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       DepthConvertLayerF16toQASYMM8Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                                                                       DepthConvertLayerZeroShiftDataset),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToQASYMM8Fixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                     DepthConvertLayerF16toQASYMM8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                                                                     DepthConvertLayerZeroShiftDataset),
                                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // F16_to_QASYMM8

TEST_SUITE(F16_to_U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF16toU8Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                  DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_uint8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF16toU8Dataset),
                                                                                                                        framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_uint8);
}
TEST_SUITE_END() // F16_to_U8

TEST_SUITE(F16_to_F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF32Fixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF16toF32Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF32Fixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF16toF32Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                 DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16_to_F32

TEST_SUITE(F16_to_S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF16toS32Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF16toS32Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                 DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}

TEST_SUITE_END() // F16_to_S32

TEST_SUITE(QASYMM8_to_F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerQuantizedToF16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerQASYMM8toF16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerQuantizedToF16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                       DepthConvertLayerQASYMM8toF16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_to_F16

TEST_SUITE(F32_to_F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF16Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF32toF16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF16Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF32toF16Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                  DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32_to_F16

TEST_SUITE(S32_to_F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF16Fixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS32toF16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF16Fixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS32toF16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32_to_F16

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32_to_S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF32toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF32toS32Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                  DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}
TEST_SUITE_END() // F32_to_S32

TEST_SUITE(F32_to_U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerF32toU8Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerF32toU8Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                 DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_one_int32);
}
TEST_SUITE_END() // F32_to_U8

TEST_SUITE(F32_to_QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToQASYMM8Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                        DepthConvertLayerF32toQASYMM8Dataset),
                                                                                                                        framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                                                                        DepthConvertLayerZeroShiftDataset),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToQASYMM8Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                      DepthConvertLayerF32toQASYMM8Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                                                                      DepthConvertLayerZeroShiftDataset),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // F32_to_QASYMM8

TEST_SUITE(S32_to_F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToF32Fixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS32toF32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToF32Fixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS32toF32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32_to_F32

TEST_SUITE(S32_to_QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToQASYMM8Fixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerS32toQASYMM8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       DepthConvertLayerZeroShiftDataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToQASYMM8Fixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                        DepthConvertLayerS32toQASYMM8Dataset),
                                                                                                                        framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                                                                        DepthConvertLayerZeroShiftDataset),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // S32_to_QASYMM8

TEST_SUITE(S32_to_U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS32toU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS32toU8Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerZeroShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32_to_U8

TEST_SUITE_END() // DepthConvertLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
