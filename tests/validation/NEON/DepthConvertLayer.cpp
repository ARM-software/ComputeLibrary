/*
 * Copyright (c) 2017-2019 ARM Limited.
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
const auto DepthConvertLayerQASYMM8toF16Dataset   = combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerQASYMM8toF32Dataset   = combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerU8toU16Dataset        = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U16));
const auto DepthConvertLayerU8toS16Dataset        = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
const auto DepthConvertLayerU8toS32Dataset        = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerU16toU8Dataset        = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerU16toU32Dataset       = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U32));
const auto DepthConvertLayerS16toU8Dataset        = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerS16toS32Dataset       = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerF16toF32Dataset       = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerF32toF16Dataset       = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F16));
const auto DepthConvertLayerF16toQASYMM8Dataset   = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::QASYMM8));
const auto DepthConvertLayerF32toQASYMM8Dataset   = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::QASYMM8));
const auto DepthConvertLayerShiftDatasetNightly   = framework::dataset::make("Shift", 0, 7);
const auto DepthConvertLayerShiftDatasetPrecommit = framework::dataset::make("Shift", { 0, 3, 6 });
const auto DepthConvertLayerZeroShiftDataset      = framework::dataset::make("Shift", 0);

constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthConvertLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U16),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),  // Invalid data type combination
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),  // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),  // Invalid shift
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
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
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, true})),
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
using NEDepthConvertLayerToF16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, half>;
template <typename T>
using NEDepthConvertLayerToF32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, float>;
template <typename T>
using NEDepthConvertLayerToQASYMM8Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint8_t>;
template <typename T>
using NEDepthConvertLayerQuantizedToF16Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, half>;
template <typename T>
using NEDepthConvertLayerQuantizedToF32Fixture = DepthConvertLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthConvertLayer, T, float>;

TEST_SUITE(QASYMM8_to_F32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::QASYMM8, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::F32, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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

TEST_SUITE(U8_to_U16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U16, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S16, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S32, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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

TEST_SUITE(U16_to_U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U32, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::S16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDatasetNightly),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::S16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S32, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16_to_QASYMM8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::F16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::QASYMM8, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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

TEST_SUITE(F16_to_F32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::F16, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::F32, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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

TEST_SUITE(QASYMM8_to_F16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::QASYMM8, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::F16, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::F32, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::F16, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32_to_QASYMM8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerZeroShiftDataset),
               shape, policy, shift)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::F32, 1);
    Tensor dst = create_tensor<Tensor>(shape, DataType::QASYMM8, 1);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

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
TEST_SUITE_END() // DepthConvertLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
