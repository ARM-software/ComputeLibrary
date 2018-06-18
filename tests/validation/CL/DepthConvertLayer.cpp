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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLDepthConvertLayer.h"
#include "tests/CL/CLAccessor.h"
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
const auto DepthConvertLayerU8toU16Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U16));
const auto DepthConvertLayerU8toS16Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
const auto DepthConvertLayerU8toS32Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerU16toU8Dataset             = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerU16toU32Dataset            = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U32));
const auto DepthConvertLayerS16toU8Dataset             = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerS16toS32Dataset            = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerShiftDataset               = framework::dataset::make("Shift", 0, 7);
const auto DepthConvertLayerFixedPointQuantizedDataset = framework::dataset::make("FractionalBits", 1, 7);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthConvertLayer)
template <typename T>
using CLDepthConvertLayerToU16Fixture = DepthConvertLayerValidationFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, uint16_t>;
template <typename T>
using CLDepthConvertLayerToS16Fixture = DepthConvertLayerValidationFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, int16_t>;
template <typename T>
using CLDepthConvertLayerToS32Fixture = DepthConvertLayerValidationFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, int32_t>;
template <typename T>
using CLDepthConvertLayerToU8Fixture = DepthConvertLayerValidationFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, uint8_t>;
template <typename T>
using CLDepthConvertLayerToU32Fixture = DepthConvertLayerValidationFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, uint32_t>;
template <typename T>
using CLDepthConvertLayerToFP32FixedPointFixture = DepthConvertLayerValidationFractionalBitsFixture<CLTensor, CLAccessor, CLDepthConvertLayer, T, float>;

TEST_SUITE(U8_to_U16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U16, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U8_to_S16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::S16, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE(U8_to_S32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::S32, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16_to_U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U16, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U8, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16_to_U32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U16, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U32, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                       DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16_to_U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::S16, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U8, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16_to_S32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::S16, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::S32, 1, fixed_point_position);

    // Create and Configure function
    CLDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
