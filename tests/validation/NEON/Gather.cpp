/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGather.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/GatherDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GatherFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Gather)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),     // Invalid Indices data type
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),     // Invalid Indices dimensionality
                                                TensorInfo(TensorShape(5U, 5U, 5U, 5U, 5U), 1, DataType::F32),    // Invalid Input dimensionality
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F16),     // Mismatching data type input/output
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),     // Invalid positive axis value
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F16),     // Invalid negative axis value
        }),
        framework::dataset::make("IndicesInfo", {
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U8),
                                                TensorInfo(TensorShape(10U, 10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
                                                TensorInfo(TensorShape(10U), 1, DataType::U32),
        })),
        framework::dataset::make("OutputInfo", {
                                                TensorInfo(TensorShape(27U, 10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(10U, 27U), 1, DataType::F32),
                                                TensorInfo(TensorShape(10U, 27U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(10U, 5U, 5U, 5U, 5U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 27U), 1, DataType::F16),
        })),
        framework::dataset::make("Axis", {
                                            0,
                                            1,
                                            -2,
                                            0,
                                            1,
                                            0,
                                            1,
                                            2,
                                            -3,
        })),
        framework::dataset::make("Expected", { true, true, false, false, false, false, false, false })),
        input_info, indices_info, output_info, axis, expected)
{
    const Status status = NEGather::validate(&input_info.clone()->set_is_resizable(true), &indices_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), axis);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               combine(arm_compute::test::datasets::SmallGatherDataset(), framework::dataset::make("DataType", { DataType::F32 })),
               input_shape, indices_shape, axis, data_type)
{
    const uint32_t actual_axis = wrap_around(axis, static_cast<int>(input_shape.num_dimensions()));
    Tensor         src         = create_tensor<Tensor>(input_shape, data_type);
    Tensor         indices     = create_tensor<Tensor>(indices_shape, DataType::U32);
    TensorShape    dst_shape   = arm_compute::misc::shape_calculator::compute_gather_shape(input_shape, indices_shape, actual_axis);
    Tensor         dst         = create_tensor<Tensor>(dst_shape, data_type);

    // Create and Configure function
    NEGather gather;
    gather.configure(&src, &indices, &dst, axis);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(dst.info()->tensor_shape());
    validate(dst.info()->valid_region(), valid_region);
}

template <typename T>
using NEGatherFixture = GatherFixture<Tensor, Accessor, NEGather, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGatherFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGatherDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGatherFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGatherDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGatherFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGatherDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGatherFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGatherDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGatherFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGatherDataset(), framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGatherFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGatherDataset(), framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGatherFixture<uint16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGatherDataset(), framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGatherFixture<uint16_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGatherDataset(), framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE_END() // Gather
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
