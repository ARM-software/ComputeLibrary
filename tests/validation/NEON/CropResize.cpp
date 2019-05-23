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
#include "arm_compute/runtime/NEON/functions/NECropResize.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/datasets/CropResizeDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CropResizeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(CropResize)

RelativeTolerance<float> tolerance_fp32(0.001f);

template <typename T>
using NECropResizeFixture = CropResizeFixture<Tensor, Accessor, NECropResize, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::U8),  // Invalid input data type.
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32), // Invalid box_ind shape.
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32), // Invalid output shape.
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32), // Invalid output data type.
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32), // Invalid output shape.
                                                       TensorInfo(TensorShape(15U, 30U, 40U, 10U), 1, DataType::S32), // Invalid boxes shape.
                                                     }),
               framework::dataset::make("BoxesInfo",{  TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4, 20), 1, DataType::F32),
                                                       TensorInfo(TensorShape(3, 20), 1, DataType::F32),
                                                     })),
               framework::dataset::make("BoxIndInfo",{ TensorInfo(TensorShape(20), 1, DataType::S32),
                                                       TensorInfo(TensorShape(20), 1, DataType::S32),
                                                       TensorInfo(TensorShape(10), 1, DataType::S32),
                                                       TensorInfo(TensorShape(20), 1, DataType::S32),
                                                       TensorInfo(TensorShape(20), 1, DataType::S32),
                                                       TensorInfo(TensorShape(20), 1, DataType::S32),
                                                       TensorInfo(TensorShape(20), 1, DataType::S32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(15U, 5, 5, 20U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(15U, 5, 5, 20U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(15U, 5, 5, 20U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(15U, 5, 5, 10U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(15U, 5, 5, 20U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(5U, 5, 5, 20U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(15U, 5, 5, 20U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, false, false, false, false, false, false})),
               input, boxes, box_ind, output, expected)
{
    ARM_COMPUTE_EXPECT(bool(NECropResize::validate(&input.clone()->set_data_layout(DataLayout::NHWC).set_is_resizable(false),
                                                   &boxes.clone()->set_is_resizable(false),
                                                   &box_ind.clone()->set_is_resizable(false),
                                                   &output.clone()->set_data_layout(DataLayout::NHWC).set_is_resizable(false),
                                                   Coordinates2D{ 5, 5 }, InterpolationPolicy::BILINEAR, 100)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<uint16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::U16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // U16

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::S16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // S16

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<uint32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::U32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // U32

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NECropResizeFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallCropResizeDataset(),
                               combine(framework::dataset::make("IsOutOfBounds", { true, false }),
                                       framework::dataset::make("DataType", DataType::S32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // S32

TEST_SUITE_END() // CropResize
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
